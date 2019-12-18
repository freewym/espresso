# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.options import eval_str_list

from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

from espresso.tools import wer


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dict = task.target_dictionary
        self.scorer = wer.Scorer(dict, wer_output_filter=task.args.wer_output_filter)
        self.train_tgt_dataset = None
        self.valid_tgt_dataset = None
        self.num_updates = -1
        self.epoch = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--print-training-sample-interval', type=int,
                            metavar='N', dest='print_interval', default=500,
                            help='print a training sample (reference + '
                                 'prediction) every this number of updates')
        parser.add_argument('--scheduled-sampling-probs', type=lambda p: eval_str_list(p),
                            metavar='P_1,P_2,...,P_N', default=1.0,
                            help='schedule sampling probabilities of sampling the truth '
                            'labels for N epochs starting from --start-schedule-sampling-epoch; '
                            'all later epochs using P_N')
        parser.add_argument('--start-scheduled-sampling-epoch', type=int,
                            metavar='N', default=1,
                            help='start schedule sampling from the specified epoch')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample; periodically print out
        randomly sampled predictions if model is in training mode, otherwise
        aggregate word error stats for validation.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        dict = self.scorer.dict
        if model.training:
            if (
                (len(self.args.scheduled_sampling_probs) > 1 or
                 self.args.scheduled_sampling_probs[0] < 1.0) and
                self.epoch >= self.args.start_scheduled_sampling_epoch
            ):
                # scheduled sampling
                ss_prob = self.args.scheduled_sampling_probs[
                    min(self.epoch - self.args.start_scheduled_sampling_epoch,
                        len(self.args.scheduled_sampling_probs) - 1)
                ]
                assert isinstance(model.decoder, FairseqIncrementalDecoder)
                incremental_states = {}
                encoder_input = {
                    k: v for k, v in sample['net_input'].items()
                    if k != 'prev_output_tokens'
                }
                encoder_out = model.encoder(**encoder_input)
                target = sample['target']
                tokens = sample['net_input']['prev_output_tokens']
                lprobs = []
                pred = None
                for step in range(target.size(1)):
                    if step > 0:
                        sampling_mask = torch.rand(
                            [target.size(0), 1],
                            device=target.device,
                        ).lt(ss_prob)
                        feed_tokens = torch.where(
                            sampling_mask, tokens[:, step:step + 1], pred,
                        )
                    else:
                        feed_tokens = tokens[:, step:step + 1]
                    log_probs, _ = self._decode(
                        feed_tokens, model, encoder_out, incremental_states,
                    )
                    pred = log_probs.argmax(-1, keepdim=True)
                    lprobs.append(log_probs)
                lprobs = torch.stack(lprobs, dim=1)
            else:
                # normal training
                net_output = model(**sample['net_input'])
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                target = model.get_targets(sample, net_output)
        else:
            assert isinstance(model.decoder, FairseqIncrementalDecoder)
            incremental_states = {}
            encoder_input = {
                k: v for k, v in sample['net_input'].items()
                if k != 'prev_output_tokens'
            }
            encoder_out = model.encoder(**encoder_input)
            target = sample['target']
            # make the maximum decoding length equal to at least the length of
            # target, and the length of encoder_out if possible
            maxlen = max(encoder_out['encoder_out'][0].size(1), target.size(1))
            tokens = target.new_full([target.size(0), maxlen + 2], self.padding_idx)
            tokens[:, 0] = dict.eos()
            lprobs = []
            attn = [] if getattr(model.decoder, 'need_attn', False) else None
            dummy_log_probs = encoder_out['encoder_out'][0].new_full(
                [target.size(0), len(dict)], -np.log(len(dict)))
            for step in range(maxlen + 1):  # one extra step for EOS marker
                is_eos = tokens[:, step].eq(dict.eos())
                # if all predictions are finished (i.e., ended with eos),
                # pad lprobs to target length with dummy log probs,
                # truncate tokens up to this step and break
                if step > 0 and is_eos.sum() == is_eos.size(0):
                    for _ in range(step, target.size(1)):
                        lprobs.append(dummy_log_probs)
                    tokens = tokens[:, :step + 1]
                    break
                log_probs, attn_scores = self._decode(
                    tokens[:, :step + 1], model, encoder_out, incremental_states,
                )
                tokens[:, step + 1] = log_probs.argmax(-1)
                if step > 0:  # deal with finished predictions
                    # make log_probs uniform if the previous output token is EOS
                    # and add consecutive EOS to the end of prediction
                    log_probs[is_eos, :] = -np.log(log_probs.size(1))
                    tokens[is_eos, step + 1] = dict.eos()
                if step < target.size(1):
                    lprobs.append(log_probs)
                if getattr(model.decoder, 'need_attn', False):
                    attn.append(attn_scores)
            # bsz x min(tgtlen, maxlen + 1) x vocab_size
            lprobs = torch.stack(lprobs, dim=1)
            if getattr(model.decoder, 'need_attn', False):
                # bsz x (maxlen + 1) x (length of encoder_out)
                attn = torch.stack(attn, dim=1)
        # word error stats code starts
        if (
            not model.training or
            (
                self.num_updates // self.args.print_interval >
                (self.num_updates - 1) // self.args.print_interval
            )
        ):
            pred = lprobs.argmax(-1).cpu() if model.training else \
                tokens[:, 1:].data.cpu()  # bsz x len

            if not model.training:  # validation step, compute WER stats with scorer
                assert pred.size(0) == target.size(0)
                self.scorer.reset()
                for i in range(target.size(0)):
                    utt_id = sample['utt_id'][i]
                    id = sample['id'].data[i].item()
                    # ref_tokens = dict.string(target.data[i])
                    # if it is a dummy batch (e.g., a "padding" batch in a sharded
                    # dataset), id might exceeds the dataset size; in that case we
                    # just skip it
                    if id < len(self.valid_tgt_dataset):
                        ref_tokens = self.valid_tgt_dataset.get_original_tokens(id)
                        pred_tokens = dict.string(pred.data[i])
                        self.scorer.add_evaluation(
                            utt_id, ref_tokens, pred_tokens,
                            bpe_symbol=self.args.remove_bpe,
                        )
            else:  # print a randomly sampled result every print_interval updates
                assert pred.size() == target.size()
                with data_utils.numpy_seed(self.num_updates):
                    i = np.random.randint(0, len(sample['id']))
                id = sample['id'].data[i].item()
                length = utils.strip_pad(target.data[i], self.padding_idx).size(0)
                # ref_one = dict.tokens_to_sentence(dict.string(target.data[i]))
                ref_one = self.train_tgt_dataset.get_original_text(
                    id, dict, bpe_symbol=self.args.remove_bpe,
                )
                pred_one = dict.tokens_to_sentence(
                    dict.string(pred.data[i][:length]),
                    bpe_symbol=self.args.remove_bpe,
                )
                print('| sample REF: ' + ref_one)
                print('| sample PRD: ' + pred_one)
        # word error stats code ends
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if not model.training:  # do not compute word error in training mode
            logging_output['word_error'] = self.scorer.tot_word_error()
            logging_output['word_count'] = self.scorer.tot_word_count()
            logging_output['char_error'] = self.scorer.tot_char_error()
            logging_output['char_count'] = self.scorer.tot_char_count()
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_output = CrossEntropyCriterion.aggregate_logging_outputs(logging_outputs)
        word_error = sum(log.get('word_error', 0) for log in logging_outputs)
        word_count = sum(log.get('word_count', 0) for log in logging_outputs)
        char_error = sum(log.get('char_error', 0) for log in logging_outputs)
        char_count = sum(log.get('char_count', 0) for log in logging_outputs)
        if word_count > 0:  # model.training == False
            agg_output['word_error'] = word_error
            agg_output['word_count'] = word_count
        if char_count > 0:  # model.training == False
            agg_output['char_error'] = char_error
            agg_output['char_count'] = char_count
        return agg_output

    def _decode(self, tokens, model, encoder_out, incremental_states):
        decoder_out = list(model.forward_decoder(
            tokens, encoder_out=encoder_out, incremental_state=incremental_states,
        ))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=True)
        probs = probs[:, -1, :]
        return probs, attn

    def set_train_tgt_dataset(self, dataset):
        self.train_tgt_dataset = dataset

    def set_valid_tgt_dataset(self, dataset):
        self.valid_tgt_dataset = dataset

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def set_epoch(self, epoch):
        self.epoch = epoch
