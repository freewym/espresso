# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq import utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

from espresso.tools import wer


def label_smoothed_nll_loss(
    lprobs, target, epsilon, ignore_index=None, reduce=True,
    smoothing_type='uniform', prob_mask=None, unigram_tensor=None,
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if smoothing_type == 'temporal':
        assert torch.is_tensor(prob_mask)
        smooth_loss = -lprobs.mul(prob_mask).sum(-1, keepdim=True)
    elif smoothing_type == 'unigram':
        assert torch.is_tensor(unigram_tensor)
        smooth_loss = -lprobs.matmul(unigram_tensor.to(lprobs))
    elif smoothing_type == 'uniform':
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    else:
        raise ValueError('Unsupported smoothing type: {}'.format(smoothing_type))
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1) if smoothing_type == 'uniform' else epsilon
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_with_wer')
class LabelSmoothedCrossEntropyWithWERCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dictionary = task.target_dictionary
        self.scorer = wer.Scorer(dictionary, wer_output_filter=task.args.wer_output_filter)
        self.train_tgt_dataset = None
        self.valid_tgt_dataset = None
        self.num_updates = -1
        self.epoch = 0
        self.unigram_tensor = None
        if args.smoothing_type == 'unigram':
            self.unigram_tensor = torch.cuda.FloatTensor(dictionary.count).unsqueeze(-1) \
                if torch.cuda.is_available() and not args.cpu \
                else torch.FloatTensor(dictionary.count).unsqueeze(-1)
            self.unigram_tensor += args.unigram_pseudo_count  # for further backoff
            self.unigram_tensor.div_(self.unigram_tensor.sum())

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--print-training-sample-interval', type=int,
                            metavar='N', dest='print_interval', default=500,
                            help='print a training sample (reference + '
                                 'prediction) every this number of updates')
        parser.add_argument('--smoothing-type', type=str, default='uniform',
                            choices=['uniform', 'unigram', 'temporal'],
                            help='label smoothing type. Default: uniform')
        parser.add_argument('--unigram-pseudo-count', type=float, default=1.0,
                            metavar='C', help='pseudo count for unigram label '
                            'smoothing. Only relevant if --smoothing-type=unigram')
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
        dictionary = self.scorer.dictionary
        if model.training:
            net_output = model(**sample['net_input'], epoch=self.epoch)
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
            tokens[:, 0] = dictionary.eos()
            lprobs = []
            attn = [] if getattr(model.decoder, 'need_attn', False) else None
            dummy_log_probs = encoder_out['encoder_out'][0].new_full(
                [target.size(0), len(dictionary)], -np.log(len(dictionary)))
            for step in range(maxlen + 1):  # one extra step for EOS marker
                is_eos = tokens[:, step].eq(dictionary.eos())
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
                    tokens[is_eos, step + 1] = dictionary.eos()
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
                    # ref_tokens = dictionary.string(target.data[i])
                    # if it is a dummy batch (e.g., a "padding" batch in a sharded
                    # dataset), id might exceeds the dataset size; in that case we
                    # just skip it
                    if id < len(self.valid_tgt_dataset):
                        ref_tokens = self.valid_tgt_dataset.get_original_tokens(id)
                        pred_tokens = dictionary.string(pred.data[i])
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
                # ref_one = dictionary.tokens_to_sentence(dictionary.string(target.data[i]))
                ref_one = self.train_tgt_dataset.get_original_text(
                    id, dictionary, bpe_symbol=self.args.remove_bpe,
                )
                pred_one = dictionary.tokens_to_sentence(
                    dictionary.string(pred.data[i][:length]),
                    bpe_symbol=self.args.remove_bpe,
                )
                print('| sample REF: ' + ref_one)
                print('| sample PRD: ' + pred_one)
        # word error stats code ends
        prob_mask = None
        if self.args.smoothing_type == 'temporal':
            # see https://arxiv.org/pdf/1612.02695.pdf
            # prob_mask.dtype=int for deterministic behavior of Tensor.scatter_add_()
            prob_mask = torch.zeros_like(lprobs, dtype=torch.int)  # bsz x tgtlen x vocab_size
            idx_tensor = target.new_full(target.size(), self.padding_idx).unsqueeze(-1)  # bsz x tgtlen x 1
            # hard-code the remaining probabilty mass distributed symmetrically
            # over neighbors at distance ±1 and ±2 with a 5 : 2 ratio
            idx_tensor[:, 2:, 0] = target[:, :-2]  # two neighbors to the left
            prob_mask.scatter_add_(-1, idx_tensor, prob_mask.new([2]).expand_as(idx_tensor))
            idx_tensor.fill_(self.padding_idx)[:, 1:, 0] = target[:, :-1]
            prob_mask.scatter_add_(-1, idx_tensor, prob_mask.new([5]).expand_as(idx_tensor))
            idx_tensor.fill_(self.padding_idx)[:, :-2, 0] = target[:, 2:]  # two neighbors to the right
            prob_mask.scatter_add_(-1, idx_tensor, prob_mask.new([2]).expand_as(idx_tensor))
            idx_tensor.fill_(self.padding_idx)[:, :-1, 0] = target[:, 1:]
            prob_mask.scatter_add_(-1, idx_tensor, prob_mask.new([5]).expand_as(idx_tensor))
            prob_mask[:, :, self.padding_idx] = 0  # clear cumulative count on <pad>
            prob_mask = prob_mask.float()  # convert to float
            sum_prob = prob_mask.sum(-1, keepdim=True)
            sum_prob[sum_prob.squeeze(-1).eq(0.)] = 1.  # to deal with the "division by 0" problem
            prob_mask = prob_mask.div_(sum_prob).view(-1, prob_mask.size(-1))

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            smoothing_type=self.args.smoothing_type, prob_mask=prob_mask,
            unigram_tensor=self.unigram_tensor,
        )
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
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
        agg_output = LabelSmoothedCrossEntropyCriterion.aggregate_logging_outputs(logging_outputs)
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
