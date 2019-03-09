# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils, wer
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder

from . import FairseqCriterion, register_criterion
from .cross_entropy import CrossEntropyCriterion


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dict = task.dict if hasattr(task, 'dict') else getattr(task, 'tgt_dict')
        self.scorer = wer.Scorer(dict,
            wer_output_filter=task.args.wer_output_filter)
        self.train_tgt_dataset = task.dataset(args.train_subset).tgt
        self.valid_tgt_dataset = None
        self.num_updates = -1

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--print-training-sample-interval', type=int,
                            metavar='N', dest='print_interval', default=500,
                            help='print a training sample (reference + '
                                 'prediction) every this number of updates')
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
            # and at least the length of target
            maxlen = max(encoder_out['encoder_out'][0].size(1), target.size(1))
            tokens = target.new_full([target.size(0), maxlen + 2], dict.pad())
            tokens[:, 0] = dict.eos()
            lprobs = []
            attn = [] if model.decoder.need_attn else None
            dummy_log_probs = encoder_out['encoder_out'][0].new_full(
                [target.size(0), len(dict)], -np.log(len(dict)))
            for step in range(maxlen + 1): # one extra step for EOS marker
                is_eos = tokens[:, step].eq(dict.eos())
                # if all predictions are finished (i.e., ended with eos),
                # pad lprobs to target length with dummy log probs,
                # truncate tokens up to this step and break
                if step > 0 and is_eos.sum() == is_eos.size(0):
                    for _ in range(step, target.size(1)):
                        lprobs.append(dummy_log_probs)
                    tokens = tokens[:, :step + 1]
                    break
                log_probs, attn_scores = self._decode(tokens[:, :step + 1],
                    model, encoder_out, incremental_states)
                log_probs[:, dict.pad()] = -math.inf  # never select pad
                tokens[:, step + 1] = log_probs.argmax(-1)
                if step > 0: # deal with finished predictions
                    # make log_probs uniform if the previous output token is EOS
                    # and add consecutive EOS to the end of prediction
                    log_probs[is_eos, :] = -np.log(log_probs.size(1))
                    tokens[is_eos, step + 1] = dict.eos()
                if step < target.size(1):
                    lprobs.append(log_probs)
                if model.decoder.need_attn:
                    attn.append(attn_scores)
            # bsz x min(tgtlen, maxlen + 1) x vocab_size
            lprobs = torch.stack(lprobs, dim=1)
            if model.decoder.need_attn:
                # bsz x (maxlen + 1) x (length of encoder_out)
                attn = torch.stack(attn, dim=1)
        # word error stats code starts
        if not model.training or (self.num_updates // self.args.print_interval >
            (self.num_updates - 1) // self.args.print_interval):
            pred = lprobs.argmax(-1).cpu() if model.training else \
                tokens[:, 1:].data.cpu() # bsz x len

            if not model.training: # validation step, compute WER stats with scorer
                assert pred.size(0) == target.size(0)
                self.scorer.reset()
                for i in range(target.size(0)):
                    utt_id = sample['utt_id'][i]
                    id = sample['id'].data[i]
                    #ref_tokens = dict.string(target.data[i])
                    ref_tokens = self.valid_tgt_dataset.get_original_tokens(id)
                    pred_tokens = dict.string(pred.data[i])
                    self.scorer.add_evaluation(utt_id, ref_tokens, pred_tokens)
            else: # print a randomly sampled result every print_interval updates
                assert pred.size() == target.size()
                with data_utils.numpy_seed(self.num_updates):
                    i = np.random.randint(0, len(sample['id']))
                id = sample['id'].data[i]
                length = utils.strip_pad(target.data[i], self.padding_idx).size(0)
                #ref_one = dict.tokens_to_sentence(dict.string(target.data[i]))
                ref_one = self.train_tgt_dataset.get_original_text(id, dict)
                pred_one = dict.tokens_to_sentence(
                    dict.string(pred.data[i][:length]))
                print('| sample REF: ' + ref_one)
                print('| sample PRD: ' + pred_one)
        # word error stats code ends
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(lprobs, target.view(-1), ignore_index=self.padding_idx,
                          reduction='sum' if reduce else 'none')
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if not model.training: # do not compute word error in training mode
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
        if word_count > 0: # model.training == False
            agg_output['word_error'] = word_error
            agg_output['word_count'] = word_count
        if char_count > 0: # model.training == False
            agg_output['char_error'] = char_error
            agg_output['char_count'] = char_count
        return agg_output

    def _decode(self, tokens, model, encoder_out, incremental_states):
        decoder_out = list(model.decoder(tokens, encoder_out,
            incremental_state=incremental_states))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        if attn is not None:
            if type(attn) is dict:
                attn = attn['attn']
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=True)
        probs = probs[:, -1, :]
        return probs, attn

    def set_valid_tgt_dataset(self, dataset):
        self.valid_tgt_dataset = dataset

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
