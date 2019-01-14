# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import torch.nn.functional as F

from fairseq import utils, wer
from fairseq.data import data_utils

from . import FairseqCriterion, register_criterion
from .cross_entropy import CrossEntropyCriterion 


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dict = task.dict if hasattr(task, 'dict') else getattr(task, 'tgt_dict')
        self.scorer = wer.Scorer(dict)
        self.num_calls = 0 #getattr(task, 'iterations_in_epoch', 0)
        self.train_tgt_dataset = task.dataset(args.train_subset).tgt
        self.valid_tgt_dataset = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--sample-results-interval', type=int, metavar='N', default=500,
                            help='print sample results interval every this '
                                 'number of forward times')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # wer stats code starts
        if not model.training or \
            self.num_calls % self.args.sample_results_interval == 0:
            pred = lprobs.argmax(-1).long().cpu() # bsz x len
            target = sample['target'].long().cpu() # bsz x len
            assert pred.size() == sample['net_input']['prev_output_tokens'].size()
            assert pred.size() == target.size()

            def lengths_strip_padding(idx_array, padding_idx):
                # assume sequences are right-padded, so work out the length by
                # looking for the first occurence of padding_idx
                assert idx_array.ndim == 1 or idx_array.ndim == 2
                if idx_array.ndim == 1:
                    try:
                        return idx_array.tolist().index(padding_idx)
                    except ValueError:
                        return len(idx_array)
                return [lengths_strip_padding(row, padding_idx) for row in idx_array]

            target_lengths = lengths_strip_padding(target.numpy(),
                self.padding_idx)
            dict = self.scorer.dict
            if not model.training: # validation step, compute WER stats with scorer
                self.scorer.reset()
                for i, length in enumerate(target_lengths):
                    utt_id = sample['utt_id'][i]
                    id = sample['id'].data[i]
                    #ref_str = dict.string(target.data[i])
                    ref_str = self.valid_tgt_dataset.get_original_tokens(id)
                    pred_str = dict.string(pred.data[i][:length])
                    self.scorer.add_evaluation(utt_id, ref_str, pred_str)
            else: # print a randomly sampled result every sample_results_interval batch
                with data_utils.numpy_seed(self.num_calls):
                    i = np.random.randint(0, len(sample['id']))
                id = sample['id'].data[i]
                #ref_str_one = dict.string(target.data[i])
                ref_str_one = self.train_tgt_dataset.get_original_tokens(id)
                pred_str_one = dict.string(pred.data[i][:target_lengths[i]])
                print('| ' + sample['utt_id'][i])
                print('| sample REF: ' + ref_str_one)
                print('| sample PRD: ' + pred_str_one)
        self.num_calls += 1
        # wer stats code ends
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
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
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_output = CrossEntropyCriterion.aggregate_logging_outputs(logging_outputs)
        word_error = sum(log.get('word_error', 0) for log in logging_outputs)
        word_count = sum(log.get('word_count', 0) for log in logging_outputs)
        if word_count > 0: # model.training == False
            agg_output['word_error'] = word_error
            agg_output['word_count'] = word_count
        return agg_output

    def set_valid_tgt_dataset(self, dataset):
        self.valid_tgt_dataset = dataset
