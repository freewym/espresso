# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils, wer

from . import FairseqCriterion, register_criterion
from .cross_entropy import CrossEntropyCriterion 


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        dict = self.task.dict if hasattr(self.task, 'dict') \
            else self.task.tgt_dict
        self.scorer = wer.Scorer(dict)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # wer code starts
        if not model.training:
            pred = lprobs.argmax(-1).int().cpu() # bsz x len
            assert pred.size() == sample['net_input']['prev_output_tokens'].size()
            assert pred.size() == sample['target'].size()
            dict = self.task.dict if hasattr(self.task, 'dict') \
                else self.task.tgt_dict
            self.scorer.reset()
            ref_str_list = dict.string(sample['target'].int().cpu()).split('\n')
            pred_str_list = dict.string(pred).split('\n')
            for ref_str, pred_str in zip(ref_str_list, pred_str_list):
                scorer.add(ref_str, pred_str)
        # wer code ends
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
        if not model.training:
            logging_output['word_error'] = scorer.acc_word_error()
            logging_output['word_count'] = scorer.acc_word_count()
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_output = super().aggregate_logging_outputs(logging_outputs)
        word_error = sum(log.get('word_error', 0) for log in logging_outputs)
        word_count = sum(log.get('word_count', 0) for log in logging_outputs)
        if word_count > 0:
            agg_output['word_error'] = word_error
            agg_output['word_count'] = word_count
        else:
            print('Not aggregating WER in training mode.')
        return agg_output
