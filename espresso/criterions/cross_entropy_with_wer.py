# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np

import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.data import data_utils

from espresso.tools import wer
from espresso.tools.simple_greedy_decoder import SimpleGreedyDecoder


logger = logging.getLogger(__name__)


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dictionary = task.target_dictionary
        self.scorer = wer.Scorer(dictionary, wer_output_filter=task.args.wer_output_filter)
        self.decoder_for_validation = SimpleGreedyDecoder(dictionary, for_validation=True)
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
            if (
                self.num_updates // self.args.print_interval >
                (self.num_updates - 1) // self.args.print_interval
            ):  # print a randomly sampled result every print_interval updates
                pred = lprobs.argmax(-1).cpu()  # bsz x len
                assert pred.size() == target.size()
                with data_utils.numpy_seed(self.num_updates):
                    i = np.random.randint(0, len(sample['id']))
                ref_tokens = sample['target_raw_text'][i]
                length = utils.strip_pad(target.data[i], self.padding_idx).size(0)
                ref_one = dictionary.tokens_to_sentence(
                    ref_tokens, use_unk_sym=False, bpe_symbol=self.args.remove_bpe,
                )
                pred_one = dictionary.tokens_to_sentence(
                    dictionary.string(pred.data[i][:length]), use_unk_sym=True,
                    bpe_symbol=self.args.remove_bpe,
                )
                logger.info('sample REF: ' + ref_one)
                logger.info('sample PRD: ' + pred_one)
        else:
            tokens, lprobs, _ = self.decoder_for_validation.decode([model], sample)
            pred = tokens[:, 1:].data.cpu()  # bsz x len
            target = sample['target']
            # compute word error stats
            assert pred.size(0) == target.size(0)
            self.scorer.reset()
            for i in range(target.size(0)):
                utt_id = sample['utt_id'][i]
                ref_tokens = sample['target_raw_text'][i]
                pred_tokens = dictionary.string(pred.data[i])
                self.scorer.add_evaluation(
                    utt_id, ref_tokens, pred_tokens, bpe_symbol=self.args.remove_bpe,
                )

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
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        CrossEntropyCriterion.reduce_metrics(logging_outputs)

        word_error = sum(log.get('word_error', 0) for log in logging_outputs)
        word_count = sum(log.get('word_count', 0) for log in logging_outputs)
        char_error = sum(log.get('char_error', 0) for log in logging_outputs)
        char_count = sum(log.get('char_count', 0) for log in logging_outputs)
        if word_count > 0:  # model.training == False
            metrics.log_scalar('wer', float(word_error) / word_count * 100, word_count, round=4)
        if char_count > 0:  # model.training == False
            metrics.log_scalar('wer', float(word_error) / word_count * 100, word_count, round=4)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def set_epoch(self, epoch):
        self.epoch = epoch
