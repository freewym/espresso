# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import data_utils

from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

from espresso.tools import wer
from espresso.tools.simple_greedy_decoder import SimpleGreedyDecoder


@register_criterion('cross_entropy_with_wer')
class CrossEntropyWithWERCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        dictionary = task.target_dictionary
        self.scorer = wer.Scorer(dictionary, wer_output_filter=task.args.wer_output_filter)
        self.decoder_for_validation = SimpleGreedyDecoder(dictionary, for_validation=True)
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
        else:
            tokens, lprobs, _ = self.decoder_for_validation.decode([model], sample)
            pred = tokens[:, 1:].data.cpu()  # bsz x len
            target = sample['target']
            # compute word error stats
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

    def set_train_tgt_dataset(self, dataset):
        self.train_tgt_dataset = dataset

    def set_valid_tgt_dataset(self, dataset):
        self.valid_tgt_dataset = dataset

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def set_epoch(self, epoch):
        self.epoch = epoch
