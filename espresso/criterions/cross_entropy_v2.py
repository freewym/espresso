# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np

import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.data import data_utils


logger = logging.getLogger(__name__)


@register_criterion("cross_entropy_v2")
class CrossEntropyV2Criterion(CrossEntropyCriterion):

    def __init__(self, task, sentence_avg, print_interval):
        super().__init__(task, sentence_avg)

        self.dictionary = task.target_dictionary
        self.print_interval = print_interval
        self.epoch = 1

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--print-training-sample-interval", type=int,
                            metavar="N", dest="print_interval", default=500,
                            help="print a training sample (reference + "
                                 "prediction) every this number of updates")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample; periodically print out
        randomly sampled predictions from the training set.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], epoch=self.epoch)
        loss, _, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if (
            hasattr(model, "num_updates") and model.training and
            model.num_updates // self.print_interval >
            (model.num_updates - 1) // self.print_interval
        ):  # print a randomly sampled result every print_interval updates
            target = model.get_targets(sample, net_output)
            pred = lprobs.argmax(-1).cpu()  # bsz x len
            assert pred.size() == target.size()
            with data_utils.numpy_seed(model.num_updates):
                i = np.random.randint(0, len(sample["id"]))
            ref_tokens = sample["target_raw_text"][i]
            length = utils.strip_pad(target.data[i], self.padding_idx).size(0)
            ref_one = self.dictionary.wordpiece_decode(ref_tokens)
            pred_one = self.dictionary.wordpiece_decode(self.dictionary.string(pred.data[i][:length]))
            logger.info("sample REF: " + ref_one)
            logger.info("sample PRD: " + pred_one)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss, lprobs

    def set_epoch(self, epoch):
        self.epoch = epoch
