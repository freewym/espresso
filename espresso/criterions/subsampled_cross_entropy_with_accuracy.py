# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F

from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig
from fairseq.logging import metrics


logger = logging.getLogger(__name__)


@dataclass
class SubsampledCrossEntropyWithAccuracyCriterionConfig(CrossEntropyCriterionConfig):
    pass


@register_criterion("subsampled_cross_entropy_with_accuracy", dataclass=SubsampledCrossEntropyWithAccuracyCriterionConfig)
class SubsampledCrossEntropyWithAccuracyCriterion(CrossEntropyCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)
        self.subsampling_factor = None
        # indicate whether to transpose the first two dimensions of net_output
        # so that it is B x T x V
        self.transpose_net_output = getattr(task, "transpose_net_output", True)
        self.state_prior_update_interval = getattr(task, "state_prior_update_interval", None)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, num_corr, num_tot, state_post = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "num_corr": num_corr,
            "num_tot": num_tot,
            "state_post": state_post,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        if self.subsampling_factor is None:
            self.subsampling_factor = int(round(100 / model.output_lengths(100)))
            logger.info("subsampling factor for target labels = {}".format(self.subsampling_factor))
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if self.transpose_net_output:
            lprobs = lprobs.transpose(0, 1).contiguous()  # T x B x V -> B x T x V
        net_output_length = lprobs.size(1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        if self.subsampling_factor > 1:
            target = target[:, ::self.subsampling_factor]
            target = target[:, :net_output_length]  # truncate if necessary
            right_pad_length = net_output_length - target.size(1)
            if right_pad_length > 0:  # pad with the right-most labels on the right
                target = torch.cat([target, target[:, -1:].expand(-1, right_pad_length)], 1)
        target = target.view(-1)
        if not model.training:
            # hack for dummy batches, assuming lprobs is longer than targets
            lprobs = lprobs[:target.size(0)]
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        with torch.no_grad():
            mask = target.ne(self.padding_idx)
            num_corr = (lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)).int().sum()
            num_tot = mask.int().sum()

            state_post = None
            if (
                hasattr(model, "num_updates") and model.training and
                self.state_prior_update_interval is not None and
                model.num_updates // self.state_prior_update_interval >
                (model.num_updates - 1) // self.state_prior_update_interval
            ):
                frame_indices = torch.nonzero(mask, as_tuple=True)[0]
                state_post = lprobs.index_select(0, frame_indices).exp().mean(0).detach()

        return loss, num_corr, num_tot, state_post

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        CrossEntropyCriterion.reduce_metrics(logging_outputs)
        num_corr = sum(log.get("num_corr", 0) for log in logging_outputs)
        num_tot = sum(log.get("num_tot", 0) for log in logging_outputs)
        metrics.log_scalar(
            "accuracy", num_corr.float() / num_tot * 100 if num_tot > 0 else 0.0, num_tot, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # because new_state_prior is not a scalar
        return False
