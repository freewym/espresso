# Copyright (c) Facebook, Inc. and its affiliates, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import data_utils
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


@dataclass
class CtcLossCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    print_training_sample_interval: int = field(
        default=500,
        metadata={
            "help": "print a training sample (reference + prediction) every this number of updates"
        },
    )


@register_criterion("ctc_loss", dataclass=CtcLossCriterionConfig)
class CtcLossCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcLossCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
        self.print_interval = cfg.print_training_sample_interval

        self.dictionary = task.target_dictionary
        self.prev_num_updates = -1

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in net_output:
            output_lengths = net_output["src_lengths"][0]
        else:
            if net_output["encoder_padding_mask"] is not None:
                non_padding_mask = ~net_output["encoder_padding_mask"][0]
                output_lengths = non_padding_mask.long().sum(-1)
            else:
                output_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                output_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction=("sum" if reduce else "none"),
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        if (
            hasattr(model, "num_updates")
            and model.training
            and model.num_updates // self.print_interval
            > (model.num_updates - 1) // self.print_interval
            and model.num_updates != self.prev_num_updates
        ):  # print a randomly sampled result every print_interval updates
            self.prev_num_updates = model.num_updates
            with data_utils.numpy_seed(model.num_updates):
                i = np.random.randint(0, len(sample["id"]))
                ref_one = sample["text"][i]
                lprobs_one = lprobs[: output_lengths[i], i, :]
                pred_one = self.dictionary.wordpiece_decode(
                    self.dictionary.string(
                        lprobs_one.argmax(dim=-1).unique_consecutive(),
                        extra_symbols_to_ignore=getattr(
                            self.task, "extra_symbols_to_ignore", None
                        ),
                    )
                )
                logger.info("sample REF: " + ref_one)
                logger.info("sample PRD: " + pred_one)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss",
            loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            sample_size,
            round=3,
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss",
                loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0,
                ntokens,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
