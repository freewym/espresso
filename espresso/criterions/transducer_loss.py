# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import data_utils
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


@dataclass
class TransducerLossCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    print_training_sample_interval: int = field(
        default=500,
        metadata={
            "help": "print a training sample (reference + prediction) every this number of updates"
        },
    )


@register_criterion("transducer_loss", dataclass=TransducerLossCriterionConfig)
class TransducerLossCriterion(FairseqCriterion):
    def __init__(self, cfg: TransducerLossCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

        self.sentence_avg = cfg.sentence_avg
        self.print_interval = cfg.print_training_sample_interval
        self.dictionary = task.target_dictionary
        self.prev_num_updates = -1

    def forward(self, model, sample, reduce=True):
        try:
            from torchaudio.functional import rnnt_loss
        except ImportError:
            raise ImportError("Please install a newer torchaudio (version >= 0.10.0)")

        net_output, encoder_out_lengths = model(
            **sample["net_input"]
        )  # B x T x U x V, B

        if "target_lengths" in sample:
            target_lengths = sample[
                "target_lengths"
            ].int()  # Note: ensure EOS is excluded
        else:
            target_lengths = (
                (
                    (sample["target"] != self.pad_idx)
                    & (sample["target"] != self.eos_idx)
                )
                .sum(-1)
                .int()
            )

        loss = rnnt_loss(
            net_output,
            sample["target"][:, :-1].int().contiguous(),  # exclude the last EOS column
            encoder_out_lengths.int(),
            target_lengths,
            blank=self.blank_idx,
            clamp=-1.0,
            reduction=("sum" if reduce else "none"),
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
            lat_one = net_output[
                i, : encoder_out_lengths[i], : target_lengths[i], :
            ]  # T x U x V
            t, u = 0, 0
            pred_one = []
            # truncate the prediction to the range of the lattice
            while t < encoder_out_lengths[i] and u < target_lengths[i]:
                pred_one.append(lat_one[t, u, :].argmax().item())
                if pred_one[-1] == self.blank_idx:
                    t += 1
                else:
                    u += 1
            pred_one = self.dictionary.wordpiece_decode(
                self.dictionary.string(torch.as_tensor(pred_one))
            )
            logger.info("sample REF: " + ref_one)
            logger.info("sample PRD: " + pred_one)

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
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
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
