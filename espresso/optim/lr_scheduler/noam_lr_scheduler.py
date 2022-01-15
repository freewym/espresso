# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class NoamLRScheduleConfig(FairseqDataclass):
    warmup_steps: int = field(
        default=0,
        metadata={"help": "warmup steps used in calculating lr of Noam"},
    )
    model_size: int = field(
        default=512,
        metadata={"help": "embed dimension of the transformer model"},
    )
    final_lr: float = field(
        default=0.0,
        metadata={"help": "final learning rate; default to 0.0"},
    )
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("noam", dataclass=NoamLRScheduleConfig)
class NoamLRScheduler(FairseqLRScheduler):
    """Noam lr scheduler.

    Proposed in "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf.
    """

    def __init__(self, cfg: NoamLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with noam lr."
                " Consider --lr-scheduler=fixed instead."
            )
        self.factor = cfg.lr[0]
        self.warmup_steps = cfg.warmup_steps
        self.model_size = cfg.model_size
        self.final_lr = cfg.final_lr

        # initial learning rate
        self.lr = (
            self.factor
            * self.model_size ** (-0.5)
            * min(1.0, self.warmup_steps ** (-1.5))
        )
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        num_updates = num_updates + 1  # num_updates starts from 1
        self.lr = (
            self.factor
            * self.model_size ** (-0.5)
            * min(num_updates ** (-0.5), num_updates * self.warmup_steps ** (-1.5))
        )
        if num_updates > self.warmup_steps:
            self.lr = max(self.lr, self.final_lr)

        self.optimizer.set_lr(self.lr)

        return self.lr
