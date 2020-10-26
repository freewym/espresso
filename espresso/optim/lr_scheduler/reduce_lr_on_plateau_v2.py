# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

import torch.optim.lr_scheduler

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.optim.lr_scheduler import register_lr_scheduler
from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateau
from omegaconf import II, DictConfig


@dataclass
class ReduceLROnPlateauV2Config(FairseqDataclass):
    lr_shrink: float = field(
        default=0.1,
        metadata={"help": "shrink factor for annealing, lr_new = (lr * lr_shrink)"},
    )
    lr_threshold: float = field(
        default=1e-4,
        metadata={
            "help": "threshold for measuring the new optimum, to only focus on significant changes"
        },
    )
    lr_patience: int = field(
        default=0,
        metadata={
            "help": "number of epochs with no improvement after which learning rate will be reduced"
        },
    )
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is cfg.lr"
        },
    )
    final_lr_scale: float = field(
        default=0.01,
        metadata={"help": "final learning rate scale; default to 0.01"},
    )
    start_reduce_lr_epoch: int = field(
        default=0,
        metadata={"help": "start to reduce lr from the specified epoch"},
    )
    # TODO common vars at parent class
    lr: List[float] = II("optimization.lr")
    maximize_best_checkpoint_metric: bool = II("checkpoint.maximize_best_checkpoint_metric")


@register_lr_scheduler("reduce_lr_on_plateau_v2", dataclass=ReduceLROnPlateauV2Config)
class ReduceLROnPlateauV2(ReduceLROnPlateau):
    """Decay the LR by a factor every time the validation loss plateaus, starting
    from the epoch specified as cfg.start_reduce_lr_epoch.

    We also support specifying a final lr which will be kept until the max number
    of epochs is reached.
    """

    def __init__(self, cfg: DictConfig, fairseq_optimizer):
        super().__init__(cfg, fairseq_optimizer)

        self.cfg = cfg
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=cfg.lr_patience,
            factor=cfg.lr_shrink,
            mode="max" if cfg.maximize_best_checkpoint_metric else "min",
            threshold=cfg.lr_threshold,
            min_lr=cfg.final_lr_scale * cfg.lr[0],
        )

    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    def step(self, epoch, val_loss=None):
        if epoch < self.cfg.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            self.optimizer.set_lr(self.cfg.lr[0])
            return self.optimizer.get_lr()
        return super().step(epoch, val_loss)
