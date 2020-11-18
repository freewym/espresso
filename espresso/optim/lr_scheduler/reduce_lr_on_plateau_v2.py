# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

import torch.optim.lr_scheduler

from fairseq.optim.lr_scheduler import register_lr_scheduler
from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import (
    ReduceLROnPlateauLRSchedule,
    ReduceLROnPlateauLRScheduleConfig,
)


@dataclass
class ReduceLROnPlateauLRScheduleV2Config(ReduceLROnPlateauLRScheduleConfig):
    start_reduce_lr_epoch: int = field(
        default=0,
        metadata={"help": "start to reduce lr from the specified epoch"},
    )
    final_lr_scale: float = field(
        default=0.01,
        metadata={"help": "final learning rate scale; default to 0.01"},
    )


@register_lr_scheduler("reduce_lr_on_plateau_v2", dataclass=ReduceLROnPlateauLRScheduleV2Config)
class ReduceLROnPlateauLRScheduleV2(ReduceLROnPlateauLRSchedule):
    """Decay the LR by a factor every time the validation loss plateaus, starting
    from the epoch specified as cfg.start_reduce_lr_epoch.

    We also support specifying a final lr which will be kept until the max number
    of epochs is reached.
    """

    def __init__(self, cfg: ReduceLROnPlateauLRScheduleV2Config, optimizer):
        super().__init__(cfg, optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=cfg.lr_patience,
            factor=cfg.lr_shrink,
            mode="max" if cfg.maximize_best_checkpoint_metric else "min",
            threshold=cfg.lr_threshold,
            min_lr=cfg.final_lr_scale * cfg.lr[0],
        )

    def step(self, epoch, val_loss=None):
        if epoch < self.cfg.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            self.optimizer.set_lr(self.cfg.lr[0])
            return self.optimizer.get_lr()
        return super().step(epoch, val_loss)
