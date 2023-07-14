# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.optim.lr_scheduler import register_lr_scheduler
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import (
    PolynomialDecayLRSchedule,
    PolynomialDecayLRScheduleConfig,
)


@register_lr_scheduler("polynomial_decay_v2", dataclass=PolynomialDecayLRScheduleConfig)
class PolynomialDecayV2LRSchedule(PolynomialDecayLRSchedule):
    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass
