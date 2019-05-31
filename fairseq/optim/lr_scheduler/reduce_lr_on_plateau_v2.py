# Copyright (c) 2019-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim.lr_scheduler

from . import FairseqLRScheduler, register_lr_scheduler
from .reduce_lr_on_plateau import ReduceLROnPlateau


@register_lr_scheduler('reduce_lr_on_plateau_v2')
class ReduceLROnPlateauV2(ReduceLROnPlateau):
    """Decay the LR by a factor every time the validation loss plateaus, starting
    from the epoch specified as args.start_reduce_lr_epoch.

    We also support a warmup phase where we linearly increase the learning rate
    from 0 until the configured learning rate (``--lr``).
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        if args.warmup_updates > 0:
            self.warmup_factor = 1. / args.warmup_updates
        else:
            self.warmup_factor = 1.

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            threshold=args.lr_threshold, min_lr=args.min_lr)
    
    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        ReduceLROnPlateau.add_args(parser)
        # fmt: off
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--start-reduce-lr-epoch', default=0, type=int, metavar='N',
                            help='start to reduce lr from the specified epoch')
        # fmt: on
    
    def step(self, epoch, val_loss=None):
        if epoch < self.args.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            self.optimizer.set_lr(self.warmup_factor * self.args.lr[0])
            return self.optimizer.get_lr()
        return super().step(epoch, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_updates > 0 and num_updates <= self.args.warmup_updates:
            self.warmup_factor = num_updates / float(self.args.warmup_updates)
            self.optimizer.set_lr(self.warmup_factor * self.args.lr[0])
        return self.optimizer.get_lr()
