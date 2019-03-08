# Copyright (c) 2017-present, Facebook, Inc.
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
    """Decay the LR by a factor every time the validation loss plateausi, after start_epoch_to_reduce."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            min_lr=args.min_lr)
    
    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--start-reduce-lr-epoch', default=0, type=int, metavar='N',
                            help='start to reduce lr from specified epoch')
        # fmt: on
    
    def step(self, epoch, val_loss=None):
        if epoch < self.args.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            return self.args.lr[0]
        return super().step(epoch, val_loss)
