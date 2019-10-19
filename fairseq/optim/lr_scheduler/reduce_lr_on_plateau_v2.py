# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

        self.init_lr = args.init_lr_scale * args.lr[0] if args.warmup_updates > 0 else args.lr[0]
        self.warmup_rate = (args.lr[0] - self.init_lr) / args.warmup_updates \
            if args.warmup_updates > 0 else 0.

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            threshold=args.lr_threshold, min_lr=args.final_lr_scale * args.lr[0])
    
    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        ReduceLROnPlateau.add_args(parser)
        # fmt: off
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--init-lr-scale', default=0.01, type=float, metavar='N',
                            help='initial learning rate scale during warmup phase; default is 0.01')
        parser.add_argument('--final-lr-scale', default=0.01, type=float, metavar='N',
                            help='final learning rate scale; default to 0.01')
        parser.add_argument('--start-reduce-lr-epoch', default=0, type=int, metavar='N',
                            help='start to reduce lr from the specified epoch')
        # fmt: on
    
    def step(self, epoch, val_loss=None):
        if epoch < self.args.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            self.optimizer.set_lr(self.args.lr[0])
            return self.optimizer.get_lr()
        return super().step(epoch, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_updates > 0 and num_updates <= self.args.warmup_updates:
            self.optimizer.set_lr(self.init_lr + self.warmup_rate * num_updates)
        return self.optimizer.get_lr()
