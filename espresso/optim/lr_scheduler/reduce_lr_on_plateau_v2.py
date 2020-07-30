# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim.lr_scheduler

from fairseq.optim.lr_scheduler import register_lr_scheduler
from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateau


@register_lr_scheduler('reduce_lr_on_plateau_v2')
class ReduceLROnPlateauV2(ReduceLROnPlateau):
    """Decay the LR by a factor every time the validation loss plateaus, starting
    from the epoch specified as args.start_reduce_lr_epoch.

    We also support specifying a final lr which will be kept until the max number
    of epochs is reached.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=args.lr_patience, factor=args.lr_shrink,
            mode='max' if args.maximize_best_checkpoint_metric else 'min',
            threshold=args.lr_threshold, min_lr=args.final_lr_scale * args.lr[0]
        )

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        ReduceLROnPlateau.add_args(parser)
        # fmt: off
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
