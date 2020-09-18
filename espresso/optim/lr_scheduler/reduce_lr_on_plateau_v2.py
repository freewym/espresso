# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from omegaconf import II
from typing import List

import torch.optim.lr_scheduler

from fairseq.dataclass.utils import FairseqDataclass, gen_parser_from_dataclass
from fairseq.optim.lr_scheduler import register_lr_scheduler
from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateau


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
            "help": "initial learning rate during warmup phase; default is args.lr"
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
    lr: List[float] = II("params.optimization.lr")


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
        gen_parser_from_dataclass(parser, ReduceLROnPlateauV2Config())

    def step(self, epoch, val_loss=None):
        if epoch < self.args.start_reduce_lr_epoch:
            self.lr_scheduler.last_epoch = epoch
            self.optimizer.set_lr(self.args.lr[0])
            return self.optimizer.get_lr()
        return super().step(epoch, val_loss)
