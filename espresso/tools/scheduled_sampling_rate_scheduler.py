# Copyright (c) Tongfei Chen, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from fairseq.options import eval_str_list


class ScheduledSamplingRateScheduler:

    def __init__(self, args):
        self.args = args

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--scheduled-sampling-probs', type=lambda p: eval_str_list(p),
                            metavar='P_1,P_2,...,P_N', default=1.0,
                            help='scheduled sampling probabilities of sampling the truth '
                            'labels for N epochs starting from --start-schedule-sampling-epoch; '
                            'all later epochs using P_N')
        parser.add_argument('--start-scheduled-sampling-epoch', type=int,
                            metavar='N', default=1,
                            help='start scheduled sampling from the specified epoch')

    def step(self, epoch: int) -> float:
        if (
                (len(self.args.scheduled_sampling_probs) > 1 or
                 self.args.scheduled_sampling_probs[0] < 1.0) and
                epoch >= self.args.start_scheduled_sampling_epoch
        ):
            ss_prob = self.args.scheduled_sampling_probs[
                min(epoch - self.args.start_scheduled_sampling_epoch,
                    len(self.args.scheduled_sampling_probs) - 1)
            ]
            return ss_prob
        else:
            return 1.0
