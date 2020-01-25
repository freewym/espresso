# Copyright (c) Tongfei Chen, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List


class ScheduledSamplingRateScheduler(object):

    def __init__(
        self,
        scheduled_sampling_probs: List[float] = [1.0],
        start_scheduled_sampling_epoch: int = 1,
    ):
        """
        Args:
            scheduled_sampling_probs (List[float]): P_1,P_2,...,P_N.
                Scheduled sampling probabilities of sampling the truth labels
                for N epochs starting from --start-schedule-sampling-epoch;
                all later epochs using P_N.
            start_scheduled_sampling_epoch (int): start scheduled sampling from
                the specified epoch.
        """

        self.scheduled_sampling_probs = scheduled_sampling_probs
        self.start_scheduled_sampling_epoch = start_scheduled_sampling_epoch

    def step(self, epoch: int) -> float:
        if (
            (
                len(self.scheduled_sampling_probs) > 1
                or self.scheduled_sampling_probs[0] < 1.0
            )
            and epoch >= self.start_scheduled_sampling_epoch
        ):
            prob = self.scheduled_sampling_probs[
                min(epoch - self.start_scheduled_sampling_epoch,
                    len(self.scheduled_sampling_probs) - 1)
            ]
            return prob
        else:
            return 1.0
