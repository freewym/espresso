# Copyright (c) Facebook, Inc. and its affiliates, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import numpy as np

from fairseq.data.audio.feature_transforms import register_audio_feature_transform
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform


@register_audio_feature_transform("adaptive_specaugment")
class AdaptiveSpecAugmentTransform(SpecAugmentTransform):
    """Adaptive SpecAugment (https://arxiv.org/pdf/1912.05533.pdf)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return AdaptiveSpecAugmentTransform(
            _config.get("time_warp_W", 0),
            _config.get("freq_mask_N", 0),
            _config.get("freq_mask_F", 0),
            _config.get("time_mask_N", 0),
            _config.get("time_mask_T", 0),
            _config.get("time_mask_p", 0.0),
            _config.get("time_mask_pm", None),
            _config.get("time_mask_ps", None),
            _config.get("mask_value", None),
        )

    def __init__(
        self,
        time_warp_w: int = 0,
        freq_mask_n: int = 0,
        freq_mask_f: int = 0,
        time_mask_n: int = 0,
        time_mask_t: int = 0,
        time_mask_p: float = 0.0,
        time_mask_pm: Optional[float] = None,
        time_mask_ps: Optional[float] = None,
        mask_value: Optional[float] = 0.0,
    ):
        super().__init__(
            time_warp_w=time_warp_w,
            freq_mask_n=freq_mask_n,
            freq_mask_f=freq_mask_f,
            time_mask_n=time_mask_n,
            time_mask_t=time_mask_t,
            time_mask_p=time_mask_p,
            mask_value=mask_value,
        )
        self.time_mask_pm = time_mask_pm
        self.time_mask_ps = time_mask_ps

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"time_warp_w={self.time_warp_w}",
                    f"freq_mask_n={self.freq_mask_n}",
                    f"freq_mask_f={self.freq_mask_f}",
                    f"time_mask_n={self.time_mask_n}",
                    f"time_mask_t={self.time_mask_t}",
                    f"time_mask_p={self.time_mask_p}",
                    f"time_mask_pm={self.time_mask_pm}",
                    f"time_mask_ps={self.time_mask_ps}",
                ]
            )
            + ")"
        )

    def __call__(self, spectrogram):
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames = spectrogram.shape[0]  # or 'tau' in the paper.
        num_freqs = spectrogram.shape[1]  # or 'miu' in the paper.
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            return spectrogram

        if num_freqs < self.freq_mask_f:
            return spectrogram

        if self.time_warp_w > 0:
            if 2 * self.time_warp_w < num_frames:
                import cv2

                w0 = np.random.randint(self.time_warp_w, num_frames - self.time_warp_w)
                w = np.random.randint(-self.time_warp_w + 1, self.time_warp_w)
                upper, lower = distorted[:w0, :], distorted[w0:, :]
                upper = cv2.resize(
                    upper, dsize=(num_freqs, w0 + w), interpolation=cv2.INTER_LINEAR
                )
                lower = cv2.resize(
                    lower,
                    dsize=(num_freqs, num_frames - w0 - w),
                    interpolation=cv2.INTER_LINEAR,
                )
                distorted = np.concatenate((upper, lower), axis=0)

        for _i in range(self.freq_mask_n):
            f = np.random.randint(0, self.freq_mask_f)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0 : f0 + f] = mask_value

        max_time_mask_t = (
            min(self.time_mask_t, math.floor(num_frames * self.time_mask_p))
            if self.time_mask_ps is None
            else math.floor(num_frames * self.time_mask_ps)
        )
        if max_time_mask_t < 1:
            return distorted

        time_mask_n = (
            self.time_mask_n
            if self.time_mask_pm is None
            else min(20, math.floor(num_frames * self.time_mask_pm))
        )
        for _i in range(time_mask_n):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        return distorted
