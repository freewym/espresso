# Copyright (c) Nanxin Chen, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is modified from https://github.com/zcaceres/spec_augment

MIT License

Copyright (c) 2019 Zach Caceres

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

import torch


def specaug(spec, W=80, F=27, T=70, num_freq_masks=2, num_time_masks=2, p=0.2, replace_with_zero=False):
    """SpecAugment

    Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        (https://arxiv.org/pdf/1904.08779.pdf)

    This implementation modified from https://github.com/zcaceres/spec_augment

    Args:
        spec (numpy.ndarray): input tensor of the shape `(T, dim)`
        W (int): time warp parameter
        F (int): maximum width of each freq mask
        T (int): maximum width of each time mask
        num_freq_masks (int): number of frequency masks
        num_time_masks (int): number of time masks
        p (int): toal mask width shouldn't exeed this times num of frames
        replace_with_zero (bool): if True, masked parts will be filled with 0, if False, filled with mean

    Returns:
        output (numpy.ndarray): resultant matrix of shape `(T, dim)`
    """
    spec = torch.from_numpy(spec)
    if replace_with_zero:
        pad_value = 0.
    else:
        pad_value = spec.mean()
    time_warped = time_warp(spec.transpose(0, 1), W=W)
    freq_masked = freq_mask(time_warped, F=F, num_masks=num_freq_masks, pad_value=pad_value)
    time_masked = time_mask(freq_masked, T=T, num_masks=num_time_masks, p=p, pad_value=pad_value)
    return time_masked.transpose(0, 1).numpy()


def time_warp(spec, W=5):
    """Time warping

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        W (int): time warp parameter

    Returns:
        time warpped tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    t = spec.shape[1]
    if t - W <= W:
        return spec
    center = np.random.randint(W, t - W)
    warped = np.random.randint(center - W + 1, center + W + 1)
    spec = spec.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():  # to make the results deterministic 
        left = torch.nn.functional.interpolate(spec[:, :, :, :center],
            size=(spec.shape[2], warped), mode='bicubic', align_corners=False)
        right = torch.nn.functional.interpolate(spec[:, :, :, center:],
            size=(spec.shape[2], t - warped), mode='bicubic', align_corners=False)
    return torch.cat((left, right), dim=-1).squeeze(0).squeeze(0)


def freq_mask(spec, F=30, num_masks=1, pad_value=0.):
    """Frequency masking

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        F (int): maximum width of each mask
        num_masks (int): number of masks
        pad_value (float): value for padding

     Returns:
         freq masked tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    cloned = spec.unsqueeze(0).clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = np.random.randint(0, F)
        f_zero = np.random.randint(0, num_mel_channels - f)

        # avoids randint error if values are equal and range is empty
        if f == 0:
            return cloned.squeeze(0)

        cloned[0][f_zero:f_zero + f] = pad_value

    return cloned.squeeze(0)


def time_mask(spec, T=40, num_masks=1, p=0.2, pad_value=0.):
    """Time masking

    Args:
        spec (torch.Tensor): input tensor of shape `(dim, T)`
        T (int): maximum width of each mask
        num_masks (int): number of masks
        p (float): toal mask width shouldn't exeed this times num of frames
        pad_value (float): value for padding

    Returns:
        time masked tensor (torch.Tensor): output tensor of shape `(dim, T)`
    """
    cloned = spec.unsqueeze(0).clone()
    len_spectro = cloned.shape[2]
    T = max(1, min(T, int(len_spectro * p / num_masks)))

    for i in range(0, num_masks):
        t = np.random.randint(0, T)
        t_zero = np.random.randint(0, len_spectro - t)

        # avoids randint error if values are equal and range is empty
        if t == 0:
            return cloned.squeeze(0)

        cloned[0][:, t_zero:t_zero + t] = pad_value
    return cloned.squeeze(0)
