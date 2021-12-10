# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn

import espresso.tools.utils as speech_utils


class ConvBNReLU(nn.Module):
    """Sequence of convolution-BatchNorm-ReLU layers."""

    def __init__(self, out_channels, kernel_sizes, strides, in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_channels = in_channels

        num_layers = len(out_channels)
        assert num_layers == len(kernel_sizes) and num_layers == len(strides)

        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for i in range(num_layers):
            self.convolutions.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i - 1],
                    self.out_channels[i],
                    self.kernel_sizes[i],
                    self.strides[i],
                )
            )
            self.batchnorms.append(nn.BatchNorm2d(out_channels[i]))

    def output_lengths(self, in_lengths: Union[torch.Tensor, int]):
        out_lengths = in_lengths
        for stride in self.strides:
            if isinstance(stride, (list, tuple)):
                assert len(stride) > 0
                s = stride[0]
            else:
                assert isinstance(stride, int)
                s = stride
            if version.parse(torch.__version__) >= version.parse(
                "1.8.0"
            ) and isinstance(in_lengths, torch.Tensor):
                out_lengths = torch.div(out_lengths + s - 1, s, rounding_mode="floor")
            else:
                out_lengths = (out_lengths + s - 1) // s
        return out_lengths

    def forward(self, src, src_lengths):
        # B X T X C -> B X (input channel num) x T X (C / input channel num)
        x = src.view(
            src.size(0),
            src.size(1),
            self.in_channels,
            src.size(2) // self.in_channels,
        ).transpose(1, 2)
        for conv, bn in zip(self.convolutions, self.batchnorms):
            x = F.relu(bn(conv(x)))
        # B X (output channel num) x T X C' -> B X T X (output channel num) X C'
        x = x.transpose(1, 2)
        # B X T X (output channel num) X C' -> B X T X C
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x, x_lengths, padding_mask


def Convolution2d(in_channels, out_channels, kernel_size, stride):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = (kernel_size[0], kernel_size[0])
    else:
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = (stride[0], stride[0])
    else:
        assert isinstance(stride, int)
        stride = (stride, stride)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    m = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
    )
    return m
