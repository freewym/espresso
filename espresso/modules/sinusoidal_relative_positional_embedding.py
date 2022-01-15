# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn


class SinusoidalRelativePositionalEmbedding(nn.Module):
    """This module produces sinusoidal relative positional embeddings of any length."""

    def __init__(
        self,
        embedding_dim,
        padding_idx: Optional[int] = None,
        init_size=1024,
        max_size: Optional[int] = None,
        scale_embedding=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_scale = embedding_dim ** -0.5 if scale_embedding else 1.0
        self.weight = (
            self.embedding_scale
            * SinusoidalRelativePositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            )
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_size = max_size

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        seq_len: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal relative embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".

        Positive when keys are to the right of the query, and negative otherwise.
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(seq_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb_pos = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(seq_len, -1)
        emb_neg = torch.cat([torch.sin(-emb), torch.cos(-emb)], dim=1).view(seq_len, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb_pos = torch.cat([emb_pos, torch.zeros(seq_len, 1)], dim=1)
            emb_neg = torch.cat([emb_neg, torch.zeros(seq_len, 1)], dim=1)
        emb_neg = torch.flip(emb_neg, [0])
        emb_pos = emb_pos[1:]
        emb = torch.cat([emb_neg, emb_pos], dim=0)
        if padding_idx is not None:
            emb = torch.cat([emb.new_zeros([padding_idx + 1, emb.size(1)]), emb], dim=0)
        return emb

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """
        Args:
            input (torch.Tensor): input tensor of shape `(batch_size, seq_len)`
            incremental_state (dict, optional): dictionary used for storing state during
                :ref:`Incremental decoding` or training a streaming model. No use here.
            positions (torch.Tenser, optional): not used in this function

        Returns:
            relative posotional embedding for key of shape `(batch_size, 2*seq_len-1, embed_dim)`,
                where `seq_len` is the length of key
        """
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_positions = self.weight.size(0)
        if self.padding_idx is not None:
            max_positions -= self.padding_idx + 1
        if self.weight is None or (
            2 * seq_len - 1 > max_positions
            and (self.max_size is None or seq_len <= self.max_size)
        ):
            # recompute/expand embeddings if needed
            self.weight = (
                self.embedding_scale
                * SinusoidalRelativePositionalEmbedding.get_embedding(
                    seq_len, self.embedding_dim, self.padding_idx
                )
            )
            max_positions = self.weight.size(0)
            if self.padding_idx is not None:
                max_positions -= self.padding_idx + 1

        self.weight = self.weight.to(self._float_tensor)

        start = max_positions // 2 - seq_len + 1
        end = max_positions // 2 + seq_len
        positions = torch.arange(start, end)
        if self.max_size is not None and seq_len > self.max_size:
            positions = positions.clamp(min=0, max=max_positions - 1)
        if self.padding_idx is not None:
            positions = positions + (self.padding_idx + 1)

        used_weight = self.weight[positions, :]
        if self.onnx_trace:
            return used_weight.unsqueeze(0).repeat(bsz, 1, 1)
        else:
            return used_weight.expand(bsz, -1, -1)
