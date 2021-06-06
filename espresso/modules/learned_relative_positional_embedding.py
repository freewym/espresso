# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class LearnedRelativePositionalEmbedding(nn.Embedding):
    """
    This module learns relative positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, embedding_dim: int, padding_idx: Optional[int] = None, max_size: int = 1024):
        num_embeddings = 2 * max_size - 1
        if padding_idx is not None:
            num_embeddings += padding_idx + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if padding_idx is not None:
            self.max_positions = self.num_embeddings - padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.normal_(self.weight, mean=0.0, std=self.embedding_dim ** -0.5)
        nn.init.xavier_uniform_(self.weight)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0.0)
    
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
                :ref:`Incremental decoding` or training a streaming model
            positions (torch.Tenser, optional): position ids passed in

        Returns:
            relative posotional embedding for key of shape `(batch_size, 2*src_len-1, embed_dim)`,
                where `src_len` is the length of key
        """
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            bsz, seq_len = input.size(0), input.size(1)
            src_len = seq_len
            state = self.get_incremental_state(incremental_state, "attn_state")
            if state is not None and "prev_key" in state:
                src_len = src_len + state["prev_key"].size(2)  # prev_key is of shape `(bsz, num_heads, prev_key_len, head_dim)

            start = self.max_positions // 2 - src_len + 1
            end = self.max_positions // 2 + src_len
            positions = torch.arange(start, end).expand(bsz, -1, -1)
            if src_len > self.max_size:
                positions = positions.clamp(min=0, max=self.max_positions - 1)
            if self.padding_idx is not None:
                positions = positions + (self.padding_idx + 1)

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
