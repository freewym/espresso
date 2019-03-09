# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from torch import nn
from torch.nn import Parameter

from fairseq import utils


class BaseAttention(nn.Module):
    """Base class for attention layers."""

    def __init__(self, query_dim, value_dim, embed_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        pass

    def forward(self, query, value, key_padding_mask=None, state=None):
        # query: bsz x q_hidden
        # value: len x bsz x v_hidden
        # key_padding_mask: len x bsz
        raise NotImplementedError


class BahdanauAttention(BaseAttention):
    """ Bahdanau Attention."""

    def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v = Parameter(torch.Tensor(embed_dim))
        self.normalize = normalize
        if self.normalize:
            self.b = Parameter(torch.Tensor(embed_dim))
            self.g = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.query_proj.weight.data.uniform_(-0.1, 0.1)
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)
        if self.normalize:
            nn.init.constant_(self.b, 0.)
            nn.init.constant_(self.g, math.sqrt(1. / self.embed_dim))

    def forward(self, query, value, key_padding_mask=None, state=None):
        # projected_query: 1 x bsz x embed_dim
        projected_query = self.query_proj(query).unsqueeze(0)
        key = self.value_proj(value) # len x bsz x embed_dim
        if self.normalize:
            # normed_v = g * v / ||v||
            normed_v = self.g * self.v / torch.norm(self.v)
            attn_scores = (normed_v * torch.tanh(projected_query + key + \
                self.b)).sum(dim=2) # len x bsz
        else:
            attn_scores = v * torch.tanh(projected_query + key).sum(dim=2)

        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = utils.softmax(attn_scores, dim=0,
            onnx_trace=self.onnx_trace).type_as(attn_scores)  # len x bsz

        # sum weighted value. context: bsz x value_dim
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores

        return context, attn_scores, next_state


class LuongAttention(BaseAttention):
    """ Luong Attention."""

    def __init__(self, query_dim, value_dim, embed_dim=None, scale=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.value_proj = nn.Linear(self.value_dim, self.query_dim, bias=False)
        self.scale = scale
        if self.scale:
            self.g = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        if self.scale:
            nn.init.constant_(self.g, 1.)

    def forward(self, query, value, key_padding_mask=None, state=None):
        query = query.unsqueeze(1)  # bsz x 1 x query_dim
        key = self.value_proj(value).transpose(0, 1) # bsz x len x query_dim
        attn_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        attn_scores = attn_scores.transpose(0, 1)  # len x bsz
        if self.scale:
            attn_scores = self.g * attn_scores

        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                key_padding_mask, float('-inf'),
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = utils.softmax(attn_scores, dim=0,
            onnx_trace=self.onnx_trace).type_as(attn_scores)  # len x bsz

        # sum weighted value. context: bsz x value_dim
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores

        return context, attn_scores, next_state

