# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from espresso.modules.relative_positional_embedding import RelativePositionalEmbedding
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.conformer_layer import ConvolutionModule, FeedForwardModule
from fairseq.modules.fairseq_dropout import FairseqDropout


class ConformerWithRelativePositionalEmbeddingEncoderLayerBase(nn.Module):
    """Conformer encoder layer block based on https://arxiv.org/abs/2005.08100, with optional relative positional embedding."""

    def __init__(
        self,
        cfg,
        return_fc=False,
        positional_embedding: Optional[RelativePositionalEmbedding] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.ffn1 = FeedForwardModule(
            input_feat=self.embed_dim,
            hidden_units=cfg.encoder.ffn_embed_dim,
            dropout1=cfg.activation_dropout,
            dropout2=cfg.dropout,
            activation_fn="swish",
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim, cfg, positional_embedding=positional_embedding
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )

        self.conv_module = ConvolutionModule(
            embed_dim=self.embed_dim,
            channels=self.embed_dim,
            depthwise_kernel_size=cfg.encoder.depthwise_conv_kernel_size,
            dropout=cfg.dropout,
            activation_fn="swish",
        )

        self.ffn2 = FeedForwardModule(
            input_feat=self.embed_dim,
            hidden_units=cfg.encoder.ffn_embed_dim,
            dropout1=cfg.activation_dropout,
            dropout2=cfg.dropout,
            activation_fn="swish",
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_self_attention(self, embed_dim, cfg, positional_embedding=None):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
            positional_embedding=positional_embedding,
        )

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = x + residual

        residual = x
        x = self.ffn2(x)
        fc_result = x
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x


# backward compatible with the legacy argparse format
class ConformerWithRelativePositionalEmbeddingEncoderLayer(
    ConformerWithRelativePositionalEmbeddingEncoderLayerBase
):
    def __init__(
        self, args, positional_embedding: Optional[RelativePositionalEmbedding] = None
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            positional_embedding=positional_embedding,
        )
        self.args = args

    def build_self_attention(self, embed_dim, args, positional_embedding=None):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            positional_embedding=positional_embedding,
        )
