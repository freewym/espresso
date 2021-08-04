# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    TransformerDecoder,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from espresso.modules import TransformerWithRelativePositionalEmbeddingDecoderLayer


logger = logging.getLogger(__name__)


class SpeechTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        scheduled_sampling_rate_scheduler=None,
    ):
        is_no_token_positional_embeddings_changed = False
        if not args.no_token_positional_embeddings and args.decoder_relative_positional_embeddings:
            args.no_token_positional_embeddings = True
            is_no_token_positional_embeddings_changed = True
            logger.info("disabled decoder's absolute positional embeddings as decoder_relative_positional_embeddings is True.")
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        if is_no_token_positional_embeddings_changed:
            args.no_token_positional_embeddings = not args.no_token_positional_embeddings

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler
        for layer in self.layers:
            if isinstance(layer, TransformerWithRelativePositionalEmbeddingDecoderLayer):
                layer.need_attn = False  # make validation fast

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerWithRelativePositionalEmbeddingDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        if self.training and alignment_layer is None:  # no attention tensors during training to save memory
            alignment_layer = self.num_layers  # can be any value no less than this
        if self.training and self.scheduled_sampling_rate_scheduler is not None:
            epoch = kwargs.get("epoch", 1)
            sampling_prob = self.scheduled_sampling_rate_scheduler.step(epoch)
            if sampling_prob < 1.0:  # apply scheduled sampling
                assert not features_only
                return self._forward_with_scheduled_sampling(
                    prev_output_tokens, sampling_prob, encoder_out=encoder_out,
                    incremental_state={},  # use empty dict to preserve forward state
                    full_context_alignment=full_context_alignment,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def _forward_with_scheduled_sampling(
        self,
        prev_output_tokens,
        sampling_prob,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        bsz, seqlen = prev_output_tokens.size()
        outs = []
        pred = None
        for step in range(seqlen):
            if step > 0:
                sampling_mask = torch.rand(
                    [bsz, 1], device=prev_output_tokens.device,
                ).lt(sampling_prob)
                feed_tokens = torch.where(
                    sampling_mask, prev_output_tokens[:, step:step + 1], pred,
                )
            else:
                feed_tokens = prev_output_tokens[:, step:step + 1]  # B x 1
            x, _ = self.extract_features(
                feed_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                full_context_alignment=full_context_alignment,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
            )
            x = self.output_layer(x)  # B x 1 x V
            outs.append(x)
            pred = x.argmax(-1)  # B x 1
        x = torch.cat(outs, dim=1)  # B x T x V
        return x, None

    def masked_copy_incremental_state(self, incremental_state, another_cached_state, mask):
        raise NotImplementedError
