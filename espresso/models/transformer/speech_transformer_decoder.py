# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from espresso.models.transformer import SpeechTransformerConfig
from espresso.modules import (
    RelativePositionalEmbedding,
    TransformerWithRelativePositionalEmbeddingDecoderLayerBase,
)
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import Linear, TransformerDecoderBase
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

logger = logging.getLogger(__name__)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "SpeechTransformerDecoderBase":
        return "SpeechTransformerDecoder"
    else:
        return module_name


class SpeechTransformerDecoderBase(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        scheduled_sampling_rate_scheduler=None,
    ):
        is_no_token_positional_embeddings_changed = False
        if (
            not cfg.no_token_positional_embeddings
            and cfg.decoder.relative_positional_embeddings
        ):
            cfg.no_token_positional_embeddings = True
            is_no_token_positional_embeddings_changed = True
            logger.info(
                "disabled decoder's absolute positional embeddings as decoder_relative_positional_embeddings is True."
            )

        self.cfg = cfg
        super(TransformerDecoderBase, self).__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if cfg.decoder.relative_positional_embeddings:
            if cfg.decoder.learned_pos:
                rel_pos_embed_list = [
                    RelativePositionalEmbedding(
                        cfg.decoder.embed_dim,
                        padding_idx=None,
                        max_size=cfg.max_target_positions,
                        learned=True,
                    )
                    for _ in range(cfg.decoder.layers)
                ]
            else:
                rel_pos_embed = RelativePositionalEmbedding(
                    cfg.decoder.embed_dim,
                    padding_idx=None,
                    max_size=None,
                    learned=False,
                )
                # single instance referenced across layers
                rel_pos_embed_list = [rel_pos_embed] * cfg.decoder.layers
        else:
            rel_pos_embed_list = [None] * cfg.decoder.layers

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(
                    cfg, no_encoder_attn, positional_embedding=rel_pos_embed_list[i]
                )
                for i in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

        if is_no_token_positional_embeddings_changed:
            cfg.no_token_positional_embeddings = not cfg.no_token_positional_embeddings

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler
        for layer in self.layers:
            if isinstance(
                layer, TransformerWithRelativePositionalEmbeddingDecoderLayerBase
            ):
                layer.need_attn = False  # make validation fast

    def build_decoder_layer(
        self,
        cfg,
        no_encoder_attn=False,
        positional_embedding: Optional[RelativePositionalEmbedding] = None,
    ):
        layer = TransformerWithRelativePositionalEmbeddingDecoderLayerBase(
            cfg,
            no_encoder_attn=no_encoder_attn,
            positional_embedding=positional_embedding,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
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

        if (
            self.training and alignment_layer is None
        ):  # no attention tensors during training to save memory
            alignment_layer = self.num_layers  # can be any value no less than this
        if self.training and self.scheduled_sampling_rate_scheduler is not None:
            epoch = kwargs.get("epoch", 1)
            sampling_prob = self.scheduled_sampling_rate_scheduler.step(epoch)
            if sampling_prob < 1.0:  # apply scheduled sampling
                assert not features_only
                return self._forward_with_scheduled_sampling(
                    prev_output_tokens,
                    sampling_prob,
                    encoder_out=encoder_out,
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
                    [bsz, 1],
                    device=prev_output_tokens.device,
                ).lt(sampling_prob)
                feed_tokens = torch.where(
                    sampling_mask,
                    prev_output_tokens[:, step : step + 1],
                    pred,
                )
            else:
                feed_tokens = prev_output_tokens[:, step : step + 1]  # B x 1
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

    def initialize_cached_state(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz = prev_output_tokens.size(0)
        x = self.embed_tokens(prev_output_tokens)
        num_heads = self.cfg.decoder.attention_heads
        embed_dim = self.cfg.decoder.embed_dim
        zero_states = x.new_zeros(
            len(self.layers), bsz, num_heads, 0, embed_dim // num_heads
        )
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_key": zero_states,
                "prev_value": zero_states,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[List[Tensor]]]:
        prev_key, prev_value, prev_key_padding_mask = [], [], []
        for layer in self.layers:
            attn_state = layer.self_attn.get_incremental_state(
                incremental_state, "attn_state"
            )
            assert attn_state is not None
            assert attn_state["prev_key"] is not None
            prev_key.append(attn_state["prev_key"])
            assert attn_state["prev_value"] is not None
            prev_value.append(attn_state["prev_value"])
            if len(attn_state) >= 3 and attn_state["prev_key_padding_mask"] is not None:
                prev_key_padding_mask.append(attn_state["prev_key_padding_mask"])

        if len(prev_key_padding_mask) == 0:
            prev_key_padding_mask = None
        else:
            assert len(prev_key_padding_mask) == len(prev_key)

        return prev_key, prev_value, prev_key_padding_mask

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        if key != "cached_state":
            return super().get_incremental_state(incremental_state, key)
        if incremental_state is None:
            return None

        prev_key, prev_value, prev_key_padding_mask = self.get_cached_state(
            self, incremental_state
        )
        cached_state: Dict[str, Optional[Tensor]] = {
            "prev_key": torch.stack(prev_key),
            "prev_value": torch.stack(prev_value),
        }
        if prev_key_padding_mask is not None:
            cached_state["prev_key_padding_mask"] = torch.stack(prev_key_padding_mask)

        return cached_state

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        if key != "cached_state":
            return super().set_incremental_state(incremental_state, key, value)

        if incremental_state is not None:
            pre_key = value["pre_key"]
            pre_value = value["pre_value"]
            for i, layer in enumerate(self.layers):
                attn_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": pre_key[i] if pre_key is not None else None,
                    "prev_value": pre_value[i] if pre_value is not None else None,
                }
                if len(value) >= 3:
                    prev_key_padding_mask = value["prev_key_padding_mask"]
                    attn_state["prev_key_padding_mask"] = (
                        prev_key_padding_mask[i]
                        if prev_key_padding_mask is not None
                        else None
                    )
                layer.self_attn.set_incremental_state(
                    incremental_state, "attn_state", attn_state
                )

        return incremental_state

    def masked_copy_cached_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        src_cached_state: Tuple[Optional[Union[List[torch.Tensor], torch.Tensor]]],
        mask: Tensor,
    ):
        if incremental_state is None:
            assert src_cached_state is None or len(src_cached_state) == 0
            return

        prev_key, prev_value, prev_key_padding_mask = self.get_cached_state(
            incremental_state
        )
        src_prev_key, src_prev_value, src_prev_key_padding_mask = (
            src_cached_state[0],
            src_cached_state[1],
            src_cached_state[2],
        )
        # pad one more time step, assuming src_cached_state is just one step behind
        src_prev_key = [F.pad(src_p, (0, 0, 0, 1)) for src_p in src_prev_key]
        src_prev_value = [F.pad(src_p, (0, 0, 0, 1)) for src_p in src_prev_value]
        if src_prev_key_padding_mask is not None:
            src_prev_key_padding_mask = [
                F.pad(src_p, (0, 1)) for src_p in src_prev_key_padding_mask
            ]

        def masked_copy_state(state: Optional[Tensor], src_state: Optional[Tensor]):
            if state is None:
                assert src_state is None
                return None
            else:
                assert (
                    state.size(0) == mask.size(0)
                    and src_state is not None
                    and state.size() == src_state.size()
                )
                state[mask, ...] = src_state[mask, ...]
                return state

        prev_key = [
            masked_copy_state(p, src_p) for (p, src_p) in zip(prev_key, src_prev_key)
        ]
        prev_value = [
            masked_copy_state(p, src_p)
            for (p, src_p) in zip(prev_value, src_prev_value)
        ]
        if prev_key_padding_mask is None:
            prev_key_padding_mask = src_prev_key_padding_mask
        else:
            assert src_prev_key_padding_mask is not None
            prev_key_padding_mask = [
                masked_copy_state(p, src_p)
                for (p, src_p) in zip(prev_key_padding_mask, src_prev_key_padding_mask)
            ]

        cached_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_key": torch.stack(prev_key),
                "prev_value": torch.stack(prev_value),
            },
        )
        if prev_key_padding_mask is not None:
            cached_state["prev_key_padding_mask"] = torch.stack(prev_key_padding_mask)

        self.set_incremental_state(incremental_state, "cached_state", cached_state)


class SpeechTransformerDecoder(SpeechTransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        scheduled_sampling_rate_scheduler=None,
    ):
        self.args = args
        super().__init__(
            SpeechTransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            SpeechTransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(
        self,
        args,
        no_encoder_attn=False,
        positional_embedding: Optional[RelativePositionalEmbedding] = None,
    ):
        return super().build_decoder_layer(
            SpeechTransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            positional_embedding=positional_embedding,
        )
