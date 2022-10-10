# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

import espresso.tools.utils as speech_utils
from espresso.models.transformer import SpeechTransformerConfig
from espresso.modules import (
    ConformerWithRelativePositionalEmbeddingEncoderLayerBase,
    RelativePositionalEmbedding,
    TransformerWithRelativePositionalEmbeddingEncoderLayerBase,
)
from fairseq.data import data_utils
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import Linear, TransformerEncoderBase
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
    if module_name == "SpeechTransformerEncoderBase":
        return "SpeechTransformerEncoder"
    else:
        return module_name


class SpeechTransformerEncoderBase(TransformerEncoderBase):
    """
    Transformer encoder consisting of 2D convolution layers and
    *cfg.encoder.layers* layers. Each layer is a
    :class:`TransformerWithRelativePositionalEmbeddingEncoderLayer`.

    Args:
        cfg (FairseqDataclass): model configuration
        pre_encoder (~espresso.modules.ConvBNReLU): convolutions before
            transformer layers
        input_size (int, optional): dimension of the input to the transformer
            before being projected to cfg.encoder.embed_dim
    """

    def __init__(
        self,
        cfg,
        return_fc=False,
        pre_encoder=None,
        input_size=83,
    ):
        self.cfg = cfg
        super(TransformerEncoderBase, self).__init__(None)  # no src dictionary
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = cfg.encoder.embed_dim
        self.max_source_positions = cfg.max_source_positions

        self.pre_encoder = pre_encoder
        self.fc0 = Linear(input_size, embed_dim) if input_size != embed_dim else None

        self.embed_scale = (
            1.0
            if cfg.no_scale_embedding
            or self.fc0 is not None  # always diable scaling if fc0 is present
            else math.sqrt(embed_dim)
        )

        if (
            not cfg.no_token_positional_embeddings
            and cfg.encoder.relative_positional_embeddings
        ):
            logger.info(
                "disabled encoder's absolute positional embeddings as encoder_relative_positional_embeddings is True."
            )
        self.embed_positions = (
            PositionalEmbedding(
                self.output_lengths(self.max_source_positions),
                embed_dim,
                0,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            and not cfg.encoder.relative_positional_embeddings
            else None
        )

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if cfg.encoder.relative_positional_embeddings:
            if cfg.encoder.learned_pos:
                rel_pos_embed_list = [
                    RelativePositionalEmbedding(
                        cfg.encoder.embed_dim,
                        padding_idx=None,
                        max_size=self.output_lengths(cfg.max_source_positions),
                        learned=True,
                    )
                    for _ in range(cfg.encoder.layers)
                ]
            else:
                rel_pos_embed = RelativePositionalEmbedding(
                    cfg.encoder.embed_dim,
                    padding_idx=None,
                    max_size=None,
                    learned=False,
                )
                # single instance referenced across layers
                rel_pos_embed_list = [rel_pos_embed] * cfg.encoder.layers
        else:
            rel_pos_embed_list = [None] * cfg.encoder.layers

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_encoder_layer(
                    cfg, positional_embedding=rel_pos_embed_list[i]
                )
                for i in range(cfg.encoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before and cfg.encoder.layer_type != "conformer":
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.transformer_context = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.transformer_context,
            type=int,
        )
        if self.transformer_context is not None:
            assert len(self.transformer_context) == 2
            for i in range(2):
                assert self.transformer_context[i] is None or (
                    isinstance(self.transformer_context[i], int)
                    and self.transformer_context[i] >= 0
                )

        self.num_updates = 0

    def build_encoder_layer(
        self, cfg, positional_embedding: Optional[RelativePositionalEmbedding] = None
    ):
        if cfg.encoder.layer_type == "transformer":
            layer_cls = TransformerWithRelativePositionalEmbeddingEncoderLayerBase
        elif cfg.encoder.layer_type == "conformer":
            layer_cls = ConformerWithRelativePositionalEmbeddingEncoderLayerBase
        else:
            raise NotImplementedError
        layer = layer_cls(
            cfg, return_fc=self.return_fc, positional_embedding=positional_embedding
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

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def output_lengths(self, in_lengths):
        return (
            in_lengths
            if self.pre_encoder is None
            else self.pre_encoder.output_lengths(in_lengths)
        )

    def get_attn_mask(self, in_lengths):
        """
        Create attention mask according to sequence lengths and transformer context.

        Args:
            in_lengths (LongTensor): lengths of each input sequence of shape `(batch)`

        Returns:
            attn_mask (ByteTensor|BoolTensor, optional): self-attention mask of shape
            `(tgt_len, src_len)`, where `tgt_len` is the length of output and `src_len`
            is the length of input, though here both are equal to `seq_len`.
            `attn_mask[tgt_i, src_j] = 1` means that when calculating the
            embedding for `tgt_i`, we exclude (mask out) `src_j`.
        """
        if self.cfg.encoder.chunk_size > 0:
            with data_utils.numpy_seed(self.num_updates):
                return ~speech_utils.chunk_streaming_mask(
                    in_lengths,
                    self.cfg.encoder.chunk_size,
                    left_window=self.cfg.encoder.chunk_left_window,
                    right_window=self.cfg.encoder.chunk_right_window,
                    always_partial_in_last=(not self.training),
                )

        if self.transformer_context is None or (
            self.transformer_context[0] is None and self.transformer_context[1] is None
        ):
            return None
        max_len = in_lengths.data.max()
        all_ones = in_lengths.ones([max_len, max_len], dtype=torch.bool)
        # at this point left and right context cannot be both None
        if self.transformer_context[0] is None:  # mask is a triu matrix
            return all_ones.triu(self.transformer_context[1] + 1)
        if self.transformer_context[1] is None:  # mask is a tril matrix
            return all_ones.tril(-self.transformer_context[0] - 1)
        return all_ones.triu(self.transformer_context[1] + 1) | all_ones.tril(
            -self.transformer_context[0] - 1
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens, src_lengths, return_all_hiddens)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.pre_encoder is not None:
            x, src_lengths, encoder_padding_mask = self.pre_encoder(
                src_tokens, src_lengths
            )
        else:
            x, encoder_padding_mask = (
                src_tokens,
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1)),
            )
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        if self.fc0 is not None:
            x = self.dropout_module(x)
            x = self.fc0(x)
        x = self.embed_scale * x
        if self.embed_positions is not None:
            # 0s in `~encoder_padding_mask` are used as pad_idx for positional embeddings
            x = x + self.embed_positions((~encoder_padding_mask).int())
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        attn_mask = self.get_attn_mask(src_lengths)

        # encoder layers
        for layer in self.layers:
            lr = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                attn_mask=attn_mask,
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],  # List[B]
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if len(name) > 0:
            name += "."
        old_pattern = "{}conv_layers_before".format(name)
        new_pattern = "{}pre_encoder".format(name)
        old_keys = []
        for k in state_dict:
            if k.startswith(old_pattern):
                old_keys.append(k)
        for old_key in old_keys:
            new_key = old_key.replace(old_pattern, new_pattern, 1)
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict


class SpeechTransformerEncoder(SpeechTransformerEncoderBase):
    def __init__(
        self,
        args,
        return_fc=False,
        pre_encoder=None,
        input_size=83,
    ):
        self.args = args
        super().__init__(
            SpeechTransformerConfig.from_namespace(args),
            return_fc=return_fc,
            pre_encoder=pre_encoder,
            input_size=input_size,
        )

    def build_encoder_layer(
        self, args, positional_embedding: Optional[RelativePositionalEmbedding] = None
    ):
        return super().build_encoder_layer(
            SpeechTransformerConfig.from_namespace(args),
            positional_embedding=positional_embedding,
        )
