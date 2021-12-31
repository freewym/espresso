# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import espresso.tools.utils as speech_utils
from espresso.models.speech_lstm import SpeechLSTMDecoder
from espresso.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    SpeechTransformerEncoderBase,
    SpeechTransformerTransducerConfig,
)
from espresso.modules import ConvBNReLU
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    BaseFairseqModel,
    FairseqDecoder,
    FairseqEncoder,
    register_model,
)
from fairseq.models.fairseq_model import check_type
from fairseq.models.lstm import Linear
from fairseq.models.transformer import Embedding
from fairseq.modules import LayerNorm

logger = logging.getLogger(__name__)


@register_model(
    "speech_transformer_transducer_base", dataclass=SpeechTransformerTransducerConfig
)
class SpeechTransformerTransducerModelBase(BaseFairseqModel):
    """
    Speech Transducer model from `"Sequence Transduction with Recurrent Neural Networks" (Graves, 2012)
    <https://arxiv.org/abs/1211.3711>`_.

    Args:
        encoder (SpeechTransformerEncoderBase): the encoder
        decoder (SpeechLSTMDecoder): the decoder (or called "predictor")


    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        check_type(self.encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

        self.proj_encoder = Linear(cfg.encoder.embed_dim, cfg.joint_dim)
        self.laynorm_proj_encoder = LayerNorm(cfg.joint_dim, export=cfg.export)
        self.proj_decoder = Linear(cfg.decoder.hidden_size, cfg.joint_dim)
        self.laynorm_proj_decoder = LayerNorm(cfg.joint_dim, export=cfg.export)
        assert hasattr(self.decoder, "embed_tokens")
        if cfg.share_decoder_input_output_embed:
            assert (
                cfg.joint_dim == cfg.decoder.embed_dim
            ), "joint_dim and decoder.embed_dim must be the same if the two embeddings are to be shared"
            self.fc_out = nn.Linear(
                self.decoder.embed_tokens.embedding_dim,
                self.decoder.embed_tokens.num_embeddings,
                bias=False,
            )
            self.fc_out.weight = self.decoder.embed_tokens.weight
        else:
            self.fc_out = nn.Linear(
                cfg.joint_dim, self.decoder.embed_tokens.num_embeddings, bias=False
            )
            nn.init.normal_(self.fc_out.weight, mean=0, std=cfg.joint_dim ** -0.5)
            self.fc_out = nn.utils.weight_norm(self.fc_out, name="weight")

        self.cfg = cfg
        self.num_updates = 0

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser,
            SpeechTransformerTransducerConfig(),
            delete_default=False,
            with_prefix="",
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))

        if cfg.max_source_positions is None:
            cfg.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if cfg.max_target_positions is None:
            cfg.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
        )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        out_channels = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_channels, type=int
        )
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_kernel_sizes, type=int
        )
        strides = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_strides, type=int
        )
        logger.info(
            "input feature dimension: {}, channels: {}".format(
                task.feat_dim, task.feat_in_channels
            )
        )
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = (
            ConvBNReLU(
                out_channels,
                kernel_sizes,
                strides,
                in_channels=task.feat_in_channels,
            )
            if out_channels is not None
            else None
        )

        transformer_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                transformer_encoder_input_size = (
                    transformer_encoder_input_size + s - 1
                ) // s
            transformer_encoder_input_size *= out_channels[-1]
        else:
            transformer_encoder_input_size = task.feat_dim

        encoder_transformer_context = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.transformer_context,
            type=int,
        )
        if encoder_transformer_context is not None:
            assert len(encoder_transformer_context) == 2
            for i in range(2):
                assert encoder_transformer_context[i] is None or (
                    isinstance(encoder_transformer_context[i], int)
                    and encoder_transformer_context[i] >= 0
                )

        encoder = cls.build_encoder(
            cfg,
            conv_layers_before=conv_layers,
            input_size=transformer_encoder_input_size,
            transformer_context=encoder_transformer_context,
        )
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
        decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        return cls(cfg, encoder, decoder)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(
        cls, cfg, conv_layers_before=None, input_size=83, transformer_context=None
    ):
        return SpeechTransformerEncoderBase(
            cfg,
            conv_layers_before=conv_layers_before,
            input_size=input_size,
            transformer_context=transformer_context,
        )

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return SpeechLSTMDecoder(
            tgt_dict,
            embed_dim=cfg.decoder.embed_dim,
            hidden_size=cfg.decoder.hidden_size,
            out_embed_dim=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.layers,
            dropout_in=cfg.decoder.dropout_in,
            dropout_out=cfg.decoder.dropout_out,
            residual=cfg.decoder.residual,
            pretrained_embed=embed_tokens,
            share_input_output_embed=True,  # disallow fc_out in decoder
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        """
        Run the forward pass for a transformer transducer model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        Returns:
            x (Tensor): output of shape `(batch, src_len, tgt_len, vocab_size)`
            encoder_out_lengths (Tensor): encoder output lengths of shape `(batch)`
        """
        x, encoder_out_lengths = self.extract_features(
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens=return_all_hiddens,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, encoder_out_lengths

    def extract_features(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        **kwargs
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            features (Tensor): output of shape `(batch, src_len, tgt_len, joint_dim)`
            encoder_out_lengths (Tensor): encoder output lengths of shape `(batch)`
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder.extract_features(prev_output_tokens, **kwargs)
        enc_out = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x C
        dec_out = decoder_out[0]  # B x U x H
        features = self.joint(
            enc_out, dec_out, apply_output_layer=False
        )  # B x T x U x J
        return features, encoder_out["src_lengths"][0]

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.fc_out(features)

    def joint(self, encoder_out, decoder_out, apply_output_layer: bool = False):
        """
        Joint the output from the encoder and the decoder after projections.

        Args:
            encoder_out (Tensor): encoder output of shape `(batch, src_len, embed_dim)`
            decoder_out (Tensor): decoder output of shape `(batch, tgt_len, hidden_size)`
            apply_output_layer (bool, optional): whether to apply the output layer
                (default: False)

        Returns:
            out (Tensor): output of shape `(batch, src_len, tgt_len, joint_dim|vocab_size)`
        """
        out = F.relu(
            self.laynorm_proj_encoder(self.proj_encoder(encoder_out.unsqueeze(2)))
            + self.laynorm_proj_decoder(self.proj_decoder(decoder_out.unsqueeze(1)))
        )
        if apply_output_layer:
            out = self.output_layer(out)

        return out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)
