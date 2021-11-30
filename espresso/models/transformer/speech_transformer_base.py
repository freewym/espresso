# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import espresso.tools.utils as speech_utils
from espresso.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    SpeechTransformerConfig,
    SpeechTransformerDecoderBase,
    SpeechTransformerEncoderBase,
)
from espresso.modules import ConvBNReLU
from espresso.tools.scheduled_sampling_rate_scheduler import (
    ScheduledSamplingRateScheduler,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model
from fairseq.models.transformer import TransformerModelBase

logger = logging.getLogger(__name__)


@register_model("speech_transformer_base", dataclass=SpeechTransformerConfig)
class SpeechTransformerModelBase(TransformerModelBase):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_. It adds 2D convolutions before
    transformer layers in the encoder to process speech input.

    Args:
        encoder (SpeechTransformerEncoderBase): the encoder
        decoder (SpeechTransformerDecoderBase): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, SpeechTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # -- TODO T96535332
        # bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        if cfg.max_source_positions is None:
            cfg.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if cfg.max_target_positions is None:
            cfg.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.input_dim, cfg.decoder.embed_path
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

        scheduled_sampling_rate_scheduler = ScheduledSamplingRateScheduler(
            cfg.scheduled_sampling_probs,
            cfg.start_scheduled_sampling_epoch,
        )

        encoder = cls.build_encoder(
            cfg,
            conv_layers_before=conv_layers,
            input_size=transformer_encoder_input_size,
            transformer_context=encoder_transformer_context,
        )
        decoder = cls.build_decoder(
            cfg,
            tgt_dict,
            decoder_embed_tokens,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )
        # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
        decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        return cls(cfg, encoder, decoder)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

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
    def build_decoder(
        cls, cfg, tgt_dict, embed_tokens, scheduled_sampling_rate_scheduler=None
    ):
        return SpeechTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        epoch=1,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            epoch=epoch,
        )
        return decoder_out
