# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from dataclasses import dataclass, field, fields
from typing import Optional

from omegaconf import II

from espresso.models.transformer.speech_transformer_config import SpeechEncoderConfig
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models.transformer.transformer_config import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    QuantNoiseConfig,
    TransformerConfig,
)
from fairseq.utils import safe_getattr, safe_hasattr

DEFAULT_MAX_SOURCE_POSITIONS = 10240
DEFAULT_MAX_TARGET_POSITIONS = 1024

_NAME_PARSER = r"(decoder|encoder|quant_noise)_(.*)"


@dataclass
class SpeechDecoderConfig(FairseqDataclass):
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained decoder embedding"}
    )
    embed_dim: int = field(default=48, metadata={"help": "decoder embedding dimension"})
    hidden_size: int = field(default=320, metadata={"help": "decoder hidden size"})
    layers: int = field(default=3, metadata={"help": "num decoder layers"})
    residual: bool = field(
        default=False,
        metadata={
            "help": "create residual connections for rnn decoder "
            "layers (starting from the 2nd layer), i.e., the actual "
            "output of such layer is the sum of its input and output"
        },
    )
    dropout_in: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for decoder input embedding"},
    )
    dropout_out: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for decoder output"},
    )


@dataclass
class SpeechTransformerTransducerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    adaptive_input: bool = field(
        default=False,
        metadata={"help": "if set, uses adaptive input"},
    )
    encoder: SpeechEncoderConfig = SpeechEncoderConfig()
    decoder: SpeechDecoderConfig = SpeechDecoderConfig()
    joint_dim: int = field(
        default=512, metadata={"help": "joint network output dimension"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    layernorm_embedding: bool = field(
        default=True, metadata={"help": "add layernorm to embedding"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )

    # options from other parts of the config
    max_source_positions: Optional[int] = II("task.max_source_positions")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")

    # We need to make this hierarchical dataclass like the flat namespace
    # __getattr__ and __setattr__ here allow backward compatibility
    # for subclasses of Transformer(Legacy) that depend on read/write on
    # the flat namespace.

    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        return TransformerConfig._copy_keys(args, cls, prefix, seen)

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = SpeechDecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, SpeechDecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = SpeechEncoderConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, SpeechEncoderConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
