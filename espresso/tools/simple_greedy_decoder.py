# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class SimpleGreedyDecoder(nn.Module):
    def __init__(
        self, models, dictionary, max_len_a=0, max_len_b=200, retain_dropout=False,
        temperature=1.0, for_validation=True,
    ):
        """Decode given speech audios with the simple greedy search.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            dictionary (~fairseq.data.Dictionary): dictionary
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            for_validation (bool, optional): indicate whether the decoder is
                used for validation. It affects how max_len is determined, and
                whether a tensor of lprobs is returned. If true, target should be
                not None
        """
        super().__init__()
        from fairseq.sequence_generator import EnsembleModel
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.vocab_size = len(dictionary)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"
        if not self.retain_dropout:
            self.model.eval()
        self.for_validation = for_validation

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def decode(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate a batch of translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        self.model.reset_incremental_state()
        return self._decode(sample, **kwargs)

    @torch.no_grad()
    def _decode(self, sample: Dict[str, Dict[str, Tensor]], bos_token: Optional[int] = None):
        encoder_input: Dict[str, Tensor] = {}
        for k, v in sample["net_input"].items():
            if k != "prev_output_tokens":
                encoder_input[k] = v
        src_tokens = encoder_input["src_tokens"]
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]

        encoder_outs = self.model.forward_encoder(
            src_tokens=encoder_input["src_tokens"],
            src_lengths=encoder_input["src_lengths"],
        )
        target = sample["target"]
        # target can only be None if not for validation
        assert target is not None or not self.for_validation
        max_encoder_output_length = encoder_outs[0].encoder_out.size(0)
        # for validation, make the maximum decoding length equal to at least the
        # length of target, and the length of encoder_out if possible; otherwise
        # max_len is obtained from max_len_a/b
        max_len = max(max_encoder_output_length, target.size(1)) \
            if self.for_validation else \
            min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )

        tokens = src_tokens.new(bsz, max_len + 2).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        # lprobs is only used when target is not None (i.e., for validation)
        lprobs = encoder_outs[0].encoder_out.new_full(
            (bsz, target.size(1), self.vocab_size), -np.log(self.vocab_size),
        ) if self.for_validation else None
        attn = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            is_eos = tokens[:, step].eq(self.eos)
            if step > 0 and is_eos.sum() == is_eos.size(0):
                # all predictions are finished (i.e., ended with eos)
                tokens = tokens[:, :step + 1]
                if attn is not None:
                    attn = attn[:, :, :step + 1]
                break
            log_probs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            )
            tokens[:, step + 1] = log_probs.argmax(-1)
            if step > 0:  # deal with finished predictions
                # make log_probs uniform if the previous output token is EOS
                # and add consecutive EOS to the end of prediction
                log_probs[is_eos, :] = -np.log(log_probs.size(1))
                tokens[is_eos, step + 1] = self.eos
            if self.for_validation and step < target.size(1):
                lprobs[:, step, :] = log_probs

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:
                    attn = avg_attn_scores.new(bsz, max_encoder_output_length, max_len + 2)
                attn[:, :, step + 1].copy_(avg_attn_scores)

        return tokens, lprobs, attn
