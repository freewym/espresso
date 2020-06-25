# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GenerateLogProbsForDecoding(nn.Module):
    def __init__(self, models, retain_dropout=False, apply_log_softmax=False):
        """Generate the neural network's output intepreted as log probabilities
        for decoding with Kaldi.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            apply_log_softmax (bool, optional): apply log-softmax on top of the
                network's output (default: False)
        """
        super().__init__()
        from fairseq.sequence_generator import EnsembleModel
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.retain_dropout = retain_dropout
        self.apply_log_softmax = apply_log_softmax

        if not self.retain_dropout:
            self.model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        return self._generate(sample, **kwargs)

    def _generate(self, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        bsz = src_tokens.size(0)

        # compute the encoder output
        encoder_outs = self.model.forward_encoder(net_input)
        logits = encoder_outs[0].encoder_out.transpose(0, 1).float()  # T x B x V -> B x T x V
        assert logits.size(0) == bsz
        padding_mask = encoder_outs[0].encoder_padding_mask.t() \
            if encoder_outs[0].encoder_padding_mask is not None else None
        if self.apply_log_softmax:
            return F.log_softmax(logits, dim=-1), padding_mask
        return logits, padding_mask
