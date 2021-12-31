# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class TransducerBaseDecoder(nn.Module):
    def __init__(
        self,
        models,
        dictionary,
        max_len=0,
        max_num_expansions_per_step=2,
        temperature=1.0,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        print_alignment=False,
        **kwargs,
    ):
        """Decode given speech audios.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            dictionary (~fairseq.data.Dictionary): dictionary
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence) (default: 0, no limit)
            max_num_expansions_per_step (int, optional): the maximum number of
                non-blank expansions in a single time step (default: 2)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            lm_model (fairseq.models.FairseqLanguageModel, optional): LM model for LM fusion (default: None)
            lm_weight (float, optional): LM weight for LM fusion (default: 1.0)
            print_alignment (bool, optional): if True returns alignments (default: False)
        """
        super().__init__()
        self.model = models[0]  # currently only support single models
        self.eos = dictionary.eos() if eos is None else eos
        self.blank = dictionary.bos()  # we make the optional BOS symbol as blank
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos, self.blank})
            if symbols_to_strip_from_output is not None
            else {self.eos, self.blank}
        )
        self.vocab_size = len(dictionary)
        self.beam_size = 1  # child classes can overwrite it
        self.max_len = max_len
        self.max_num_expansions_per_step = max_num_expansions_per_step
        assert (
            max_num_expansions_per_step > 0
        ), "--max-num-expansions-per-step must be at least 1"
        self.temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"
        self.print_alignment = print_alignment

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.vocab_nonblank_mask = torch.ones(
                (self.vocab_size,), dtype=torch.bool
            )  # V
            self.vocab_nonblank_mask[self.blank] = False

            if (
                len(self.lm_model.decoder.dictionary) == self.vocab_size - 1
            ):  # LM doesn't have blank symbol
                self.no_blank_in_lm = True
                logger.info(
                    "the LM's vocabulary has 1 less symbol than that of the ASR model. Assuming it is blank symbol."
                )
            else:  # another symbol (e.g., pad) is sharing the same index with blank in the ASR model
                assert len(self.lm_model.decoder.dictionary) == self.vocab_size
                self.no_blank_in_lm = False

            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        if self.lm_model is not None:
            self.lm_model.cuda()

        return self

    @torch.no_grad()
    def decode(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate a batch of 1-best hypotheses. Match the API of other fairseq generators.
        Normally called for validation during training.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """API to be invoked from :func:`~fairseq.tasks.fairseq_task.FairseqTask.inference_step()`"""
        bos_token = kwargs.get("bos_token", None)
        tokens, scores, alignments = self._generate(sample, bos_token=bos_token)
        bsz = tokens.size(0)

        # list of completed sentences
        # see :class:`~fairseq.sequence_generator.SequenceGenerator` for specifications
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )
        for i in range(bsz):
            finalized[i].append(
                {
                    "tokens": tokens[i, :],
                    "score": scores[i],
                    "attention": None,
                    "alignment": alignments[i, :, :]
                    if self.print_alignment and alignments is not None
                    else None,
                }
            )

        return finalized

    @torch.no_grad()
    def _generate(
        self, sample: Dict[str, Dict[str, Tensor]], bos_token: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # should return a tuple of tokens, scores and alignments
        raise NotImplementedError
