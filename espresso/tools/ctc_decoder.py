# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.data.data_utils import collate_tokens
from fairseq.utils import strip_pad


class CTCDecoder(nn.Module):
    def __init__(
        self,
        models,
        dictionary,
        symbols_to_strip_from_output=None,
        lm_model: Optional[str] = None,
        lexicon: Optional[str] = None,
        beam_size=50,
        lm_weight=2.0,
        word_score=-1.0,
        print_alignment=False,
        **kwargs,
    ):
        """Decode given speech audios for CTC models.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            dictionary (~fairseq.data.Dictionary): dictionary
            lm_model (str, optional): if this is provided, use KenLM to compute WER
            lexicon (str, optional): lexicon to use with KenLM model
            beam_size (int, optional): beam width (default: 50)
            lm_weight (float, optional): LM weight to use with KenLM model (default: 2.0)
            word_score (float, optional): LM word score to use with KenLM model (default: -1.0)
            print_alignment (bool, optional): if True returns alignments (default: False)
        """
        super().__init__()
        self.model = models[0]  # currently only support single models
        self.blank = dictionary.bos()  # we make the optional BOS symbol as blank
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.symbols_to_strip_from_output = symbols_to_strip_from_output
        self.print_alignment = print_alignment

        self.model.eval()

        if lm_model is not None and lm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = lm_model
            dec_args.lexicon = lexicon
            dec_args.beam = beam_size
            dec_args.beam_size_token = min(beam_size, len(dictionary))
            dec_args.beam_threshold = min(beam_size, len(dictionary))
            dec_args.lm_weight = lm_weight
            dec_args.word_score = word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, dictionary)
        else:
            self.w2l_decoder = None

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def decode(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate a batch of 1-best hypotheses. Match the API of other fairseq generators.
        Normally called for validation during training.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        assert not self.print_alignment

        tokens_list, scores_list, _ = self._generate(sample, **kwargs)
        bsz = len(tokens_list)
        tokens = collate_tokens(
            [tokens_list[i][0, :] for i in range(bsz)],
            pad_idx=self.pad,
        )  # B x U
        scores = torch.as_tensor([scores[0] for scores in scores_list])  # B

        return tokens, scores, None

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """API to be invoked from :func:`~fairseq.tasks.fairseq_task.FairseqTask.inference_step()`"""
        tokens_list, scores_list, alignments_list = self._generate(sample)
        bsz = len(tokens_list)

        # list of completed sentences
        # see :class:`~fairseq.sequence_generator.SequenceGenerator` for specifications
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )
        for i in range(bsz):
            for j in range(tokens_list[i].size(0)):
                finalized[i].append(
                    {
                        "tokens": strip_pad(tokens_list[i][j, :], self.pad),
                        "score": scores_list[i][j],
                        "attention": None,
                        "alignment": alignments_list[i][j, :]
                        if self.print_alignment
                        else None,
                    }
                )

        return finalized

    @torch.no_grad()
    def _generate(
        self, sample: Dict[str, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        bsz = src_tokens.size(0)

        net_output = self.model(**net_input)
        lprobs = self.model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # T x B x C
        if "src_lengths" in net_output:
            enc_out_lengths = net_output["src_lengths"][0]  # B
        else:
            if net_output["encoder_padding_mask"] is not None:
                non_padding_mask = ~net_output["encoder_padding_mask"][0]
                enc_out_lengths = non_padding_mask.long().sum(-1)
            else:
                enc_out_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        sequences_list, scores_list, alignments_list = [], [], []
        for i in range(bsz):
            sequences, scores, alignments = self._generate_one_example(
                lprobs[: enc_out_lengths[i], i, :]
            )
            sequences_list.append(sequences)
            scores_list.append(scores)
            alignments_list.append(alignments)

        return sequences_list, scores_list, alignments_list

    @torch.no_grad()
    def _generate_one_example(self, lprobs: Tensor):
        lprobs = lprobs.unsqueeze(0)  # B(=1) x T x V
        if self.w2l_decoder is not None:
            lprobs = lprobs.float().contiguous().cpu()
            one_best = self.w2l_decoder.decode(lprobs)[0][0]
            scores = one_best["score"].unsqueeze(0)  # B(=1)
            sequences = one_best["tokens"].unsqueeze(0)  # B(=1) x U
            alignments = torch.as_tensor(one_best["timesteps"]).unsqueeze(
                0
            )  # B(=1) x U
        else:
            scores, tokens = lprobs.max(-1)
            scores = scores.sum(1)  # B(=1)
            sequences = tokens.unique_consecutive()
            sequences = sequences[sequences != self.blank].unsqueeze(0)  # B(=1) x U
            alignments = []
            for i in range(tokens.size(1)):
                token = tokens[0, i]
                if token == self.blank:
                    continue
                if i == 0 or token != tokens[0, i - 1]:
                    alignments.append(i)
            alignments = torch.as_tensor(alignments).unsqueeze(0)  # B(=1) x U

        return sequences, scores, alignments
