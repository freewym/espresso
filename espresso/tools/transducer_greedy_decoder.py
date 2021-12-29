# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import Tensor

from espresso.tools.transducer_base_decoder import TransducerBaseDecoder
from espresso.tools.utils import clone_cached_state


class TransducerGreedyDecoder(TransducerBaseDecoder):
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
        """Decode given speech audios with the simple greedy search.

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
        super().__init__(
            models,
            dictionary,
            max_len=max_len,
            max_num_expansions_per_step=max_num_expansions_per_step,
            temperature=temperature,
            eos=eos,
            symbols_to_strip_from_output=symbols_to_strip_from_output,
            lm_model=lm_model,
            lm_weight=lm_weight,
            print_alignment=print_alignment,
            **kwargs,
        )

        assert hasattr(self.model.decoder, "masked_copy_cached_state") and callable(
            self.model.decoder.masked_copy_cached_state
        ), "self.model.decoder should implement masked_copy_cached_state()"
        assert hasattr(self.model.decoder, "initialize_cached_state") and callable(
            self.model.decoder.initialize_cached_state
        ), "self.model.decoder should implement initialize_cached_state()"
        if self.lm_model is not None:
            assert hasattr(
                self.lm_model.decoder, "masked_copy_cached_state"
            ) and callable(
                self.lm_model.decoder.masked_copy_cached_state
            ), "self.lm_model.decoder should implement masked_copy_cached_state()"
            assert hasattr(self.model.decoder, "initialize_cached_state") and callable(
                self.lm_model.decoder.initialize_cached_state
            ), "self.lm_model.decoder should implement initialize_cached_state()"

    @torch.no_grad()
    def _generate(
        self, sample: Dict[str, Dict[str, Tensor]], bos_token: Optional[int] = None
    ):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        bsz, src_len = src_tokens.size()[:2]

        # compute the encoder output
        encoder_outs = self.model.encoder.forward_torchscript(net_input)
        enc_out = encoder_outs["encoder_out"][0].transpose(0, 1)  # B x T x C
        enc_out_lengths = encoder_outs["src_lengths"][0]  # B
        max_enc_out_length = enc_out_lengths.max().item()
        max_len = (
            min(max_enc_out_length, self.max_len)
            if self.max_len > 0
            else max_enc_out_length
        )

        # tokens stores predicted tokens
        tokens = src_tokens.new_full(
            (bsz, max_len, self.max_num_expansions_per_step + 1),
            self.blank,
            dtype=torch.long,
        )  # +1 for the last blank at each time step
        prev_nonblank_tokens = tokens.new_full(
            (bsz, 1), self.eos if bos_token is None else bos_token
        )  # B x 1
        # scores is used to store log-prob of emitting each token
        scores = enc_out.new_full(
            (bsz, max_len, self.max_num_expansions_per_step + 1), 0.0
        )

        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        # always make incremental_state non-empty
        self.model.decoder.initialize_cached_state(
            tokens, incremental_state=incremental_state
        )

        if self.lm_model is not None:
            lm_incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            # always make lm_incremental_state non-empty
            self.lm_model.decoder.initialize_cached_state(
                tokens, incremental_state=lm_incremental_state
            )

        for step in range(max_len):
            blank_mask = step >= enc_out_lengths  # B
            expansion_idx = 0

            while (
                not blank_mask.all()
                and expansion_idx < self.max_num_expansions_per_step + 1
            ):
                old_cached_state = clone_cached_state(
                    self.model.decoder.get_cached_state(incremental_state)
                )
                dec_out = self.model.decoder.extract_features(
                    prev_nonblank_tokens, incremental_state=incremental_state
                )[
                    0
                ]  # B x 1 x H
                logits = (
                    self.model.joint(
                        enc_out[:, step : step + 1, :], dec_out, apply_output_layer=True
                    )
                    .squeeze(2)
                    .squeeze(1)
                )  # B x 1 x 1 x V -> B x V
                lprobs = self.model.get_normalized_probs(
                    (logits.div_(self.temperature), None), log_probs=True
                )  # B x V

                if self.lm_model is not None:
                    old_lm_cached_state = clone_cached_state(
                        self.lm_model.decoder.get_cached_state(lm_incremental_state)
                    )
                    lm_prev_nonblank_tokens = (
                        torch.where(
                            prev_nonblank_tokens > self.blank,
                            prev_nonblank_tokens - 1,
                            prev_nonblank_tokens,
                        )
                        if self.no_blank_in_lm
                        else prev_nonblank_tokens
                    )
                    lm_out = self.lm_model(
                        lm_prev_nonblank_tokens, incremental_state=lm_incremental_state
                    )
                    lm_lprobs = self.lm_model.get_normalized_probs(
                        lm_out, log_probs=True
                    ).squeeze(
                        1
                    )  # B x 1 x V' -> B x V'

                    lprobs_no_blank = lprobs[:, self.vocab_nonblank_mask]  # B x (V - 1)
                    if not self.no_blank_in_lm:
                        lm_lprobs = lm_lprobs[
                            :, self.vocab_nonblank_mask
                        ]  # B x (V - 1)
                    # keep the nonblank probability mass unchanged after adding LM score
                    lprobs_with_lm_no_blank = (
                        lprobs_no_blank + self.lm_weight * lm_lprobs
                    )  # B x (V - 1)
                    log_scaling_factor = (
                        lprobs_no_blank.exp().sum(1).log()
                        - lprobs_with_lm_no_blank.exp().sum(1).log()
                    )  # B
                    lprobs_with_lm_no_blank += log_scaling_factor.unsqueeze(
                        1
                    )  # B x (V - 1)
                    lprobs[:, self.vocab_nonblank_mask] = lprobs_with_lm_no_blank

                if expansion_idx < self.max_num_expansions_per_step:
                    (
                        scores[:, step, expansion_idx],
                        tokens[:, step, expansion_idx],
                    ) = lprobs.max(-1)
                    scores[blank_mask, step, expansion_idx] = 0.0
                    blank_mask_local = tokens[:, step, expansion_idx] == self.blank
                    blank_mask.logical_or_(blank_mask_local)
                    tokens[blank_mask, step, expansion_idx] = self.blank
                    prev_nonblank_tokens[~blank_mask, 0] = tokens[
                        ~blank_mask, step, expansion_idx
                    ]
                else:
                    # add score for the last blank if not yet emitted within this step
                    scores[~blank_mask, step, expansion_idx] = lprobs[
                        ~blank_mask, self.blank
                    ]
                    blank_mask.fill_(True)

                self.model.decoder.masked_copy_cached_state(
                    incremental_state, old_cached_state, blank_mask
                )

                if self.lm_model is not None:
                    self.lm_model.decoder.masked_copy_cached_state(
                        lm_incremental_state, old_lm_cached_state, blank_mask
                    )

                expansion_idx += 1

        alignments = tokens if self.print_alignment else None

        return tokens.view(bsz, -1), scores.view(bsz, -1).sum(-1), alignments
