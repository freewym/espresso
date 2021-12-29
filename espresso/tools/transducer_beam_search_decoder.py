# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torch import Tensor

from espresso.tools.transducer_base_decoder import TransducerBaseDecoder
from espresso.tools.transducer_utils import (
    Hypotheses,
    is_prefix_tensorized,
    select_k_expansions,
)
from fairseq.data.data_utils import collate_tokens
from fairseq.utils import strip_pad


class TransducerBeamSearchDecoder(TransducerBaseDecoder):
    def __init__(
        self,
        models,
        dictionary,
        beam_size=1,
        max_len=0,
        max_num_expansions_per_step=2,
        expansion_beta=0,
        expansion_gamma=None,
        prefix_alpha=None,
        normalize_scores=True,
        temperature=1.0,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        print_alignment=False,
        **kwargs,
    ):
        """Decode given speech audios with a beam search algorithm "Adaptive Expansion Search"
        introduced in https://ieeexplore.ieee.org/document/9250505. This implementation is modified
        from https://github.com/espnet/espnet/blob/master/espnet/nets/beam_search_transducer.py.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            dictionary (~fairseq.data.Dictionary): dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence) (default: 0, no limit)
            max_num_expansions_per_step (int, optional): the maximum number of
                non-blank expansions in a single time step (default: 2)
            expansion_beta (int, optional): maximum number of prefix expansions allowed,
                in addition to the beam size. Effectively, the number of hypotheses =
                beam_size + expansion_beta (default: 0)
            expansion_gamma (float, optional): pruning threshold used in the prune-by-value
                step when computing the expansions. It performs a comparison
                (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices and
                max_log_prob is the "most" likely token to be predicted. Gamma therefore provides
                a margin of additional tokens which can be potential candidates for expansion apart
                from the "most likely" candidate. Lower values will reduce the number of expansions.
                Empirically tuned for each dataset (default: None)
            prefix_alpha (int, optional): maximum prefix length in prefix search.
                Must be an integer, and is advised to keep this as 1 in order to
                reduce expensive beam search cost later (default: 1)
            normalize_scores (bool, optional): normalize scores by the length
                of the output including blank (default: True)
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
        self.pad = dictionary.pad()
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.expansion_beta = expansion_beta
        assert expansion_beta >= 0, "--expansion-beta must be non-negative"
        self.expansion_gamma = expansion_gamma
        assert (
            expansion_gamma is None or expansion_gamma > 0.0
        ), "--expansion-gamma must be greater than 0.0"
        self.prefix_alpha = prefix_alpha
        assert (
            prefix_alpha is None or prefix_alpha > 0
        ), "--prefix-alpha must be None or at least 1"
        self.normalize_scores = normalize_scores

    @torch.no_grad()
    def decode(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        assert not self.print_alignment

        tokens_list, scores_list, _ = self._generate(sample, **kwargs)
        bsz = len(tokens_list)
        tokens = collate_tokens(
            [tokens_list[i][j, :] for i in range(bsz) for j in range(self.beam_size)],
            pad_idx=self.pad,
        ).view(
            bsz, self.beam_size, -1
        )  # B x beam_size x U
        scores = torch.stack(scores_list, dim=0)  # B x beam_size

        return tokens, scores, None

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """API to be invoked from :func:`~fairseq.tasks.fairseq_task.FairseqTask.inference_step()`"""
        bos_token = kwargs.get("bos_token", None)
        tokens_list, scores_list, alignments_list = self._generate(
            sample, bos_token=bos_token
        )
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
                        "alignment": alignments_list[i][j, :, :]
                        if self.print_alignment
                        else None,
                    }
                )

        return finalized

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
        sequences_list, scores_list, alignments_list = [], [], []
        for i in range(bsz):
            sequences, scores, alignments = self._generate_one_example(
                enc_out[i : i + 1, :, :], enc_out_lengths[i], bos_token=bos_token
            )
            sequences_list.append(sequences)
            scores_list.append(scores)
            alignments_list.append(alignments)

        return sequences_list, scores_list, alignments_list

    @torch.no_grad()
    def _generate_one_example(
        self, enc_out: Tensor, enc_out_length: int, bos_token: Optional[int] = None
    ):
        max_len = (
            min(enc_out_length, self.max_len) if self.max_len > 0 else enc_out_length
        )

        # prev_tokens stores the previous tokens to be fed into the decoder
        prev_tokens = enc_out.new_full(
            (1,), self.eos if bos_token is None else bos_token, dtype=torch.long
        )  # B(=1)

        if self.print_alignment:
            # token alignments for the final best hypthesis. +1 for blank
            alignments = enc_out.new_full(
                (enc_out_length, self.max_num_expansions_per_step + 1),
                self.pad,
                dtype=torch.long,
            )  # T x N (max num tokens per time step)
        else:
            alignments = None

        # scores is used to store log-prob of emitting each token
        scores = enc_out.new_full((1,), 0.0)  # B(=1)
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        dec_out = self.model.decoder.extract_features(
            prev_tokens.unsqueeze(1), incremental_state=incremental_state
        )[
            0
        ]  # B(=1) x 1 x H
        cached_state = self.model.decoder.get_incremental_state(
            incremental_state, "cached_state"
        )  # each tensor: L x B(=1) x *

        if self.lm_model is not None:
            lm_scores = enc_out.new_full((1,), 0.0)  # B(=1)
            lm_incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            lm_prev_tokens = (
                torch.where(prev_tokens > self.blank, prev_tokens - 1, prev_tokens)
                if self.no_blank_in_lm
                else prev_tokens
            )
            lm_dec_out = self.lm_model.decoder.extract_features(
                lm_prev_tokens.unsqueeze(1), incremental_state=lm_incremental_state
            )[
                0
            ]  # B(=1) x 1 x H'
            lm_cached_state = self.lm_model.decoder.get_incremental_state(
                lm_incremental_state, "cached_state"
            )  # each tensor: L x B(=1) x *
        else:
            lm_scores = lm_incremental_state = lm_dec_out = lm_cached_state = None

        next_step_hyps = Hypotheses(
            scores=scores,  # B(=1)
            sequences=prev_tokens.unsqueeze(1).clone(),  # B(=1) x U(=1)
            sequence_lengths=prev_tokens.new_ones((1,)),  # B(=1)
            num_emissions=prev_tokens.new_zeros((1,)),  # B(=1)
            cached_state=cached_state,  # each tensor: L x B(=1) x *
            dec_out=dec_out,  # B(=1) x U(=1) x H
            prev_tokens=prev_tokens,  # B(=1)
            alignments=alignments,  # B(=1) x T x N
            lm_scores=lm_scores,  # B(=1)
            lm_cached_state=lm_cached_state,  # each tensor: L x B(=1) x *
            lm_dec_out=lm_dec_out,  # B(=1) x U(=1) x H'
        )

        for step in range(max_len):
            next_step_hyps.sort_by_length_(descending=True)
            enc_out_this_step = enc_out[:, step : step + 1, :]  # B(=1) x 1 x C
            hyps = self.prefix_search_and_merge(
                next_step_hyps, enc_out_this_step, alpha=self.prefix_alpha
            )

            k_expanded_hyps_blank = None
            for expansion_idx in range(self.max_num_expansions_per_step):
                (
                    last_dec_out,
                    last_lm_dec_out,
                ) = hyps.get_last_dec_out()  # B x 1 x H, B x 1 x H'
                logits = (
                    self.model.joint(
                        enc_out_this_step.expand(
                            last_dec_out.size(0), -1, -1
                        ),  # B x 1 x C
                        last_dec_out,
                        apply_output_layer=True,
                    )
                    .squeeze(2)
                    .squeeze(1)
                )  # B x 1 x 1 x V -> B x V
                lprobs = self.model.get_normalized_probs(
                    (logits.div_(self.temperature), None), log_probs=True
                )  # B x V

                if self.lm_model is not None:
                    lm_logits = self.lm_model.output_layer(last_lm_dec_out).squeeze(
                        1
                    )  # B x 1 x V' -> B x V'
                    lm_lprobs = self.lm_model.get_normalized_probs(
                        (lm_logits, None), log_probs=True
                    )  # B x V'

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
                    lm_lprobs_padded = torch.cat(
                        (
                            lm_lprobs[:, : self.blank],
                            lm_lprobs.new_zeros((lm_lprobs.size(0), 1)),
                            lm_lprobs[:, self.blank :],
                        ),
                        dim=1,
                    )  # B x V
                else:
                    lm_lprobs_padded = None

                # compute k expansions for all the current hypotheses
                k_expanded_hyps = select_k_expansions(
                    hyps,
                    lprobs,
                    self.beam_size,
                    step,
                    expansion_idx,
                    self.blank,
                    pad_idx=self.pad,
                    lm_lprobs_padded=lm_lprobs_padded,
                    gamma=self.expansion_gamma,
                    beta=self.expansion_beta,
                    normalize_by_length=self.normalize_scores,
                )
                blank_mask = k_expanded_hyps.prev_tokens == self.blank
                if expansion_idx == 0:
                    k_expanded_hyps_blank = k_expanded_hyps.masked_select(blank_mask)
                else:
                    k_expanded_hyps_blank = k_expanded_hyps_blank.combine(
                        k_expanded_hyps.masked_select(blank_mask), pad_idx=self.pad
                    )
                k_expanded_hyps_nonblank = k_expanded_hyps.masked_select(~blank_mask)

                if k_expanded_hyps_nonblank.size() == 0:  # all expanded with blank
                    # early exit of expansions
                    next_step_hyps = k_expanded_hyps_blank.keep_top_k_(
                        self.beam_size, normalize_by_length=self.normalize_scores
                    )
                    break
                else:
                    # forward the decoder with `k_expanded_hyps_nonblank`
                    self.model.decoder.set_incremental_state(
                        incremental_state,
                        "cached_state",
                        k_expanded_hyps_nonblank.cached_state,
                    )
                    dec_out = self.model.decoder.extract_features(
                        k_expanded_hyps_nonblank.prev_tokens.unsqueeze(1),
                        incremental_state=incremental_state,
                    )[
                        0
                    ]  # B x 1 x H
                    k_expanded_hyps_nonblank.cached_state = (
                        self.model.decoder.get_incremental_state(
                            incremental_state, "cached_state"
                        )
                    )  # update `cached_state`

                    if self.lm_model is not None:
                        self.lm_model.decoder.set_incremental_state(
                            lm_incremental_state,
                            "cached_state",
                            k_expanded_hyps_nonblank.lm_cached_state,
                        )
                        prev_tokens = k_expanded_hyps_nonblank.prev_tokens
                        lm_prev_tokens = (
                            torch.where(
                                prev_tokens > self.blank, prev_tokens - 1, prev_tokens
                            )
                            if self.no_blank_in_lm
                            else prev_tokens
                        )
                        lm_dec_out = self.lm_model.extract_features(
                            lm_prev_tokens.unsqueeze(1),
                            incremental_state=lm_incremental_state,
                        )[
                            0
                        ]  # B x 1 x H'
                        k_expanded_hyps_nonblank.lm_cached_state = (
                            self.lm_model.decoder.get_incremental_state(
                                lm_incremental_state, "cached_state"
                            )
                        )  # update `lm_cached_state`
                    else:
                        lm_dec_out = None

                    k_expanded_hyps_nonblank.update_dec_out_(
                        dec_out, lm_dec_out=lm_dec_out
                    )  # update `dec_out` and `lm_dec_out`

                    if (
                        expansion_idx < self.max_num_expansions_per_step - 1
                    ):  # not the last round of expansion within this time step
                        # prepare for the next round of expansion within this time step
                        hyps = k_expanded_hyps_nonblank
                    else:  # the last round of expansion within this time step
                        # add blank probability to non-blank hyps, combine and prune the hyps for the next time step
                        logits = (
                            self.model.joint(
                                enc_out_this_step.expand(dec_out.size(0), -1, -1),
                                dec_out,
                                apply_output_layer=True,
                            )
                            .squeeze(2)
                            .squeeze(1)
                        )  # B x 1 x 1 x V -> B x V
                        lprobs = self.model.get_normalized_probs(
                            (logits.div_(self.temperature), None), log_probs=True
                        )  # B x V

                        k_expanded_hyps_nonblank.scores += lprobs[:, self.blank]
                        k_expanded_hyps_nonblank.prev_tokens.fill_(
                            self.blank
                        )  # unnecessary but conceptually should do
                        k_expanded_hyps_nonblank.num_emissions += 1

                        next_step_hyps = k_expanded_hyps_blank.combine(
                            k_expanded_hyps_nonblank, pad_idx=self.pad
                        )
                        next_step_hyps.keep_top_k_(
                            self.beam_size, normalize_by_length=self.normalize_scores
                        )

        next_step_hyps.sort_by_score_(
            descending=True, normalize_by_length=self.normalize_scores
        )
        # get the N-best hypotheses, and exclude the leading EOS token from the sequences
        sequences = next_step_hyps.sequences[:, 1:]  # B x U
        scores = next_step_hyps.scores / (next_step_hyps.sequence_lengths - 1)  # B
        if self.print_alignment:
            alignments = next_step_hyps.alignments  # B x T x N
        else:
            alignments = None

        return sequences, scores, alignments

    def prefix_search_and_merge(
        self, hyps: Hypotheses, enc_out: Tensor, alpha: Optional[int] = None
    ) -> Hypotheses:
        """Prefix search and merge scores of hypothese whose sequence is a prefix of a longer hypothesis.
        It assumes that the hypotheses have been sorted in non-increasing order of sequence length.
        This implementation is modified from
        https://github.com/espnet/espnet/blob/master/espnet/nets/beam_search_transducer.py.
        Note: this function will modify hyps.

        Args:
            hyps (Hypotheses): hypotheses to be merged
            enc_out (Tensor): encoder output of shape `(batch, src_len, embed_dim)` where batch=1 and src_len=1
            alpha (int, optional): maximum prefix length in prefix search. Must be an integer, and is advised to
                keep this as 1 in order to reduce expensive beam search cost later (default: None)

        Returns:
            hyps (Hypotheses): merged hypotheses
        """
        bsz = hyps.size()
        lengths = hyps.sequence_lengths

        to_merge = is_prefix_tensorized(hyps, are_sorted=True)  # B x B
        if alpha is not None:
            to_merge = to_merge & (lengths.unsqueeze(1) + alpha >= lengths.unsqueeze(0))

        if not to_merge.any():  # no merges
            return hyps

        for j in range(bsz - 1):
            for i in range(j + 1, bsz):
                len_j = lengths[j]
                len_i = lengths[i]

                if to_merge[i, j]:
                    logits = (
                        self.model.joint(
                            enc_out,
                            hyps.dec_out[i : i + 1, len_i - 1 : len_i, :],
                            apply_output_layer=True,
                        )
                        .squeeze(2)
                        .squeeze(1)
                    )  # 1 x 1 x 1 x V -> 1 x V
                    lprobs = self.model.get_normalized_probs(
                        (logits.div_(self.temperature), None), log_probs=True
                    ).squeeze(
                        0
                    )  # 1 x V -> V
                    token_index = hyps.sequences[j][len_i]
                    score = hyps.scores[i] + lprobs[token_index]

                    if self.lm_model is not None:
                        lm_logits = self.lm_model.output_layer(
                            hyps.lm_dec_out[i : i + 1, len_i - 1 : len_i, :]
                        ).squeeze(
                            1
                        )  # 1 x 1 x V' -> 1 x V'
                        lm_lprobs = self.lm_model.get_normalized_probs(
                            (lm_logits, None), log_probs=True
                        ).squeeze(
                            0
                        )  # 1 x V' -> V'
                        if self.no_blank_in_lm and token_index > self.blank:
                            lm_token_index = token_index - 1
                        else:
                            lm_token_index = token_index
                        local_lm_score = lm_lprobs[lm_token_index]
                        lm_score = hyps.lm_scores[i] + local_lm_score
                        score += self.lm_weight * local_lm_score

                        lprobs_no_blank = lprobs[self.vocab_nonblank_mask]  # V - 1
                        if not self.no_blank_in_lm:
                            lm_lprobs = lm_lprobs[self.vocab_nonblank_mask]  # V - 1
                        lprobs_with_lm_no_blank = (
                            lprobs_no_blank + self.lm_weight * lm_lprobs
                        )  # V - 1
                        log_scaling_factor = (
                            lprobs_no_blank.exp().sum().log()
                            - lprobs_with_lm_no_blank.exp().sum().log()
                        )
                        score += log_scaling_factor

                    for k in range(len_i, len_j - 1):
                        logits = (
                            self.model.joint(
                                enc_out,
                                hyps.dec_out[j : j + 1, k : k + 1, :],
                                apply_output_layer=True,
                            )
                            .squeeze(2)
                            .squeeze(1)
                        )  # 1 x 1 x 1 x V -> 1 x V
                        lprobs = self.model.get_normalized_probs(
                            (logits.div_(self.temperature), None), log_probs=True
                        ).squeeze(
                            0
                        )  # 1 x V -> V
                        token_index = hyps.sequences[j][k + 1]
                        score += lprobs[token_index]

                        if self.lm_model is not None:
                            lm_logits = self.lm_model.output_layer(
                                hyps.lm_dec_out[j : j + 1, k : k + 1, :]
                            ).squeeze(
                                1
                            )  # 1 x 1 x V' -> 1 x V'
                            lm_lprobs = self.lm_model.get_normalized_probs(
                                (lm_logits, None), log_probs=True
                            ).squeeze(
                                0
                            )  # 1 x V' -> V'
                            if self.no_blank_in_lm and token_index > self.blank:
                                lm_token_index = token_index - 1
                            else:
                                lm_token_index = token_index
                            local_lm_score = lm_lprobs[lm_token_index]
                            lm_score += local_lm_score
                            score += self.lm_weight * local_lm_score

                            lprobs_no_blank = lprobs[self.vocab_nonblank_mask]  # V - 1
                            if not self.no_blank_in_lm:
                                lm_lprobs = lm_lprobs[self.vocab_nonblank_mask]  # V - 1
                            lprobs_with_lm_no_blank = (
                                lprobs_no_blank + self.lm_weight * lm_lprobs
                            )  # V - 1
                            log_scaling_factor = (
                                lprobs_no_blank.exp().sum().log()
                                - lprobs_with_lm_no_blank.exp().sum().log()
                            )
                            score += log_scaling_factor

                    hyps.scores[j] = torch.logaddexp(hyps.scores[j], score)

                    if self.lm_model is not None:
                        hyps.lm_scores[j] = torch.logaddexp(hyps.lm_scores[j], lm_score)

        return hyps
