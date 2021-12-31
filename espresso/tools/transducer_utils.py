# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Hypotheses:
    """Hypotheses class for beam search algorithms. data from multiple hypotheses are
    stacked along the batch dimension of each attribute tensor.

    scores (Tensor): scores of hypotheses (including weighted LM scores if LM is present).
    sequences (Tensor): sequences of predicted non-blank tokens in hypotheses.
    sequence_lengths (Tensor): sequence lengths.
    num_emissions (Tensor): numbers of emitted tokens. It should be equal to
        (self.sequence_lengths - 1) + (number of emitted blanks). "-1" is to exclude the
        leading EOS in self.sequences which is not an emitted token.
    cached_state (Dict[str, Optional[Tensor]], optional): cached state of the ASR model.
    dec_out (Tensor, optional): decoder output of the ASR model.
    prev_tokens (Tensor, optional): the last predicted tokens (including blank).
    lm_scores (Tensor, optional): LM scores of hypotheses.
    lm_cached_state (Dict[str, Optional[Tensor]], optional): cached state of the LM
        (for LM fusion).
    lm_dec_out (Tensor, optional): decoder output of the LM (for LM fusion).
    """

    scores: Tensor  # B
    sequences: Tensor  # B x U
    sequence_lengths: Tensor  # B
    num_emissions: Tensor  # B
    cached_state: Optional[Dict[str, Optional[Tensor]]] = None  # L (num_layers) x B x *
    dec_out: Optional[Tensor] = None  # B x U x H
    prev_tokens: Optional[Tensor] = None  # B
    alignments: Optional[Tensor] = None  # B x T x N (max num tokens per time step)
    lm_scores: Optional[Tensor] = None  # B
    lm_cached_state: Optional[Dict[str, Optional[Tensor]]] = None  # L x B x *
    lm_dec_out: Optional[Tensor] = None  # B x U x H'

    def calculate_sequence_lengths(self, pad_idx: Optional[int] = 0) -> Hypotheses:
        """Returns sequence lengths in this instance calculated from `self.sequences`.

        Args:
            pad_idx (int, optional): pad id (default: 0)
        Returns:
            sequence_lengths (Tensor): sequence length tensor of shape `(batch,)`
        """
        return (self.sequences != pad_idx).long().sum(-1)

    def size(self):
        """Returns the number of hypotheses in this instance."""
        return self.scores.size(0)

    def index_select_(self, index: Tensor) -> Hypotheses:
        """Returns this instance whose attributes are index selected along the batch dimension according to the provided
        index tensor. The padding columns of new `self.sequences`, `self.dec_out` and `self.lm_dec_out` are also truncated.
        Note: this function will modify this instance.

        Args:
            index (Tensor): index tensor of shape `(batch,)`

        Returns:
            hyps (Hypotheses): this instance
        """
        if self.size() == 0:
            return self

        self.scores = self.scores.index_select(0, index)
        self.sequence_lengths = self.sequence_lengths.index_select(0, index)
        self.num_emissions = self.num_emissions.index_select(0, index)
        max_length = self.sequence_lengths.max()
        self.sequences = self.sequences[:, :max_length].index_select(0, index)
        if self.cached_state is not None:
            for k, v in self.cached_state.items():
                if v is not None:
                    self.cached_state[k] = self.cached_state[k].index_select(1, index)
        if self.dec_out is not None:
            self.dec_out = self.dec_out[:, :max_length, :].index_select(0, index)
        if self.prev_tokens is not None:
            self.prev_tokens = self.prev_tokens.index_select(0, index)
        if self.alignments is not None:
            self.alignments = self.alignments.index_select(0, index)
        if self.lm_scores is not None:
            self.lm_scores = self.lm_scores.index_select(0, index)
        if self.lm_cached_state is not None:
            for k, v in self.lm_cached_state.items():
                if v is not None:
                    self.lm_cached_state[k] = self.lm_cached_state[k].index_select(
                        1, index
                    )
        if self.lm_dec_out is not None:
            self.lm_dec_out = self.lm_dec_out[:, :max_length, :].index_select(0, index)

        return self

    def sort_by_score_(
        self,
        descending: Optional[bool] = False,
        normalize_by_length: Optional[bool] = False,
    ) -> Hypotheses:
        """Sorts the hypotheses in ascending/descending order of their scores which are
        optionally normalized by sequence length.
        Note: this function will modify this instance.

        Args:
            descending (bool, optional): whether sort in descending order of scores (default: False)
            normalize_by_length (bool, optional): if True, normalize the scores by number of
                emissions (default: False)

        Returns:
            hyps (Hypotheses): this instance
        """
        if self.size() == 0:
            return self

        scores = self.scores
        if normalize_by_length:
            scores = scores / self.num_emissions

        sort_order = scores.argsort(descending=descending)

        return self.index_select_(sort_order)

    def sort_by_length_(self, descending: Optional[bool] = False) -> Hypotheses:
        """Sorts the hypotheses in ascending/descending order of the predicted sequence lengths.
        Note: this function will modify this instance.

        Args:
            descending (bool, optional): whether sort in descending order of sequence lengths (default: False)

        Returns:
            hyps (Hypotheses): this instance
        """
        if self.size() == 0:
            return self

        sort_order = self.sequence_lengths.argsort(descending=descending)

        return self.index_select_(sort_order)

    def keep_first_k_(self, k: int) -> Hypotheses:
        """Keeps the first k hypotheses from this instance and discards the rest.
        This function is usually called after sorting the hypotheses.
        Note: this function will modify this instance.

        Args:
            k (int): the number of hypotheses to keep

        Returns:
            hyps (Hypotheses): this instance
        """
        index = torch.arange(min(k, self.size()), device=self.scores.device)
        return self.index_select_(index)

    def keep_top_k_(
        self,
        k: int,
        largest: Optional[bool] = True,
        sorted: Optional[bool] = True,
        normalize_by_length: Optional[bool] = False,
    ) -> Hypotheses:
        """Keeps the k-best hypotheses of this instance based on their scores
        and discards the rest. This function is usually faster than
        self.sort_by_score_(descending=True).keep_first_k_(k) if self.size() is
        much larger than k.
        Note: this function will modify this instance.

        Args:
            k (int): the number of best hypotheses to keep
            largest (bool, optional): controls whether to return largest or smallest elements (default: True)
            sorted (bool, optional): controls whether to return the elements in sorted order (default: True)
            normalize_by_length (bool, optional): if True, normalize the scores by number of
                emissions (default: False)

        Returns:
            hyps (Hypotheses): this instance
        """
        if k > self.size():
            return (
                self.sort_by_score_(
                    descending=largest, normalize_by_length=normalize_by_length
                )
                if sorted
                else self
            )

        scores = self.scores
        if normalize_by_length:
            scores = scores / self.num_emissions

        _, index = torch.topk(scores, k, largest=largest, sorted=sorted)

        return self.index_select_(index)

    def masked_select(self, mask: Tensor) -> Hypotheses:
        """Returns a new instance of :class:`~Hypotheses` where all its attributes are selected from this instance and
        along the batch dimension, specified as the boolean mask which is a :class:`~BoolTensor`. The padding columns of
        new `sequences`, `dec_out` and `lm_dec_out` are also truncated.
        Note: this function will NOT modify this instance.

        Args:
            mask (Tensor): mask bool tensor of shape `(batch,)`

        Returns:
            hyps (Hypotheses): selected hypotheses
        """
        assert self.size() == mask.size(0)

        if self.size() == 0 or mask.all():
            return self

        scores = self.scores[mask]
        sequence_lengths = self.sequence_lengths[mask]
        num_emissions = self.num_emissions[mask]
        max_length = sequence_lengths.max() if sequence_lengths.size(0) > 0 else None
        sequences = self.sequences[:, :max_length][mask, :]
        if self.cached_state is not None:
            cached_state = {}
            for k, v in self.cached_state.items():
                if v is not None:
                    cached_state[k] = self.cached_state[k][:, mask, ...]
                else:
                    cached_state[k] = None
        else:
            cached_state = None
        if self.dec_out is not None:
            dec_out = self.dec_out[:, :max_length, :][mask, :, :]
        else:
            dec_out = None
        if self.prev_tokens is not None:
            prev_tokens = self.prev_tokens[mask]
        else:
            prev_tokens = None
        if self.alignments is not None:
            alignments = self.alignments[mask, :, :]
        else:
            alignments = None
        if self.lm_scores is not None:
            lm_scores = self.lm_scores[mask]
        else:
            lm_scores = None
        if self.lm_cached_state is not None:
            lm_cached_state = {}
            for k, v in self.lm_cached_state.items():
                if v is not None:
                    lm_cached_state[k] = self.lm_cached_state[k][:, mask, ...]
                else:
                    lm_cached_state[k] = None
        else:
            lm_cached_state = None
        if self.lm_dec_out is not None:
            lm_dec_out = self.lm_dec_out[:, :max_length, :][mask, :, :]
        else:
            lm_dec_out = None

        return Hypotheses(
            scores=scores,
            sequences=sequences,
            sequence_lengths=sequence_lengths,
            num_emissions=num_emissions,
            cached_state=cached_state,
            dec_out=dec_out,
            alignments=alignments,
            prev_tokens=prev_tokens,
            lm_scores=lm_scores,
            lm_cached_state=lm_cached_state,
            lm_dec_out=lm_dec_out,
        )

    def append_tokens_(
        self,
        tokens: Tensor,
        time_step: int,
        expansion_idx: int,
        blank_idx: int,
        pad_idx: Optional[int] = 0,
    ) -> Hypotheses:
        """Appends non-blank tokens in `tokens` to `self.sequences`, allocates additional memory for
        `self.dec_out` and `self.lm_dec_out` if needed (which will later be updated by :func:`~Hypotheses.update_dec_out_()`),
        updates `self.prev_tokens` with `tokens`, and increment `self.sequence_lengths` and `self.num_emissions`.
        Note: this function will modify this instance.

        Args:
            tokens (Tensor): token id tensor of shape `(batch,)`
            time_step (int): the current time step of the encoder output at which `tokens` are being emitted
            expansion_idx (int): the current expansion index within the current time step
            blank_idx (int): blank id
            pad_idx (int, optional): pad id (default: 0)

        Returns:
            hyps (Hypotheses): this instance
        """
        bsz = self.size()
        assert bsz > 0
        assert self.sequences.size(0) == bsz
        assert tokens.size(0) == bsz

        max_length = self.sequence_lengths.max()
        assert max_length <= self.sequences.size(1)
        if self.dec_out is not None:
            assert self.sequences.size(1) == self.dec_out.size(
                1
            ), f"{self.sequences.size(1)} != {self.dec_out.size(1)}"
        if self.lm_dec_out is not None:
            assert self.sequences.size(1) == self.lm_dec_out.size(
                1
            ), f"{self.sequences.size(1)} != {self.lm_dec_out.size(1)}"

        if self.prev_tokens is not None:
            self.prev_tokens = tokens.clone()

        if self.alignments is not None:
            assert self.alignments.size(0) == bsz
            self.alignments[:, time_step, expansion_idx] = tokens

        blank_mask = tokens == blank_idx
        if blank_mask.all():  # a shortcut
            return self

        if (tokens[self.sequence_lengths == max_length] != blank_idx).any():
            self.sequences = F.pad(self.sequences, (0, 1), value=pad_idx)
            if self.dec_out is not None:
                self.dec_out = F.pad(self.dec_out, (0, 0, 0, 1))
            if self.lm_dec_out is not None:
                self.lm_dec_out = F.pad(self.lm_dec_out, (0, 0, 0, 1))

        tokens_with_pad = tokens.masked_fill(blank_mask, pad_idx)
        self.sequences.scatter_(
            1, self.sequence_lengths.unsqueeze(1), tokens_with_pad.unsqueeze(1)
        )

        self.sequence_lengths += (~blank_mask).long()
        self.num_emissions += 1

        return self

    def update_dec_out_(
        self, dec_out: Tensor, lm_dec_out: Optional[Tensor] = None
    ) -> Hypotheses:
        """Updates `dec_out` to `self.dec_out` and optionally `lm_dec_out` to `self.lm_dec_out`,
        along dimension 1. Memory is assumed to have been allocated in :func:`~Hypotheses.append_tokens_()`.
        Note: this function will modify this instance.

        Args:
            dec_out (Tensor): new dec_out tensor of shape `(batch, 1, hidden_dim)`
            lm_dec_out (Tensor, optional): new lm_dec_out tensor of shape `(batch, 1, hidden_dim)`

        Returns:
            hyps (Hypotheses): this instance
        """
        assert self.size() > 0
        assert self.dec_out is not None
        max_length = self.sequence_lengths.max()
        assert max_length <= self.dec_out.size(1)
        self.dec_out.scatter_(
            1,
            (self.sequence_lengths - 1)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, -1, dec_out.size(2)),
            dec_out,
        )

        if lm_dec_out is not None:
            assert self.lm_dec_out is not None
            assert max_length <= self.lm_dec_out.size(1)
            self.lm_dec_out.scatter_(
                1,
                (self.sequence_lengths - 1)
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, -1, lm_dec_out.size(2)),
                lm_dec_out,
            )

        return self

    def get_last_dec_out(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Returns the last `dec_out`/`lm_dec_out`, which is the output after feeding the
        last non-blank tokens in `self.sequences` into the decoder/LM.
        Note: this function will NOT modify this instance.

        Returns:
            last_dec_out (Tensor, optional): the last decoder out, a tensor of shape `(bsz, 1, hidden_dim)`
            last_lm_dec_out (Tensor, optional): the last LM decoder out, a tensor of shape `(bsz, 1, lm_hidden_dim)`
        """
        assert self.size() > 0

        last_dec_out = None
        if self.dec_out is not None:
            last_dec_out = self.dec_out.gather(
                1,
                (self.sequence_lengths - 1)
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, -1, self.dec_out.size(2)),
            )  # B x 1 x H

        last_lm_dec_out = None
        if self.lm_dec_out is not None:
            last_lm_dec_out = self.lm_dec_out.gather(
                1,
                (self.sequence_lengths - 1)
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, -1, self.lm_dec_out.size(2)),
            )  # B x 1 x H'

        return last_dec_out, last_lm_dec_out

    def repeat_interleave(self, repeats: int) -> Hypotheses:
        """Interleaved repeats the hypotheses `repeats` times along the batch dimension.
        Effectively similar to :func:`~torch.Tensor.repeat_interleave(repeats, dim=<batch-dim>)`.
        Note: this function will NOT modify this instance.

        Args:
            repeats (int): the number of repetitions for each element

        Returns:
            repeated_hyps (Hypotheses): repeated hypotheses
        """
        assert self.size() > 0
        scores = self.scores.repeat_interleave(repeats, dim=0)
        sequences = self.sequences.repeat_interleave(repeats, dim=0)
        sequence_lengths = self.sequence_lengths.repeat_interleave(repeats, dim=0)
        num_emissions = self.num_emissions.repeat_interleave(repeats, dim=0)
        if self.cached_state is not None:
            cached_state = {}
            for k, v in self.cached_state.items():
                if v is not None:
                    cached_state[k] = self.cached_state[k].repeat_interleave(
                        repeats, dim=1
                    )
                else:
                    cached_state[k] = None
        else:
            cached_state = None
        if self.dec_out is not None:
            dec_out = self.dec_out.repeat_interleave(repeats, dim=0)
        else:
            dec_out = None
        if self.prev_tokens is not None:
            prev_tokens = self.prev_tokens.repeat_interleave(repeats, dim=0)
        else:
            prev_tokens = None
        if self.alignments is not None:
            alignments = self.alignments.repeat_interleave(repeats, dim=0)
        else:
            alignments = None
        if self.lm_scores is not None:
            lm_scores = self.lm_scores.repeat_interleave(repeats, dim=0)
        else:
            lm_scores = None
        if self.lm_cached_state is not None:
            lm_cached_state = {}
            for k, v in self.lm_cached_state.items():
                if v is not None:
                    lm_cached_state[k] = self.lm_cached_state[k].repeat_interleave(
                        repeats, dim=1
                    )
                else:
                    lm_cached_state[k] = None
        else:
            lm_cached_state = None
        if self.lm_dec_out is not None:
            lm_dec_out = self.lm_dec_out.repeat_interleave(repeats, dim=0)
        else:
            lm_dec_out = None

        return Hypotheses(
            scores=scores,
            sequences=sequences,
            sequence_lengths=sequence_lengths,
            num_emissions=num_emissions,
            cached_state=cached_state,
            dec_out=dec_out,
            prev_tokens=prev_tokens,
            alignments=alignments,
            lm_scores=lm_scores,
            lm_cached_state=lm_cached_state,
            lm_dec_out=lm_dec_out,
        )

    def combine(
        self, another_hyps: Hypotheses, pad_idx: Optional[int] = 0
    ) -> Hypotheses:
        """Returns a new instance of :class:`~Hypotheses` where it combines hypotheses from this instance and `another_hyps`.
        It does tensor concatenations along the batch dimension, after padding along the time dimension if needed.
        Note: this function will NOT modify this instance.

        Args:
            another_hyps (Hypotheses): another hypotheses instance
            pad_idx (int, optional): pad id

        Returns:
            hyps (Hypotheses): combine hypotheses
        """
        if another_hyps.size() == 0:
            return self
        if self.size() == 0:
            return another_hyps

        scores = torch.cat((self.scores, another_hyps.scores), dim=0)
        # pad the shorter `sequences` to match the longer one
        max_len = self.sequences.size(1)
        another_max_len = another_hyps.sequences.size(1)
        new_max_len = max(max_len, another_max_len)
        padded_sequences = (
            F.pad(self.sequences, (0, new_max_len - max_len), value=pad_idx)
            if new_max_len - max_len > 0
            else self.sequences
        )
        another_padded_sequences = (
            F.pad(
                another_hyps.sequences,
                (0, new_max_len - another_max_len),
                value=pad_idx,
            )
            if new_max_len - another_max_len > 0
            else another_hyps.sequences
        )
        sequences = torch.cat((padded_sequences, another_padded_sequences), dim=0)

        sequence_lengths = torch.cat(
            (self.sequence_lengths, another_hyps.sequence_lengths), dim=0
        )
        num_emissions = torch.cat(
            (self.num_emissions, another_hyps.num_emissions), dim=0
        )

        if self.cached_state is not None:
            cached_state = {}
            for k, v in self.cached_state.items():
                if v is not None:
                    assert another_hyps.cached_state[k] is not None
                    cached_state[k] = torch.cat(
                        (self.cached_state[k], another_hyps.cached_state[k]), dim=1
                    )
                else:
                    cached_state[k] = None
        else:
            assert another_hyps.cached_state is None
            cached_state = None
        if self.dec_out is not None:
            # pad the shorter `dec_out` to match the longer one
            assert another_hyps.dec_out is not None
            assert max_len == self.dec_out.size(1)
            assert another_max_len == another_hyps.dec_out.size(1)
            padded_dec_out = (
                F.pad(self.dec_out, (0, 0, 0, new_max_len - max_len))
                if new_max_len - max_len > 0
                else self.dec_out
            )
            another_padded_dec_out = (
                F.pad(another_hyps.dec_out, (0, 0, 0, new_max_len - another_max_len))
                if new_max_len - another_max_len > 0
                else another_hyps.dec_out
            )
            dec_out = torch.cat((padded_dec_out, another_padded_dec_out), dim=0)
        else:
            assert another_hyps.dec_out is None
            dec_out = None
        if self.prev_tokens is not None:
            assert another_hyps.prev_tokens is not None
            prev_tokens = torch.cat((self.prev_tokens, another_hyps.prev_tokens), dim=0)
        else:
            assert another_hyps.prev_tokens is None
            prev_tokens = None
        if self.alignments is not None:
            assert another_hyps.alignments is not None
            alignments = torch.cat((self.alignments, another_hyps.alignments), dim=0)
        else:
            assert another_hyps.alignments is None
            alignments = None
        if self.lm_scores is not None:
            assert another_hyps.lm_scores is not None
            lm_scores = torch.cat((self.lm_scores, another_hyps.lm_scores), dim=0)
        else:
            assert another_hyps.lm_scores is None
            lm_scores = None
        if self.lm_cached_state is not None:
            lm_cached_state = {}
            for k, v in self.lm_cached_state.items():
                if v is not None:
                    assert another_hyps.lm_cached_state[k] is not None
                    lm_cached_state[k] = torch.cat(
                        (self.lm_cached_state[k], another_hyps.lm_cached_state[k]),
                        dim=1,
                    )
                else:
                    lm_cached_state[k] = None
        else:
            assert another_hyps.lm_cached_state is None
            lm_cached_state = None
        if self.lm_dec_out is not None:
            # pad the shorter `lm_dec_out` to match the longer one
            assert another_hyps.lm_dec_out is not None
            assert max_len == self.lm_dec_out.size(1)
            assert another_max_len == another_hyps.lm_dec_out.size(1)
            padded_lm_dec_out = (
                F.pad(self.lm_dec_out, (0, 0, 0, new_max_len - max_len))
                if new_max_len - max_len > 0
                else self.lm_dec_out
            )
            another_padded_lm_dec_out = (
                F.pad(another_hyps.lm_dec_out, (0, 0, 0, new_max_len - another_max_len))
                if new_max_len - another_max_len > 0
                else another_hyps.lm_dec_out
            )

            lm_dec_out = torch.cat(
                (padded_lm_dec_out, another_padded_lm_dec_out), dim=0
            )
        else:
            assert another_hyps.lm_dec_out is None
            lm_dec_out = None

        return Hypotheses(
            scores=scores,
            sequences=sequences,
            sequence_lengths=sequence_lengths,
            num_emissions=num_emissions,
            cached_state=cached_state,
            dec_out=dec_out,
            prev_tokens=prev_tokens,
            alignments=alignments,
            lm_scores=lm_scores,
            lm_cached_state=lm_cached_state,
            lm_dec_out=lm_dec_out,
        )


def select_k_expansions(
    hyps: Hypotheses,
    lprobs: Tensor,
    beam_size: int,
    time_step: int,
    expansion_idx: int,
    blank_idx: int,
    pad_idx: Optional[int] = 0,
    lm_lprobs_padded: Optional[Tensor] = None,
    gamma: Optional[float] = None,
    beta: Optional[int] = 0,
    normalize_by_length: Optional[bool] = False,
) -> Hypotheses:
    """Returns K hypotheses candidates for expansions from a set of hypotheses.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.
    Note: This function should be followed with :func:`~Hypotheses.update_dec_out_()` in the calling code
    to also update `k_expanded_hyps_nonblank.cached_state`, `k_expanded_hyps.dec_out`,
    `k_expanded_hyps_nonblank.lm_cached_state`, and `k_expanded_hyps.lm_dec_out` after non-blank expansions.

    This implementation is modified from
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transducer/utils.py
    and is adapted to tensorized operations.

    Args:
        hyps (Hypotheses): Hypotheses.
        lprobs (Tensor): Log-probabilities for hypotheses expansions (including weighted LM's lprobs if LM is present),
            a tensor of shape `(batch, vocab_size)`.
        beam_size (int): Beam size.
        time_step (int): the current time step of the encoder output at which `tokens` are being emitted.
        expansion_idx (int): the current expansion index within the current time step.
        blank_idx (int, optional): blank id.
        pad_idx (int, optional): pad id (default: 0).
        lm_lprobs_padded (Tensor, optional): Log-probabilities from the LM only, a tensor of shape `(batch, vocab_size)`.
            Note its vocabulary dimension need to be padded properly to match that of `lprobs` by inserting zeros at the
            position of blank index.
        gamma (float, optional): Allowed logp difference for prune-by-value method (default: None).
        beta (int, optional): Number of additional candidates to store (default: 0).
        normalize_by_length (bool, optional): if True, normalize the scores by number of
            emissions (default: False)

    Returns:
       k_expanded_hyps (Hypotheses): Best K expansion hypotheses candidates.
    """
    assert hyps.size() > 0
    lprobs = lprobs + hyps.scores.unsqueeze(-1)  # B x V
    K = min(beam_size + beta, lprobs.size(1) - 1)  # -1 so we never select pad
    scores, indices = torch.topk(lprobs, k=K)  # B x K
    k_expanded_hyps = hyps.repeat_interleave(K)  # (B * K) hypotheses
    k_expanded_hyps.scores = scores.view(-1)  # (B * K)
    if lm_lprobs_padded is not None:
        assert lm_lprobs_padded.size() == lprobs.size()
        assert k_expanded_hyps.lm_scores is not None
        k_expanded_hyps.lm_scores += lm_lprobs_padded.gather(1, indices).view(
            -1
        )  # (B * K)

    k_expanded_hyps.append_tokens_(
        indices.view(-1), time_step, expansion_idx, blank_idx, pad_idx=pad_idx
    )

    if gamma is not None:
        retained_mask = scores >= (scores[:, :1] - gamma)  # B x K
        if not retained_mask.all():  # prune by value
            k_expanded_hyps = k_expanded_hyps.masked_select(retained_mask.view(-1))

    k_expanded_hyps.keep_top_k_(K, normalize_by_length=normalize_by_length)

    return k_expanded_hyps


def is_prefix(a: List[Union[int, str]], b: List[Union[int, str]]):
    """Check if `a` is a prefix of `b`."""
    if len(a) >= len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def is_prefix_tensorized(hyps: Hypotheses, are_sorted: Optional[bool] = False):
    """Returns a mask tensor where the (i, j)-th element indicates if the i-th row of `hyps.sequences`
    is a prefix of the j-th row.

    Args:
        hyps (Hypotheses): sequences of tokens, a tensor of shape `(batch, tgt_len)`.
        are_sorted (bool, optional): True if the hypotheses in `hyps` are already sorted by length (in non-increasing order).

    Returns:
        prefix_relations (Tensor): `prefix_relations[i][j]` is True iff `hyps.sequences[i]` is a prefix of `hyps.sequences[j]`,
            a bool tensor of shape `(batch, batch)`.
    """
    bsz = hyps.size()
    assert bsz > 0
    assert hyps.sequences.size(0) == bsz
    lengths = hyps.sequence_lengths

    prefix_relations = hyps.sequences.new_full((bsz, bsz), False, dtype=torch.bool)

    def check_pair(i, j):
        return (
            lengths[i] < lengths[j]
            and (hyps.sequences[i, :] == hyps.sequences[j, :])[: lengths[i]].all()
        )

    if are_sorted:
        for j in range(bsz - 1):
            for i in range(j + 1, bsz):
                prefix_relations[i, j] = check_pair(i, j)
    else:
        for j in range(bsz):
            for i in range(bsz):
                prefix_relations[i, j] = check_pair(i, j)

    return prefix_relations
