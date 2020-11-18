# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq.models import FairseqIncrementalDecoder, FairseqLanguageModel

from espresso.data import AsrDictionary
from espresso.tools.lexical_prefix_tree import lexical_prefix_tree
from espresso.tools.utils import tokenize


def _clone_cached_state(cached_state):
    if cached_state is None:
        return None

    def clone_state(state):
        if isinstance(state, list):
            return [clone_state(state_i) for state_i in state]
        return state.clone() if state is not None else None

    return tuple(map(clone_state, cached_state))


class RawOutExternalLanguageModelBase(FairseqLanguageModel):
    """Base class for all external language models for ASR whose raw forward output
    will be directly used by the caller (rather than, for example, doing normalization
    by the caller).
    """
    def __init__(self, decoder):
        super().__init__(decoder)


class LookAheadWordLanguageModel(RawOutExternalLanguageModelBase):
    """A :class:`fairseq.models.external_language_model.RawOutExternalLanguageModelBase`
    wrapper for :class:`_LookAheadWordLanguageModelDecoder`.
    """
    def __init__(self, wordlm, subword_dict, oov_penalty=1e-4, open_vocab=True):
        decoder = _LookAheadWordLanguageModelDecoder(
            wordlm, subword_dict, oov_penalty, open_vocab,
        )
        super().__init__(decoder)


class _LookAheadWordLanguageModelDecoder(FairseqIncrementalDecoder):
    """Look-ahead word language model decoder for end-to-end ASR. It is intended
    to be used for beam search decoding. See https://arxiv.org/abs/1808.02608
    for details. We modify the original algorithm a little bit to adapt it to
    the case where each tokenized sentence ends with <space> before <eos>.
    """
    def __init__(self, wordlm, subword_dict, oov_penalty=1e-4, open_vocab=True):
        super().__init__(wordlm.decoder.dictionary)

        assert isinstance(wordlm, FairseqLanguageModel)
        self.lm_decoder = wordlm.decoder
        assert (
            hasattr(self.lm_decoder, "masked_copy_incremental_state")
            and callable(self.lm_decoder.masked_copy_incremental_state)
        ), "The wrapped decoder should implement masked_copy_incremental_state()"
        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.zero = 1e-10  # a sufficiently small value to avoid the log(0) issue

        word_dict = self.lm_decoder.dictionary
        assert isinstance(word_dict, AsrDictionary)
        self.word_pad_idx = word_dict.pad()
        self.word_eos_idx = word_dict.eos()
        self.word_unk_idx = word_dict.unk()

        assert isinstance(subword_dict, AsrDictionary)
        self.subword_space_idx = subword_dict.space()
        self.subword_pad_idx = subword_dict.pad()
        self.subword_eos_idx = subword_dict.eos()
        self.subword_vocab_size = len(subword_dict)

        def tokenizer(x):
            return tokenize(x, non_lang_syms=subword_dict.non_lang_syms).split(" ")
        self.lexroot = lexical_prefix_tree(word_dict, subword_dict, tokenizer)

        def max_out_degree(node):
            if len(node.children) == 0:
                return 0
            cur_max = len(node.children)
            for _, node in node.children.items():
                cur_max = max(cur_max, max_out_degree(node))
            return cur_max

        self.max_num_children = max_out_degree(self.lexroot)
        assert self.max_num_children <= self.subword_vocab_size

    @torch.no_grad()
    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        assert incremental_state is not None, "this model is for incremental decoding only"
        prev_output_tokens = prev_output_tokens[:, -1:]
        bsz = prev_output_tokens.size(0)

        batch_space_mask = prev_output_tokens.squeeze(-1).eq(self.subword_space_idx)

        cached_state = self.lm_decoder.get_incremental_state(incremental_state, "cached_state")

        if cached_state is None:  # it is the first time step
            assert (prev_output_tokens == self.subword_eos_idx).all(), \
                "expecting the input to the first time step to be <eos>"
            w = prev_output_tokens.new_full([bsz, 1], self.word_eos_idx)
            lm_probs = self.lm_decoder.get_normalized_probs(
                self.lm_decoder(w, incremental_state=incremental_state),
                log_probs=False, sample=None,
            )  # B x 1 x V
            cumsum_probs = torch.cumsum(lm_probs, dim=-1)  # B x 1 x V
            nodes = [self.lexroot] * bsz
        else:
            cumsum_probs = self.get_incremental_state(incremental_state, "cumsum_probs")
            nodes = self.get_incremental_state(incremental_state, "nodes")
            assert len(nodes) == bsz
            w = prev_output_tokens.new([
                node.word_idx if node is not None and node.word_idx >= 0 else
                self.word_unk_idx for node in nodes
            ]).unsqueeze(-1)  # B x 1
            old_cached_state = _clone_cached_state(self.lm_decoder.get_cached_state(incremental_state))
            # recompute cumsum_probs from inter-word transition probabilities
            # only for those whose prev_output_token is <space>
            lm_probs = self.lm_decoder.get_normalized_probs(
                self.lm_decoder(w, incremental_state=incremental_state),
                log_probs=False, sample=None,
            )  # B x 1 x V
            self.lm_decoder.masked_copy_incremental_state(
                incremental_state, old_cached_state, batch_space_mask,
            )  # restore those not masked
            cumsum_probs[batch_space_mask] = (
                torch.cumsum(lm_probs, dim=-1)[batch_space_mask]
            )
            tokens_list = prev_output_tokens.squeeze(-1).tolist()
            for i in range(bsz):
                if tokens_list[i] == self.subword_space_idx:
                    # inter-word transition: go back to root
                    nodes[i] = self.lexroot
                elif nodes[i] is not None and tokens_list[i] in nodes[i].children:
                    # intra-word transition: go to child
                    nodes[i] = nodes[i].children[tokens_list[i]]
                else:  # no path in the tree
                    nodes[i] = None

        self.set_incremental_state(incremental_state, "cumsum_probs", cumsum_probs)
        self.set_incremental_state(incremental_state, "nodes", nodes)

        # initialize out_probs (B x 1 x V)
        if self.open_vocab:
            # set out_probs to oov_penalty * P(<unk>|h) (case 3 in Eqn. 15)
            out_probs = self.oov_penalty * (
                cumsum_probs[:, :, self.word_unk_idx] -
                cumsum_probs[:, :, self.word_unk_idx - 1]
            ).unsqueeze(-1).repeat(1, 1, self.subword_vocab_size)
            # set the probability of emitting <space> to 0 if prev_output_tokens
            # is <space> or <eos>, and that of emitting <eos> to 0 if
            # prev_output_tokens is not <space>
            batch_space_eos_mask = (
                batch_space_mask |
                prev_output_tokens.squeeze(-1).eq(self.subword_eos_idx)
            )
            out_probs[batch_space_eos_mask, :, self.subword_space_idx] = self.zero
            out_probs[~batch_space_mask, :, self.subword_eos_idx] = self.zero
            # set transition probability to 1 for those whose node is out of the
            # tree, i.e. node is None (case 4 in Eqn. 15)
            batch_node_none_mask = batch_space_mask.new(
                [node is None for node in nodes]
            )
            out_probs[batch_node_none_mask] = 1.0
        else:
            # set out_probs to 0
            out_probs = cumsum_probs.new_full([bsz, 1, self.subword_vocab_size], self.zero)

        # compute parent probabilities for those whose node is not None
        sum_probs = cumsum_probs.new_full([bsz, 1], 1.0)  # default for root node
        left_ranges, right_ranges, batch_node_not_root_mask = [], [], []
        for node in nodes:
            if node is not None and node.word_set is not None:
                left_ranges.append([node.word_set[0]])
                right_ranges.append([node.word_set[1]])
                batch_node_not_root_mask.append(True)
            else:
                batch_node_not_root_mask.append(False)
        if len(left_ranges) > 0:
            # b x 1 x 1
            left_ranges = prev_output_tokens.new(left_ranges).unsqueeze(-1)
            right_ranges = prev_output_tokens.new(right_ranges).unsqueeze(-1)
            batch_node_not_root_mask = batch_space_mask.new(batch_node_not_root_mask)
            sum_probs[batch_node_not_root_mask] = (
                cumsum_probs[batch_node_not_root_mask].gather(-1, right_ranges) -
                cumsum_probs[batch_node_not_root_mask].gather(-1, left_ranges)
            ).squeeze(-1)

        # compute transition probabilities to child nodes (case 2 in Eqn. 15)
        subword_idx = [
            [self.subword_pad_idx] * self.max_num_children for _ in range(bsz)
        ]
        left_ranges = [
            [self.word_pad_idx] * self.max_num_children for _ in range(bsz)
        ]
        right_ranges = [
            [self.word_pad_idx] * self.max_num_children for _ in range(bsz)
        ]
        for i in range(bsz):
            node = nodes[i]
            if node is not None and len(node.children) > 0:
                for j, (sidx, child) in enumerate(node.children.items()):
                    subword_idx[i][j] = sidx
                    left_ranges[i][j] = child.word_set[0]
                    right_ranges[i][j] = child.word_set[1]
        # B x 1 x max_num_children
        subword_idx = prev_output_tokens.new(subword_idx).unsqueeze(1)
        left_ranges = prev_output_tokens.new(left_ranges).unsqueeze(1)
        right_ranges = prev_output_tokens.new(right_ranges).unsqueeze(1)
        cumsum_probs_children = (
            cumsum_probs.gather(-1, right_ranges) -
            cumsum_probs.gather(-1, left_ranges)
        ) / sum_probs.unsqueeze(-1)
        cumsum_probs_children[sum_probs.squeeze(-1) < self.zero, :, :] = self.zero
        out_probs.scatter_(-1, subword_idx, cumsum_probs_children)
        out_probs[:, :, self.subword_pad_idx] = self.zero

        # apply word-level probabilies for <space> (case 1 in Eqn. 15)
        word_idx, batch_node_word_end_mask = [], []
        for node in nodes:
            if node is not None and node.word_idx >= 0:
                word_idx.append([node.word_idx])
                batch_node_word_end_mask.append(True)
            else:
                batch_node_word_end_mask.append(False)
        if len(word_idx) > 0:
            word_idx = prev_output_tokens.new(word_idx).unsqueeze(-1)  # b x 1 x 1
            batch_node_word_end_mask = batch_space_mask.new(batch_node_word_end_mask)
            word_probs = torch.where(
                sum_probs[batch_node_word_end_mask] < self.zero,
                cumsum_probs.new([self.zero]),
                (
                    cumsum_probs[batch_node_word_end_mask].gather(-1, word_idx) -
                    cumsum_probs[batch_node_word_end_mask].gather(-1, word_idx - 1)
                ).squeeze(-1).div_(sum_probs[batch_node_word_end_mask]),
            )  # b x 1
            out_probs[batch_node_word_end_mask, :, self.subword_space_idx] = word_probs

        # take log of probs and clip it from below to avoid log(0)
        out_logprobs = out_probs.clamp(min=self.zero).log_()

        # assign log-probs of emitting word <eos> to that of emitting subword <eos>
        out_logprobs[batch_space_mask, :, self.subword_eos_idx] = (
            lm_probs.log_()[batch_space_mask, :, self.word_eos_idx]
        )

        # note that here we return log-probs rather than logits, and the second
        # element is None, which is usually a tensor of attention weights in
        # attention-based models
        return out_logprobs, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)

        cumsum_probs = self.get_incremental_state(incremental_state, "cumsum_probs")
        if cumsum_probs is not None:
            new_cumsum_probs = cumsum_probs.index_select(0, new_order)
            self.set_incremental_state(incremental_state, "cumsum_probs", new_cumsum_probs)

        nodes = self.get_incremental_state(incremental_state, "nodes")
        if nodes is not None:
            new_order_list = new_order.tolist()
            new_nodes = [nodes[i] for i in new_order_list]
            self.set_incremental_state(incremental_state, "nodes", new_nodes)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # in-place op as not being used for backprop
        return net_output[0] if log_probs else net_output[0].exp_()

    def max_positions(self):
        return int(1e5)  # an arbitrary large number


class MultiLevelLanguageModel(RawOutExternalLanguageModelBase):
    """A :class:`fairseq.external_language_model.RawOutExternalLanguageModelBase`
    wrapper for :class:`_MultiLevelLanguageModel`.
    """
    def __init__(
        self, wordlm, subwordlm, subwordlm_weight=0.8, oov_penalty=1.0, open_vocab=True,
    ):
        decoder = _MultiLevelLanguageModel(
            wordlm, subwordlm, subwordlm_weight, oov_penalty, open_vocab,
        )
        super().__init__(decoder)


class _MultiLevelLanguageModel(FairseqIncrementalDecoder):
    """Multi-level (subword/word) language model decoder for end-to-end ASR.
    It is intended to be used for beam search decoding.
    See https://ieeexplore.ieee.org/document/8268948 for details. We modify the
    original algorithm a little bit to adapt it to the case where each tokenized
    sentence ends with <space> before <eos>.
    """
    def __init__(
        self, wordlm, subwordlm, subwordlm_weight=0.8, oov_penalty=1.0, open_vocab=True,
    ):
        super().__init__(wordlm.decoder.dictionary)

        assert isinstance(wordlm, FairseqLanguageModel)
        self.wordlm_decoder = wordlm.decoder
        assert (
            hasattr(self.wordlm_decoder, "masked_copy_incremental_state") and
            callable(self.wordlm_decoder.masked_copy_incremental_state)
        ), "The wrapped decoder should implement masked_copy_incremental_state()"
        assert isinstance(subwordlm, FairseqLanguageModel)
        self.subwordlm_decoder = subwordlm.decoder
        self.subwordlm_weight = subwordlm_weight
        self.log_oov_penalty = math.log(oov_penalty)
        self.open_vocab = open_vocab
        self.logzero = -10.0

        word_dict = self.wordlm_decoder.dictionary
        assert isinstance(word_dict, AsrDictionary)
        self.word_eos_idx = word_dict.eos()
        self.word_unk_idx = word_dict.unk()

        subword_dict = self.subwordlm_decoder.dictionary
        assert isinstance(subword_dict, AsrDictionary)
        self.subword_space_idx = subword_dict.space()
        self.subword_eos_idx = subword_dict.eos()
        self.subword_vocab_size = len(subword_dict)

        def tokenizer(x):
            return tokenize(x, non_lang_syms=subword_dict.non_lang_syms).split(" ")
        self.lexroot = lexical_prefix_tree(word_dict, subword_dict, tokenizer)

    @torch.no_grad()
    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        assert incremental_state is not None, "this model is for incremental decoding only"
        prev_output_tokens = prev_output_tokens[:, -1:]
        bsz = prev_output_tokens.size(0)

        batch_space_mask = prev_output_tokens.squeeze(-1).eq(self.subword_space_idx)
        batch_not_space_mask = ~batch_space_mask

        wordlm_cached_state = self.wordlm_decoder.get_incremental_state(
            incremental_state, "cached_state",
        )
        subwordlm_cached_state = self.subwordlm_decoder.get_incremental_state(
            incremental_state, "cached_state",
        )

        if wordlm_cached_state is None:  # it is the first time step
            assert subwordlm_cached_state is None
            assert (prev_output_tokens == self.subword_eos_idx).all(), \
                "expecting the input to the first time step to be <eos>"
            w = prev_output_tokens.new_full([bsz, 1], self.word_eos_idx)
            wordlm_logprobs = self.wordlm_decoder.get_normalized_probs(
                self.wordlm_decoder(w, incremental_state=incremental_state),
                log_probs=True,
                sample=None,
            )  # B x 1 x V
            sw = prev_output_tokens.new_full([bsz, 1], self.subword_eos_idx)
            out_logprobs = self.subwordlm_decoder.get_normalized_probs(
                self.subwordlm_decoder(sw, incremental_state=incremental_state),
                log_probs=True,
                sample=None,
            ) * self.subwordlm_weight  # B x 1 x V
            subword_cumlogprobs = out_logprobs.new_zeros(sw.size())
            nodes = [self.lexroot] * bsz
        else:
            wordlm_logprobs = self.get_incremental_state(incremental_state, "wordlm_logprobs")
            out_logprobs = self.get_incremental_state(incremental_state, "out_logprobs")
            subword_cumlogprobs = self.get_incremental_state(incremental_state, "subword_cumlogprobs")
            nodes = self.get_incremental_state(incremental_state, "nodes")
            assert len(nodes) == bsz
            w = prev_output_tokens.new([
                node.word_idx if node is not None and node.word_idx >= 0 else
                self.word_unk_idx for node in nodes
            ]).unsqueeze(-1)  # B x 1
            old_wordlm_cached_state = _clone_cached_state(self.wordlm_decoder.get_cached_state(incremental_state))

            # recompute wordlm_logprobs from inter-word transition probabilities
            # only for those whose prev_output_token is <space>
            wordlm_logprobs[batch_space_mask] = self.wordlm_decoder.get_normalized_probs(
                self.wordlm_decoder(w, incremental_state=incremental_state),
                log_probs=True,
                sample=None,
            )[batch_space_mask]
            self.wordlm_decoder.masked_copy_incremental_state(
                incremental_state, old_wordlm_cached_state, batch_space_mask,
            )  # restore those not masked

            tokens_list = prev_output_tokens.squeeze(-1).tolist()
            token_idx, batch_is_child_mask = [], []
            for i in range(bsz):
                if tokens_list[i] == self.subword_space_idx:
                    # inter-word transition: go back to root
                    nodes[i] = self.lexroot
                    batch_is_child_mask.append(False)
                elif nodes[i] is not None and tokens_list[i] in nodes[i].children:
                    # intra-word transition: go to child
                    nodes[i] = nodes[i].children[tokens_list[i]]
                    token_idx.append([tokens_list[i]])
                    batch_is_child_mask.append(True)
                else:  # no path in the tree
                    nodes[i] = None
                    if self.open_vocab:
                        token_idx.append([tokens_list[i]])
                    batch_is_child_mask.append(False)
            token_idx = prev_output_tokens.new(token_idx).unsqueeze(-1)  # b x 1 x 1
            if self.open_vocab:
                subword_cumlogprobs[batch_space_mask] = 0.0
                assert batch_not_space_mask.sum().item() == len(token_idx)
                subword_cumlogprobs[batch_not_space_mask] += (
                    out_logprobs[batch_not_space_mask].gather(-1, token_idx).squeeze(-1)
                )
            else:
                subword_cumlogprobs[~batch_is_child_mask] = 0.0
                assert batch_is_child_mask.sum().item() == len(token_idx)
                subword_cumlogprobs[batch_is_child_mask] += (
                    out_logprobs[batch_is_child_mask].gather(-1, token_idx).squeeze(-1)
                )

            out_logprobs = self.subwordlm_decoder.get_normalized_probs(
                self.subwordlm_decoder(prev_output_tokens, incremental_state=incremental_state),
                log_probs=True,
                sample=None,
            ) * self.subwordlm_weight

            if not self.open_vocab:
                batch_oov_mask = batch_not_space_mask & ~batch_is_child_mask
                out_logprobs[batch_oov_mask] = self.logzero

        self.set_incremental_state(incremental_state, "wordlm_logprobs", wordlm_logprobs)
        self.set_incremental_state(incremental_state, "subword_cumlogprobs", subword_cumlogprobs)
        self.set_incremental_state(incremental_state, "nodes", nodes)

        # apply word-level probabilies for emitting <space>
        w = prev_output_tokens.new([
            node.word_idx if node is not None and node.word_idx >= 0 else
            self.word_unk_idx for node in nodes
        ]).unsqueeze(-1)  # B x 1
        word_logprobs = wordlm_logprobs.gather(-1, w.unsqueeze(-1)).squeeze(-1)  # B x 1
        batch_word_end_mask = w.ne(self.word_unk_idx)
        word_logprobs += torch.where(
            batch_word_end_mask,
            -subword_cumlogprobs, word_logprobs.new([self.log_oov_penalty]),
        )
        out_logprobs[:, :, self.subword_space_idx] = word_logprobs

        # set the probability of emitting <space> to 0 if prev_output_tokens is
        # <space> or <eos>, and that of emitting <eos> to 0 if prev_output_tokens
        # is not <space>
        batch_space_eos_mask = (
            batch_space_mask | prev_output_tokens.squeeze(-1).eq(self.subword_eos_idx)
        )
        out_logprobs[batch_space_eos_mask, :, self.subword_space_idx] = self.logzero
        out_logprobs[~batch_space_mask, :, self.subword_eos_idx] = self.logzero

        # add log-probs of emitting word <eos> to that of emitting subword <eos>
        out_logprobs[batch_space_mask, :, self.subword_eos_idx] += (
            wordlm_logprobs[batch_space_mask, :, self.word_eos_idx]
        )

        self.set_incremental_state(incremental_state, "out_logprobs", out_logprobs)

        # note that here we return log-probs rather than logits, and the second
        # element is None, which is usually a tensor of attention weights in
        # attention-based models
        return out_logprobs, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)

        for state_name in ["wordlm_logprobs", "out_logprobs", "subword_cumlogprobs"]:
            state = self.get_incremental_state(incremental_state, state_name)
            if state is not None:
                new_state = state.index_select(0, new_order)
                self.set_incremental_state(incremental_state, state_name, new_state)

        nodes = self.get_incremental_state(incremental_state, "nodes")
        if nodes is not None:
            new_order_list = new_order.tolist()
            new_nodes = [nodes[i] for i in new_order_list]
            self.set_incremental_state(incremental_state, "nodes", new_nodes)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # in-place op as not being used for backprop
        return net_output[0] if log_probs else net_output[0].exp_()

    def max_positions(self):
        return int(1e5)  # an arbitrary large number
