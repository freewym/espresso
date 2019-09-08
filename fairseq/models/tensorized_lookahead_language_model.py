from typing import *
import torch

from speech_tools.tensorized_prefix_tree import TensorizedPrefixTree
from fairseq.models import FairseqLanguageModel, FairseqIncrementalDecoder
from fairseq.data import TokenDictionary
from fairseq import utils

from speech_tools.utils import tokenize


def _clone_cached_state(cached_state):
    if cached_state is None:
        return None

    def clone_state(state):
        if isinstance(state, list):
            return [clone_state(state_i) for state_i in state]
        return state.clone() if state is not None else None

    return tuple(map(clone_state, cached_state))


class TensorizedLookaheadLanguageModel(FairseqLanguageModel):

    def __init__(self,
                 word_lm: FairseqLanguageModel,
                 subword_dict: TokenDictionary,
                 oov_penalty: float = 1e-4,
                 open_vocab: bool = True
                 ):
        decoder = _TensorizedLookaheadLanguageModelDecoder(word_lm, subword_dict, oov_penalty, open_vocab)
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        raise NotImplementedError


class _TensorizedLookaheadLanguageModelDecoder(FairseqIncrementalDecoder):

    def __init__(self,
                 word_lm: FairseqLanguageModel,
                 subword_dict: TokenDictionary,
                 oov_penalty: float = 1e-4,
                 open_vocab: bool = True):
        super().__init__(word_lm.decoder.dictionary)

        self.lm_decoder: FairseqIncrementalDecoder = word_lm.decoder
        assert hasattr(self.lm_decoder, 'masked_copy_incremental_state') and \
            callable(self.lm_decoder.masked_copy_incremental_state), \
            'The wrapped decoder should implement masked_copy_incremental_state()'

        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.zero = 1e-10

        word_dict: TokenDictionary = self.lm_decoder.dictionary
        self.word_pad_idx = word_dict.pad()
        self.word_eos_idx = word_dict.eos()
        self.word_unk_idx = word_dict.unk()

        self.subword_space_idx = subword_dict.space()
        self.subword_pad_idx = subword_dict.pad()
        self.subword_eos_idx = subword_dict.eos()
        self.subword_vocab_size = len(subword_dict)

        tokenizer: Callable[[str], List[str]] = lambda x: tokenize(x, subword_dict.non_lang_syms).split(' ')
        self.tree = TensorizedPrefixTree.build(word_dict, subword_dict, tokenizer)

        assert self.tree.max_out_degree() <= self.subword_vocab_size

    @torch.no_grad()
    def forward(self,
                prev_output_tokens: torch.Tensor,  # Z_Tokens[Batch, SeqLength]
                encoder_out=None,
                incremental_state: Dict[str, Any] = None,
                **kwargs):

        if prev_output_tokens.device != self.tree.word_idx.device:
            self.tree.to_cuda(device=prev_output_tokens.device)

        assert incremental_state is not None, 'This model is for incremental decoding only'
        prev_output_tokens = prev_output_tokens[:, -1:]  # Z_Tokens[Batch, Len=1]
        bsz = prev_output_tokens.size(0)

        batch_range = torch.arange(0, bsz, device=prev_output_tokens.device)

        # Move the batched state to the next state according to the automaton
        batch_space_mask = prev_output_tokens.squeeze(-1).eq(self.subword_space_idx)
        cached_state = utils.get_incremental_state(self.lm_decoder, incremental_state, 'cached_state')

        if cached_state is None:  # First step
            assert (prev_output_tokens == self.subword_eos_idx).all(), \
                'expecting the input to the first time step to be <eos>'
            w: torch.Tensor = prev_output_tokens.new_full([bsz, 1], self.word_eos_idx)  # Z[Batch, Len=1]
            lm_probs: torch.Tensor = self.lm_decoder.get_normalized_probs(
                self.lm_decoder(w, incremental_state), log_probs=False, sample=None
            )  # R[Batch, 1, Vocab]
            cumsum_probs: torch.Tensor = lm_probs.cumsum(dim=-1)  # R[Batch, 1, Vocab]
            nodes: torch.Tensor = torch.full([bsz], self.tree.root_id, dtype=torch.int64)  # Z_NodeId[Batch]
            all_children = self.tree.children[nodes, :]  # Z[Batch, PossibleChildren]
            prev_possible_tokens = self.tree.prev_subword_idx[all_children]  # Z[Batch, PossibleChildren]

        else:  # Not the first step
            cumsum_probs: torch.Tensor = utils.get_incremental_state(
                self, incremental_state, 'cumsum_probs')  # R[Batch, 1, Vocab]
            nodes: torch.Tensor = utils.get_incremental_state(self, incremental_state, 'nodes')  # Z_NodeId[Batch]
            assert nodes.size(0) == bsz
            w: torch.Tensor = self.tree.word_idx[nodes].unsqueeze(dim=1)  # Z[Batch, Len=1]
            w[w < 0] = self.word_unk_idx

            old_cached_state = _clone_cached_state(cached_state)
            # recompute cumsum_probs from inter-word transition probabilities
            # only for those whose prev_output_token is <space>
            lm_probs: torch.Tensor = self.lm_decoder.get_normalized_probs(
                self.lm_decoder(w, incremental_state), log_probs=False, sample=None
            )  # R[Batch, 1, Vocab]
            self.lm_decoder.masked_copy_incremental_state(
                incremental_state, old_cached_state, batch_space_mask)  # restore those not masked
            cumsum_probs[batch_space_mask] = lm_probs.cumsum(dim=-1)[batch_space_mask]

            all_children = self.tree.children[nodes, :]  # Z[Batch, PossibleChildren]
            prev_possible_tokens = self.tree.prev_subword_idx[all_children]  # Z[Batch, PossibleChildren]
            mask = prev_possible_tokens.eq(prev_output_tokens.expand_as(prev_possible_tokens))
            nodes: torch.Tensor = (all_children * mask).sum(dim=1)  # Z[Batch]
            nodes[batch_space_mask] = self.tree.root_id  # Z[Batch]

        utils.set_incremental_state(self, incremental_state, 'cumsum_probs', cumsum_probs)
        utils.set_incremental_state(self, incremental_state, 'nodes', nodes)

        # Compute probabilities
        # initialize out_probs [Batch, 1, Vocab]
        if self.open_vocab:
            out_probs = self.oov_penalty * lm_probs[:, :, self.word_unk_idx] \
                .unsqueeze(-1).repeat(1, 1, self.subword_vocab_size)

            # set the probability of emitting <space> to 0 if prev_output_tokens
            # is <space> or <eos>, and that of emitting <eos> to 0 if
            # prev_output_tokens is not <space>
            batch_space_eos_mask = batch_space_mask | \
                prev_output_tokens.squeeze(-1).eq(self.subword_eos_idx)
            out_probs[batch_space_eos_mask, :, self.subword_space_idx] = self.zero
            out_probs[~batch_space_mask, :, self.subword_eos_idx] = self.zero
            # set transition probability to 1 for those whose node is out of the
            # tree, i.e. node is None (case 4 in Eqn. 15)

            batch_node_none_mask = (nodes == self.tree.none_id)  # B[Batch]
            out_probs[batch_node_none_mask] = 1.
        else:
            # set out_probs to 0
            out_probs = cumsum_probs.new_full([bsz, 1, self.subword_vocab_size], self.zero)

        # compute parent probabilities for those whose node is not None
        left_ranges = self.tree.word_set_idx[nodes, 0]  # Z[Batch]
        right_ranges = self.tree.word_set_idx[nodes, 1]  # Z[Batch]
        batch_node_not_root_mask = (nodes != self.tree.root_id) & (nodes != self.tree.none_id)  # B[Batch]

        sum_probs = torch.where(
            batch_node_not_root_mask,
            cumsum_probs.squeeze(dim=1)[batch_range, right_ranges] -
            cumsum_probs.squeeze(dim=1)[batch_range, left_ranges],
            cumsum_probs.new([1.0])
        )  # R[Batch]

        left_ranges_of_all_children = self.tree.word_set_idx[all_children, 0]  # Z[Batch, PossibleChildren]
        right_ranges_of_all_children = self.tree.word_set_idx[all_children, 1]  # Z[Batch, PossibleChildren]

        cumsum_probs_of_all_children = (
            cumsum_probs.squeeze(dim=1).gather(-1, right_ranges_of_all_children) -
            cumsum_probs.squeeze(dim=1).gather(-1, left_ranges_of_all_children)
        ) / sum_probs.unsqueeze(dim=1)

        out_probs.scatter_(-1, prev_possible_tokens.unsqueeze(dim=1), cumsum_probs_of_all_children.unsqueeze(dim=1))
        out_probs[:, :, self.subword_pad_idx] = self.zero

        # apply word-level probabilities for <space> (case 1 in Eqn. 15)
        word_idx = self.tree.word_idx[nodes]  # Z[Batch]
        batch_node_word_end_mask = word_idx >= 0  # B[Batch]
        word_idx[word_idx < 0] = 0  # get rid of -1's (word idx of root or non-terminal states)

        word_probs = torch.where(
            sum_probs < self.zero,
            cumsum_probs.new([self.zero]),
            lm_probs[batch_range, 0, word_idx]
        ).squeeze(-1) / sum_probs  # R[Batch]

        out_probs[:, 0, self.subword_space_idx] = torch.where(
            batch_node_word_end_mask,
            word_probs,
            out_probs[:, 0, self.subword_space_idx]
        )

        # take log of probs and clip it from below to avoid log(0)
        out_logprobs = torch.log(torch.max(out_probs, out_probs.new([self.zero])))

        # assign log-probs of emitting word <eos> to that of emitting subword <eos>
        out_logprobs[batch_space_mask, :, self.subword_eos_idx] = \
            torch.log(lm_probs)[batch_space_mask, :, self.word_eos_idx]

        utils.set_incremental_state(self, incremental_state, 'out_logprobs', out_logprobs)

        # note that here we return log-probs rather than logits, and the second
        # element is None, which is usually a tensor of attention weights in
        # attention-based models
        return out_logprobs, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)

        cumsum_probs = utils.get_incremental_state(
            self, incremental_state, 'cumsum_probs')
        if cumsum_probs is not None:
            new_cumsum_probs = cumsum_probs.index_select(0, new_order)
            utils.set_incremental_state(self, incremental_state, 'cumsum_probs',
                new_cumsum_probs)

        nodes = utils.get_incremental_state(self, incremental_state, 'nodes')
        if nodes is not None:
            new_order_list = new_order.tolist()
            new_nodes = [nodes[i] for i in new_order_list]
            utils.set_incremental_state(self, incremental_state, 'nodes',
                new_nodes)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # in-place op as not being used for backprop
        return net_output[0] if log_probs else net_output[0].exp_()

    def max_positions(self):
        return int(1e5)  # an arbitrary large number

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        pass

    def output_layer(self, features, **kwargs):
        pass

