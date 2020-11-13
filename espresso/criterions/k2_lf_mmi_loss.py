# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import math
from typing import List

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from omegaconf import II

try:
    import k2
except ImportError:
    raise ImportError("Please install k2 by `pip install k2`")


logger = logging.getLogger(__name__)


@dataclass
class K2LatticeFreeMMICriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    denominator_fst_path: str = field(
        default="???", metadata={"help": "path to the denominator fst file (torch saved)"}
    )
    HCL_fst_path: str = field(
        default="???", metadata={"help": "path to the HCL fst file (torch saved)"}
    )
    word_symbol_table_path: str = field(
        default="???", metadata={"help": "path to the word symbol table file"}
    )
    leaky_hmm_coefficient: float = field(
        default=1.0e-05,
        metadata={"help": "leaky-hmm coefficient for the denominator"},
    )
    xent_regularization_coefficient: float = field(
        default=0.0,
        metadata={"help": "cross-entropy regularization coefficient"},
    )
    output_l2_regularization_coefficient: float = field(
        default=0.0,
        metadata={"help": "L2 regularization coefficient for the network's output"},
    )


def create_numerator_graphs(texts: List[str], HCL_fst_inv: k2.Fsa, symbols: k2.SymbolTable):
    word_ids_list = []
    for text in texts:
        filtered_text = [
            word if word in symbols._sym2id else "<UNK>" for word in text.split(" ")
        ]
        word_ids = [symbols.get(word) for word in filtered_text]
        word_ids_list.append(word_ids)

    fsa = k2.linear_fsa(word_ids_list)  # create an FsaVec from a list of list of word ids
    num_graph = k2.intersect(fsa, HCL_fst_inv).invert_()
    num_graph = k2.add_epsilon_self_loops(num_graph)
    return num_graph


@register_criterion("k2_lattice_free_mmi", dataclass=K2LatticeFreeMMICriterionConfig)
class K2LatticeFreeMMICriterion(FairseqCriterion):

    def __init__(
        self, task, sentence_avg, denominator_fst_path, HCL_fst_path, word_symbol_table_path,
        leaky_hmm_coefficient, xent_regularization_coefficient, output_l2_regularization_coefficient,
    ):
        super().__init__(task)

        self.sentence_avg = sentence_avg
        self.den_graph = k2.create_fsa_vec(
            k2.Fsa.from_dict(torch.load(denominator_fst_path))
        )  # has to be FsaVec to be able to intersect with a batch of dense fsas
        self.den_graph.scores.requires_grad_(False)
        self.HCL_fst_inv = k2.Fsa.from_dict(torch.load(HCL_fst_path)).invert_()
        self.symbol_table = k2.SymbolTable.from_file(word_symbol_table_path)
        self.leaky_hmm_coefficient = leaky_hmm_coefficient
        self.xent_regularize = xent_regularization_coefficient
        self.output_l2_regularize = output_l2_regularization_coefficient

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].batch_size if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        # create the dense fsts from the network's output
        encoder_out = net_output.encoder_out.transpose(0, 1)  # T x B x V -> B x T x V
        out_lengths = net_output.src_lengths.long()  # B
        supervision_segments = torch.stack(
            # seq_index, start_frame, lengths
            (sample["target"]["sequence_idx"], sample["target"]["start_frame"], out_lengths),
            dim=1
        )
        dense_fsa_vec = k2.DenseFsaVec(encoder_out, supervision_segments)

        # numerator computation
        num_graphs = create_numerator_graphs(sample["target"]["text"], self.HCL_fst_inv, self.symbol_table)
        num_graphs.to_(encoder_out.device)
        num_graphs.scores.requires_grad_(False)
        num_graphs_unrolled = k2.intersect_dense_pruned(
            num_graphs, dense_fsa_vec, beam=100000, max_active_states=10000, min_active_states=0
        )
        num_scores = k2.get_tot_scores(num_graphs_unrolled, log_semiring=False, use_float_scores=True)

        # denominator computation
        self.den_graph.to_(encoder_out.device)
        den_graph_unrolled = k2.intersect_dense_pruned(
            self.den_graph, dense_fsa_vec, beam=100000, max_active_states=10000, min_active_states=0
        )
        den_scores = k2.get_tot_scores(den_graph_unrolled, log_semiring=False, use_float_scores=True)

        # obtain the loss
        loss = -num_scores + den_scores
        nll_loss = loss.clone().detach()
        if self.xent_regularize > 0.0:
            loss -= self.xent_regularize * num_scores

        if self.output_l2_regularize > 0.0:
            encoder_padding_mask = net_output.encoder_padding_mask
            encoder_out_squared = encoder_out.pow(2.0)
            if encoder_padding_mask is not None:
                pad_mask = encoder_padding_mask.transpose(0, 1).unsqueeze(-1)  # T x B -> B x T x 1
                encoder_out_squared.masked_fill_(pad_mask, 0.0)
            loss += 0.5 * self.output_l2_regularize * encoder_out_squared.sum()

        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=7
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=7
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, round=4)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
