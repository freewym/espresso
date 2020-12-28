# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import lru_cache
import logging
import math
from omegaconf import II
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.tasks import FairseqTask
from fairseq.logging import metrics

try:
    import k2
except ImportError:
    raise ImportError("Please install k2 by `pip install k2`")


logger = logging.getLogger(__name__)


@dataclass
class K2LatticeFreeMMICriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    denominator_graph_path: str = field(
        default="???", metadata={"help": "path to the denominator graph file (torch saved)"}
    )
    H_path: str = field(
        default="???", metadata={"help": "path to the H fst file (torch saved). Note: pdf-ids are offset by +1"}
    )
    L_path: str = field(
        default="???", metadata={"help": "path to the L fst file (openfst text format or torch saved)"}
    )
    word_symbol_table_path: str = field(
        default="???", metadata={"help": "path to the word symbol table file"}
    )
    phone_symbol_table_path: str = field(
        default="???", metadata={"help": "path to the phone symbol table file"}
    )
    xent_regularization_coefficient: float = field(
        default=0.0,
        metadata={"help": "cross-entropy regularization coefficient"},
    )


class TrainingGraphCompiler(object):
    """
    :class:`TrainingGraphCompiler` is used to create training graphs (numerator graphs) for LF-MMI.

    Args:
        H_inv (k2.Fsa): invert of H. Note: H.labels has been offset by +1
        L_inv (k2.Fsa): invert of L
        symbol_table (k2.SymbolTable): word symbol table
        phone_symbol_table (k2.SymbolTable): phone symbol table
        unk_str (optional: str): unk string
        den_graph (optional: k2.Fsa): denominator graph, to be composed with numerator graphs for normalization
    """
    def __init__(
        self, H_inv: k2.Fsa, L_inv: k2.Fsa, symbol_table: k2.SymbolTable, phone_symbol_table: k2.SymbolTable,
        unk_str: Optional[str] = None, den_graph: Optional[k2.Fsa] = None
    ):
        if H_inv.properties & k2.fsa_properties.ARC_SORTED == 0:
            H_inv = k2.arc_sort(H_inv)
        if L_inv.properties & k2.fsa_properties.ARC_SORTED == 0:
            L_inv = k2.arc_sort(L_inv)

        if unk_str is not None:
            assert unk_str in symbol_table
        if den_graph is not None:
            assert len(den_graph.shape) == 2

        self.H_inv = H_inv
        self.L_inv = L_inv
        self.symbol_table = symbol_table
        self.first_phone_disambig_id = min(v for k, v in phone_symbol_table._sym2id.items() if k.startswith("#"))
        self.unk_str = unk_str
        self.den_graph = den_graph

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        num_graphs = k2.create_fsa_vec([self.compile_one_and_cache(text) for text in texts])
        num_graphs.requires_grad_(False)
        return num_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        if self.unk_str is not None:
            tokens = [token if token in self.symbol_table._sym2id else self.unk_str for token in text.split(" ")]
        else:
            tokens = [token for token in text.split(" ") if token in self.symbol_table._sym2id]
        word_ids = [self.symbol_table[token] for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        LG = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        LG = k2.connect(k2.determinize(LG))
        LG.labels[LG.labels >= self.first_phone_disambig_id] = 0
        LG = k2.arc_sort(k2.connect(k2.remove_epsilons_iterative_tropical(LG)))
        del LG.aux_labels
        num_graph = k2.arc_sort(k2.invert(k2.connect(k2.intersect(self.H_inv, LG))))
        num_graph = k2.connect(k2.remove_epsilons_iterative_tropical(num_graph))
        num_graph = k2.connect(k2.determinize(num_graph))
        del num_graph.aux_labels
        num_graph.labels = torch.where(num_graph.labels > 0, num_graph.labels - 1, num_graph.labels)
        if self.den_graph is not None:
            num_graph = k2.arc_sort(num_graph)
            # zero the score before intersect to avoid double counting
            num_graph.scores = num_graph.scores.new_zeros(num_graph.scores.size())
            # treat epsilon as normal labels, i.e., blanks
            num_graph = k2.connect(k2.intersect(num_graph, self.den_graph, treat_epsilons_specially=False))
        return num_graph


@register_criterion("k2_lattice_free_mmi", dataclass=K2LatticeFreeMMICriterionConfig)
class K2LatticeFreeMMICriterion(FairseqCriterion):
    def __init__(self, cfg: K2LatticeFreeMMICriterionConfig, task: FairseqTask):
        super().__init__(task)

        self.sentence_avg = cfg.sentence_avg
        self.den_graph = k2.create_fsa_vec(
            [k2.Fsa.from_dict(torch.load(cfg.denominator_graph_path))]
        )  # has to be an FsaVec to be able to intersect with a batch of dense fsas
        if hasattr(self.den_graph, "aux_labels"):
            del self.den_graph.aux_labels
        self.den_graph.requires_grad_(False)
        H_inv = k2.invert(k2.Fsa.from_dict(torch.load(cfg.H_path)))
        if cfg.L_path[-3:] == ".pt":
             L_inv = k2.Fsa.from_dict(torch.load(args.L_path)).invert_()
        else:
            with open(cfg.L_path, "r", encoding="utf-8") as f:
                L_inv = k2.Fsa.from_openfst(f.read(), acceptor=False).invert_()
        self.graph_compiler = TrainingGraphCompiler(
            H_inv=H_inv,
            L_inv=L_inv,
            symbol_table=k2.SymbolTable.from_file(cfg.word_symbol_table_path),
            phone_symbol_table=k2.SymbolTable.from_file(cfg.phone_symbol_table_path),
            unk_str=task.target_dictionary.unk_string() if task.target_dictionary is not None else None,
            den_graph=self.den_graph[0].clone(),  # to be intersect with a single numerator fsa
        )

        self.xent_regularize = cfg.xent_regularization_coefficient
        self.subsampling_factor = None

    def forward(
        self, model: BaseFairseqModel, sample: List[Dict[str, Any]], reduce: Optional[bool] = True,
    ):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.subsampling_factor is None:
            assert hasattr(model, "output_lengths"), "model should implement the method `output_lengths()`"
            self.subsampling_factor = int(round(120.0 / model.output_lengths(120)))

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

    def compute_loss(
        self, net_output: Dict[str, List[Tensor]], sample: List[Dict[str, Any]], reduce: Optional[bool] = True,
    ):
        # create the dense fsts from the network's output
        encoder_out = net_output["encoder_out"][0].transpose(0, 1)  # T x B x V -> B x T x V
        encoder_out = encoder_out.clamp(-30, 30)  # clamp to avoid numerical overflows
        out_lengths = net_output["src_lengths"][0]  # B
        supervision_segments = torch.stack(
            # seq_index, start_frame, lengths
            (
                sample["target"]["sequence_idx"],
                torch.floor_divide(sample["target"]["start_frame"], self.subsampling_factor),
                out_lengths
            ),
            dim=1
        ).int().cpu()  # assume batched in descending order of lengths
        dense_fsa_vec = k2.DenseFsaVec(encoder_out, supervision_segments)

        # numerator computation
        num_graphs = self.graph_compiler.compile(sample["target"]["text"]).to(encoder_out.device)
        num_graphs_unrolled = k2.intersect_dense_pruned(
            num_graphs, dense_fsa_vec, search_beam=100000, output_beam=100000, min_active_states=0, max_active_states=10000
        )
        num_scores = k2.get_tot_scores(num_graphs_unrolled, log_semiring=True, use_double_scores=False)

        # denominator computation
        self.den_graph = self.den_graph.to(encoder_out.device)
        den_graph_unrolled = k2.intersect_dense_pruned(
            self.den_graph, dense_fsa_vec, search_beam=100000, output_beam=100000, min_active_states=0, max_active_states=10000
        )
        den_scores = k2.get_tot_scores(den_graph_unrolled, log_semiring=True, use_double_scores=False)

        # obtain the loss
        if reduce:
            if torch.isnan(num_scores).int().sum().item() > 0 or torch.isinf(num_scores).int().sum().item() > 0:
                print("num nan", torch.isnan(num_scores).int().sum().item(), "inf", torch.isinf(num_scores).int().sum().item())
            if torch.isnan(den_scores).int().sum().item() > 0 or torch.isinf(den_scores).int().sum().item() > 0:
                print("den nan", torch.isnan(den_scores).int().sum().item(), "inf", torch.isinf(den_scores).int().sum().item())
            num_scores = num_scores.sum()
            den_scores = den_scores.sum()
        loss = -num_scores + den_scores  # negative log-probs
        nll_loss = loss.clone().detach()
        if self.xent_regularize > 0.0:
            loss -= self.xent_regularize * num_scores

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
