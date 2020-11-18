# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import math
from omegaconf import II

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging import metrics


logger = logging.getLogger(__name__)


@dataclass
class LatticeFreeMMICriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    denominator_fst_path: str = field(
        default="???", metadata={"help": "path to the denominator fst file"}
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


class ChainLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_lengths, num_graphs, den_graphs, leaky_coefficient=1e-5):
        try:
            import pychain_C
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` "
                "after entering espresso/tools"
            )

        input = input.clamp(-30, 30)  # clamp for both the denominator and the numerator
        B = input.size(0)
        if B != num_graphs.batch_size or B != den_graphs.batch_size:
            raise ValueError(
                "input batch size ({}) does not equal to num graph batch size ({}) "
                "or den graph batch size ({})"
                .format(B, num_graphs.batch_size, den_graphs.batch_size)
            )
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True,
        )
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()

        exp_input = input.exp()
        den_objf, input_grad, denominator_ok = pychain_C.forward_backward(
            den_graphs.forward_transitions,
            den_graphs.forward_transition_indices,
            den_graphs.forward_transition_probs,
            den_graphs.backward_transitions,
            den_graphs.backward_transition_indices,
            den_graphs.backward_transition_probs,
            den_graphs.leaky_probs,
            den_graphs.initial_probs,
            den_graphs.final_probs,
            den_graphs.start_state,
            exp_input,
            batch_sizes,
            input_lengths,
            den_graphs.num_states,
            leaky_coefficient,
        )
        denominator_ok = denominator_ok.item()

        assert num_graphs.log_domain
        num_objf, log_probs_grad, numerator_ok = pychain_C.forward_backward_log_domain(
            num_graphs.forward_transitions,
            num_graphs.forward_transition_indices,
            num_graphs.forward_transition_probs,
            num_graphs.backward_transitions,
            num_graphs.backward_transition_indices,
            num_graphs.backward_transition_probs,
            num_graphs.initial_probs,
            num_graphs.final_probs,
            num_graphs.start_state,
            input,
            batch_sizes,
            input_lengths,
            num_graphs.num_states,
        )
        numerator_ok = numerator_ok.item()

        loss = -num_objf + den_objf

        if (loss - loss) != 0.0 or not denominator_ok or not numerator_ok:
            default_loss = 10
            input_grad = torch.zeros_like(input)
            logger.warning(
                f"Loss is {loss} and denominator computation "
                f"(if done) returned {denominator_ok} "
                f"and numerator computation returned {numerator_ok} "
                f", setting loss to {default_loss} per frame"
            )
            loss = torch.full_like(num_objf, default_loss * input_lengths.sum())
        else:
            num_grad = log_probs_grad.exp()
            input_grad -= num_grad

        ctx.save_for_backward(input_grad)
        return loss

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)

        return input_grad, None, None, None, None


@register_criterion("lattice_free_mmi", dataclass=LatticeFreeMMICriterionConfig)
class LatticeFreeMMICriterion(FairseqCriterion):
    def __init__(self, cfg: LatticeFreeMMICriterionConfig, task: FairseqTask):
        super().__init__(task)
        try:
            from pychain.graph import ChainGraph
            import simplefst
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` "
                "after entering espresso/tools"
            )

        self.sentence_avg = cfg.sentence_avg
        den_fst = simplefst.StdVectorFst.read(cfg.denominator_fst_path)
        self.den_graph = ChainGraph(den_fst, initial_mode="leaky", final_mode="ones")
        self.leaky_hmm_coefficient = cfg.leaky_hmm_coefficient
        self.xent_regularize = cfg.xent_regularization_coefficient
        self.output_l2_regularize = cfg.output_l2_regularization_coefficient

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
        try:
            from pychain.graph import ChainGraphBatch
            from pychain.loss import ChainFunction
        except ImportError:
            raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

        encoder_out = net_output["encoder_out"][0].transpose(0, 1)  # T x B x V -> B x T x V
        out_lengths = net_output["src_lengths"][0].long()  # B
        den_graphs = ChainGraphBatch(self.den_graph, sample["nsentences"])
        if self.xent_regularize > 0.0:
            den_objf = ChainFunction.apply(encoder_out, out_lengths, den_graphs, self.leaky_hmm_coefficient)
            num_objf = ChainFunction.apply(encoder_out, out_lengths, sample["target"])
            loss = - num_objf + den_objf  # negative log-probs
            nll_loss = loss.clone().detach()
            loss -= self.xent_regularize * num_objf
        else:
            # demonstrate another more "integrated" usage of the PyChain loss. it's equivalent to
            # the first three lines in the above "if" block, but also supports throwing away
            # batches with the NaN loss by setting their gradients to 0.
            loss = ChainLossFunction.apply(
                encoder_out, out_lengths, sample["target"], den_graphs, self.leaky_hmm_coefficient
            )
            nll_loss = loss.clone().detach()

        if self.output_l2_regularize > 0.0:
            encoder_padding_mask = (
                net_output["encoder_padding_mask"][0] if len(net_output["encoder_padding_mask"]) > 0
                else None
            )
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
