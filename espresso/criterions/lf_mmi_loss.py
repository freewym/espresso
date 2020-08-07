# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics


@register_criterion("lattice_free_mmi")
class LatticeFreeMMICriterion(FairseqCriterion):

    def __init__(
        self, task, sentence_avg, denominator_fst_path,
        leaky_hmm_coefficient, xent_regularize, l2_regularize,
    ):
        super().__init__(task)
        try:
            from pychain.graph import ChainGraph
            import simplefst
        except ImportError:
            raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

        self.sentence_avg = sentence_avg
        den_fst = simplefst.StdVectorFst.read(denominator_fst_path)
        self.den_graph = ChainGraph(den_fst, initial_mode="leaky", final_mode="ones")
        self.leaky_hmm_coefficient = leaky_hmm_coefficient
        self.xent_regularize = xent_regularize
        self.l2_regularize = l2_regularize

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        FairseqCriterion.add_args(parser)
        parser.add_argument("--denominator-fst-path", type=str, metavar="FILE",
                            help="path to the denominator fst file")
        parser.add_argument("--leaky-hmm-coefficient", default=1.0e-05, type=float, metavar="F",
                            help="leaky-hmm coefficient for the denominator")
        parser.add_argument("--xent-regularization-coefficient", default=0.0,
                            type=float, metavar="F", dest="xent_regularize",
                            help="cross-entropy regularization coefficient")
        parser.add_argument("--output-l2-regularization-coefficient", default=0.0,
                            type=float, metavar="F", dest="output_l2_regularize",
                            help="L2 regularization coefficient for the network's output")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(net_output, sample, reduce=reduce)

        sample_size = sample["target"].batch_size if self.sentence_avg else sample["ntokens"]
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
            from pychain.loss import ChainFunction, ChainLossFunction
        except ImportError:
            raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

        encoder_out = net_output.encoder_out.transpose(0, 1)  # T x B x V -> B x T x V
        out_lengths = net_output.src_lengths.long()  # B
        if self.xent_regularize > 0.0:
            den_graphs = ChainGraphBatch(self.den_graph, sample["nsentences"])
            den_objf = ChainFunction.apply(encoder_out, out_lengths, den_graphs, self.leaky_hmm_coefficient)
            num_objf = ChainFunction.apply(encoder_out, out_lengths, sample["target"])
            loss = - num_objf + den_objf  # negative log-probs
            nll_loss = loss.clone().detach()
            loss -= self.xent_regularize * num_objf
        else:
            # demonstrate another more "integrated" usage of the PyChain loss. it's equivalent to the
            # first four lines in the previous "if" statement, but also supports throwing away
            # batches with the NaN loss by setting their gradients to 0.
            loss = ChainLossFunction.apply(
                encoder_out, out_lengths, sample["target"], self.den_graph, self.leaky_hmm_coefficient
            )
            nll_loss = loss.clone().detach()

        if self.output_l2_regularize > 0.0:
            encoder_padding_mask = net_output.encoder_padding_mask
            encoder_out_squared = encoder_out.pow(2.0)
            if encoder_padding_mask is not None:
                pad_mask = encoder_padding_mask.transpose(0, 1).unsqueeze(-1)  # T x B -> B x T x 1
                encoder_out_squared.masked_fill_(pad_mask, 0.0)
            loss += 0.5 * self.output_l2_regularize * encoder_out_squared.sum()

        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=7)
        metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=7)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, round=4))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
