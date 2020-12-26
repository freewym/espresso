#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys

import torch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("mobvoihotwords.generate_graphs")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate graphs for training"
    )
    # fmt: off
    parser.add_argument("--hmm-paths", nargs="+", help="list of HMM paths (in openfst text format)", required=True)
    parser.add_argument("--L-path", type=str, help="path to L fst (in openfst text formet)", required=True)
    parser.add_argument("--phone-lm-fsa-path", type=str, help="path to the phone LM fsa (in openfst text format)", required=True)
    parser.add_argument("--out-dir", type=str, default="data", help="directory to save output graphs")
    # fmt: on

    return parser


def main(args):
    try:
        import k2
    except ImportError:
        raise ImportError("Please install k2 by `pip install k2`")

    hmms = []
    for hmm in args.hmm_paths:
        with open(hmm, "r", encoding="utf-8") as f:
            hmms.append(k2.Fsa.from_openfst(f.read(), acceptor=False))
        hmms[-1] = k2.arc_sort(hmms[-1])
    hmm_vec = k2.create_fsa_vec(hmms)
    # temporarily +1 to make label 0 of HMMs (means "blank") be treated as normal label
    hmm_vec.labels = torch.where(hmm_vec.labels >= 0, hmm_vec.labels + 1, hmm_vec.labels)
    H = k2.connect(k2.remove_epsilons_iterative_tropical(k2.closure(k2.union(hmm_vec))))
    H.labels = torch.where(H.labels > 0, H.labels - 1, H.labels)  # restore the label indices
    H_inv = k2.arc_sort(k2.invert(H))
    #save_path = os.path.join(args.out_dir, "Hinv.pt")
    #torch.save(H_inv.as_dict(), save_path)
    #logger.info(f"saved H_inv as {save_path}")

    with open(args.L_path, "r", encoding="utf-8") as f:
        L = k2.arc_sort(k2.Fsa.from_openfst(f.read(), acceptor=False))

    with open(args.phone_lm_fsa_path, "r", encoding="utf-8") as f:
        phone_lm = k2.arc_sort(k2.Fsa.from_openfst(f.read(), acceptor=True))

    # emulate composition
    #L_clone = L.clone()
    #if hasattr(L, "aux_labels"):
    #    L.temp_labels = L.aux_labels
    #    del L.aux_labels
    #HL = k2.invert(k2.connect(k2.intersect(L, H_inv)))
    #if hasattr(HL, "temp_labels"):
    #    HL.aux_labels = HL.temp_labels
    #    del HL.temp_labels
    #HL_inv = k2.arc_sort(k2.invert(HL))
    HL_inv = k2.arc_sort(k2.connect(k2.compose(L.invert(), H_inv)))
    #print(k2.is_rand_equivalent(HL_inv,HL_inv_new,log_semiring=True))
    save_path = os.path.join(args.out_dir, "HLinv.pt")
    torch.save(HL_inv.as_dict(), save_path)
    logger.info(f"saved the HL_inv fst as {save_path}")

    den_graph = k2.arc_sort(k2.invert(k2.connect(k2.intersect(H_inv, phone_lm))))
    del den_graph.aux_labels
    save_path = os.path.join(args.out_dir, "denominator.pt")
    torch.save(den_graph.as_dict(), save_path)
    logger.info(f"saved the denominator graph as {save_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
