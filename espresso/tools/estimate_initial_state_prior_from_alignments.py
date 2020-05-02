#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys

import numpy as np

try:
    import kaldi_io
except ImportError:
    raise ImportError('Please install kaldi_io with: pip install kaldi_io')


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.estimate_initial_state_prior_from_alignments")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Obtain initial state prior from alignments")
    # fmt: off
    parser.add_argument("--alignment-files", nargs="+", required=True,
                        help="path(s) to alignment file(s)")
    parser.add_argument("--prior-dim", required=True, type=int,
                        help="state prior dimension, i.e., the number of states")
    parser.add_argument("--prior-floor", type=float, default=5.0e-6,
                        help="floor for the state prior")
    parser.add_argument("--output", required=True, type=str,
                        help="output path")
    # fmt: on
    return parser


def main(args):
    assert args.prior_floor > 0.0 and args.prior_floor < 1.0
    prior = np.zeros((args.prior_dim,), dtype=np.int32)
    for path in args.alignment_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                _, rxfile = line.strip().split(None, 1)
                try:
                    ali = kaldi_io.read_vec_int(rxfile)
                except Exception:
                    raise Exception("failed to read int vector {}.".format(rxfile))
                assert ali is not None and isinstance(ali, np.ndarray)
                for id in ali:
                    prior[id] += 1
    prior = np.maximum(prior / float(np.sum(prior)), args.prior_floor)  # normalize and floor
    prior = prior / float(np.sum(prior))  # normalize again
    kaldi_io.write_vec_flt(args.output, prior)

    logger.info("Saved the initial state prior estimate in {}".format(args.output))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
