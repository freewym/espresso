#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import re
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO
from subprocess import PIPE, run
from typing import Tuple

import numpy as np
from tqdm import tqdm

from espresso.tools.utils import get_torchaudio_fbank_or_mfcc
from fairseq.data.audio.audio_utils import get_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def get_parser():
    parser = argparse.ArgumentParser(description="Compute global CMVN stats")
    # fmt: off
    parser.add_argument(
        "file", type=str, nargs="?",
        help="input file where each line has the format '<utt-id> <file-path>' or '<utt-id> <command>'"
    )
    parser.add_argument("output_dir", type=str, help="path to the output file")
    parser.add_argument(
        "--feature-type", type=str, default="fbank", choices=["fbank", "mfcc"],
        help="feature type, currently support fbank or mfcc"
    )
    parser.add_argument("--feat-dim", type=int, default=80, help="feature dimension. Normally 80 for fbank and 40 for mfcc")
    parser.add_argument(
        "--max-num-utts", type=int, default=None,
        help="max number of utterances to compute cmvn stats (first N in the input file)"
    )
    parser.add_argument("--num-workers", type=int, default=20, help="number of workers for multi-thread processing")
    # fmt: on

    return parser


def process(
    line: str, n_bins: int = 80, feature_type: str = "fbank"
) -> Tuple[np.ndarray, np.ndarray, int]:
    _, rxfile = line.rstrip().split(None, 1)
    if re.search(r"\|$", rxfile) is not None:  # from a command
        source = BytesIO(run(rxfile[:-1], shell=True, stdout=PIPE).stdout)
    else:  # from a raw waveform file
        source = rxfile
    waveform, sample_rate = get_waveform(source, normalization=False, always_2d=True)
    feat = get_torchaudio_fbank_or_mfcc(
        waveform, sample_rate, n_bins=n_bins, feature_type=feature_type
    )
    cur_sum = feat.sum(axis=0)
    cur_frames = feat.shape[0]
    cur_unnorm_var = np.var(feat, axis=0) * cur_frames
    return cur_sum, cur_unnorm_var, cur_frames


def main(args):
    if args.max_num_utts is not None:
        logger.info(
            f"Computing {args.feature_type} global CMVN stats from first {args.max_num_utts} utterances in {args.file}"
        )
    else:
        logger.info(
            f"Computing {args.feature_type} global CMVN stats from utterances in {args.file}"
        )

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        with (open(args.file, "r", encoding="utf-8") if args.file else sys.stdin) as f:
            for i, line in enumerate(f):
                if args.max_num_utts is not None and i == args.max_num_utts:
                    break
                futures.append(
                    ex.submit(process, line, args.feat_dim, args.feature_type)
                )

        # implementation based on Lhotse: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/base.py#L657
        total_sum = np.zeros((args.feat_dim,), dtype=np.float64)
        total_unnorm_var = np.zeros((args.feat_dim,), dtype=np.float64)
        total_frames = 0
        for future in tqdm(futures, desc="Processing", leave=False):
            cur_sum, cur_unnorm_var, cur_frames = future.result()
            updated_total_sum = total_sum + cur_sum
            updated_total_frames = total_frames + cur_frames
            total_over_cur_frames = total_frames / cur_frames
            if total_frames > 0:
                total_unnorm_var = (
                    total_unnorm_var
                    + cur_unnorm_var
                    + total_over_cur_frames
                    / updated_total_frames
                    * (total_sum / total_over_cur_frames - cur_sum) ** 2
                )
            else:
                total_unnorm_var = cur_unnorm_var
            total_sum = updated_total_sum
            total_frames = updated_total_frames

    stats = {
        "mean": total_sum / total_frames,
        "std": np.sqrt(total_unnorm_var / total_frames),
    }
    with open(os.path.join(args.output_dir, "gcmvn.npz"), "wb") as f:
        np.savez(f, mean=stats["mean"], std=stats["std"])
        logger.info(f"Saved CMVN stats file as {f.name}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
