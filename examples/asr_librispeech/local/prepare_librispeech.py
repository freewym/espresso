#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import os
import sys

from tqdm import tqdm

try:
    from torchaudio.datasets import LIBRISPEECH
except ImportError:
    raise ImportError("Please install torchaudio with: pip install torchaudio")


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech corpus"
    )
    # fmt: off
    parser.add_argument("corpus_root", type=str, help="path to the LibriSpeech root directory")
    parser.add_argument("output_root", type=str, help="path to the root directory of generated data")
    parser.add_argument("--folder-in-archive", type=str, default="LibriSpeech", help="the top-level directory of the dataset")
    parser.add_argument("--download", action="store_true", help="download the dataset if it is not found at corpus root path")
    # fmt: on

    return parser


def main(args):
    corpus_root = Path(args.corpus_root).absolute()
    output_root = Path(args.output_root).absolute()
    corpus_root.mkdir(exist_ok=True)
    output_root.mkdir(exist_ok=True)
    for split in SPLITS:
        logger.info(f"Preparing data for split {split}...")
        output_dir = output_root / split.replace("-", "_")
        output_dir.mkdir(exist_ok=True)
        wave_file = output_dir / "wav.txt"
        text_file = output_dir / "text.txt"
        if os.path.exists(wave_file) and os.path.exists(text_file):
            logger.info(f"Both {wave_file} and {text_file} exist, skip regenerating")
            continue
        dataset = LIBRISPEECH(
            corpus_root.as_posix(), url=split, folder_in_archive=args.folder_in_archive, download=args.download
        )
        with open(wave_file, "w", encoding="utf-8") as wave_f, open(text_file, "w", encoding="utf-8") as text_f:
            for data_tuple in tqdm(dataset):
                if len(data_tuple) == 6:  # torchaudio=0.7.0
                    # (waveform, sample_rate, text, speaker_id, chapter_id, utterance_idx)
                    text, speaker_id, chapter_id, utterance_idx = data_tuple[2], data_tuple[3], data_tuple[4], data_tuple[5]
                else:  # torchaudio>=0.8.0
                    # (waveform, sample_rate, orignal_text, normalized_text, speaker_id, chapter_id, utterance_idx)
                    assert len(data_tuple) == 7
                    text, speaker_id, chapter_id, utterance_idx = data_tuple[3], data_tuple[4], data_tuple[5], data_tuple[6]

                utterance_idx = str(utterance_idx).zfill(4)
                utterance_id = f"{speaker_id}-{chapter_id}-{utterance_idx}"
                utterance_path = os.path.join(
                    corpus_root.as_posix(), args.folder_in_archive, split, str(speaker_id), str(chapter_id), utterance_id
                )
                
                print(f"{utterance_id} {utterance_path}.flac", file=wave_f)
                print(f"{utterance_id} {text}", file=text_f)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
