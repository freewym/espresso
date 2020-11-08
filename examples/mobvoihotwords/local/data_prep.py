#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from typing import List
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np

from fairseq.data.data_utils import numpy_seed


try:
    # TODO use pip install once it's available
    from espresso.tools.lhotse.lhotse import (
        CutSet, Mfcc, MfccConfig, LilcomFilesWriter, RecordingSet, SupervisionSet
    )
    from espresso.tools.lhotse.lhotse.augmentation import SoxEffectTransform, RandomValue
    from espresso.tools.lhotse.lhotse.manipulation import combine
    from espresso.tools.lhotse.lhotse.recipes.mobvoihotwords import download_and_untar, prepare_mobvoihotwords
except ImportError:
    raise ImportError("Please install Lhotse by `make lhotse` after entering espresso/tools")


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("mobvoihotwords.data_prep")


def get_parser():
    parser = argparse.ArgumentParser(
        description="data preparation for the MobvoiHotwords corpus"
    )
    # fmt: off
    parser.add_argument("--data-dir", default="data", type=str, help="data directory")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument(
        "--num-workers", default=1, type=int, help="number of workers for features extraction"
    )
    parser.add_argument(
        "--max-remaining-duration", default=0.3, type=float,
        help="not split if the left-over duration is less than this many seconds"
    )
    parser.add_argument(
        "--overlap-duration", default=0.3, type=float,
        help="overlap between adjacent segments while splitting negative recordings"
    )
    # fmt: on

    return parser


def main(args):
    root_dir = Path(args.data_dir)
    corpus_dir = root_dir / "MobvoiHotwords"
    output_dir = root_dir

    logger.info(f"Download and extract the corpus")
    download_and_untar(root_dir)

    logger.info(f"Prepare the manifests")
    partitions = ["train", "dev", "test"]
    if all(
        (output_dir / f"{key}_{part}.json").is_file()
        for key in ["recordings", "supervisions"] for part in partitions
    ):
        logger.info(f"All the manifests files are found in {output_dir}. Load from them directly")
        mobvoihotwords_manifests = defaultdict(dict)
        for part in partitions:
            mobvoihotwords_manifests[part] = {
                "recordings": RecordingSet.from_json(output_dir / f"recordings_{part}.json"),
                "supervisions": SupervisionSet.from_json(output_dir / f"supervisions_{part}.json")
            }
    else:
        logger.info("It may take long time")
        mobvoihotwords_manifests = prepare_mobvoihotwords(corpus_dir, output_dir)
    logger.info(
        "train/dev/test size: {}/{}/{}".format(
            len(mobvoihotwords_manifests["train"]["recordings"]),
            len(mobvoihotwords_manifests["dev"]["recordings"]),
            len(mobvoihotwords_manifests["test"]["recordings"])
        )
    )

    # Data augmentation
    np.random.seed(args.seed)
    # equivalent to Kaldi's mfcc_hires config
    mfcc = Mfcc(config=MfccConfig(num_mel_bins=40, num_ceps=40, low_freq=20, high_freq=-400))
    for part, manifests in mobvoihotwords_manifests.items():
        cut_set = CutSet.from_manifests(
            recordings=manifests["recordings"],
            supervisions=manifests["supervisions"],
        )
        sampling_rate = next(iter(cut_set)).sampling_rate
        with ProcessPoolExecutor(args.num_workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            if part == "train":
                # split negative recordings into smaller chunks with lengths sampled from
                # length distribution of positive recordings
                logger.info(f"Split negative recordings in '{part}' set")
                pos_durs = get_positive_durations(manifests["supervisions"])
                with numpy_seed(args.seed):
                    cut_set = keep_positives_and_split_negatives(
                        cut_set,
                        pos_durs,
                        max_remaining_duration=args.max_remaining_duration,
                        overlap_duration=args.overlap_duration,
                    )

                # "clean" set
                logger.info(f"Extract features for '{part}' set")
                json_path = output_dir / f"cuts_{part}_clean.json.gz"
                if json_path.is_file():
                    logger.info(f"{json_path} exists, skip the extraction (remove it if you want to re-generate it)")
                    cut_set_clean = CutSet.from_json(json_path)
                else:
                    with LilcomFilesWriter(f"{output_dir}/feats_{part}_clean") as storage:
                        cut_set_clean = cut_set.compute_and_store_features(
                            extractor=mfcc,
                            storage=storage,
                            augment_fn=None,
                            executor=ex,
                        )
                    cut_set_clean.to_json(json_path)

                # augmented with reverberation
                logger.info(f"Extract features for '{part}' set with reverberation")
                json_path = output_dir / f"cuts_{part}_rev.json.gz"
                if json_path.is_file():
                    logger.info(f"{json_path} exists, skip the extraction (remove it if you want to re-generate it)")
                    cut_set_rev = CutSet.from_json(json_path)
                else:
                    augment_fn = SoxEffectTransform(effects=reverb(sampling_rate=sampling_rate))
                    with LilcomFilesWriter(f"{output_dir}/feats_{part}_rev") as storage:
                        with numpy_seed(args.seed):
                            cut_set_rev = cut_set.compute_and_store_features(
                                extractor=mfcc,
                                storage=storage,
                                augment_fn=augment_fn,
                                executor=ex,
                            )
                    cut_set_rev = CutSet.from_cuts(
                        cut.with_id("rev-" + cut.id) for cut in cut_set_rev
                    )
                    cut_set_rev.to_json(json_path)

                # augmented with speed perturbation
                logger.info(f"Extract features for '{part}' set with speed perturbation")
                json_path = output_dir / f"cuts_{part}_sp1.1.json.gz"
                if json_path.is_file():
                    logger.info(f"{json_path} exists, skip the extraction (remove it if you want to re-generate it)")
                    cut_set_sp1p1 = CutSet.from_json(json_path)
                else:
                    augment_fn = SoxEffectTransform(effects=speed(sampling_rate=sampling_rate, factor=1.1))
                    with LilcomFilesWriter(f"{output_dir}/feats_{part}_sp1.1") as storage:
                        cut_set_sp1p1 = cut_set.compute_and_store_features(
                            extractor=mfcc,
                            storage=storage,
                            augment_fn=augment_fn,
                            executor=ex,
                        )
                    cut_set_sp1p1 = CutSet.from_cuts(
                        cut.with_id("sp1.1-" + cut.id) for cut in cut_set_sp1p1
                    )
                    cut_set_sp1p1.to_json(json_path)
                json_path = output_dir / f"cuts_{part}_sp0.9.json.gz"
                if json_path.is_file():
                    logger.info(f"{json_path} exists, skip the extraction")
                    cut_set_sp1p1 = CutSet.from_json(json_path)
                else:
                    augment_fn = SoxEffectTransform(effects=speed(sampling_rate=sampling_rate, factor=0.9))
                    with LilcomFilesWriter(f"{output_dir}/feats_{part}_sp0.9") as storage:
                        cut_set_sp0p9 = cut_set.compute_and_store_features(
                            extractor=mfcc,
                            storage=storage,
                            augment_fn=augment_fn,
                            executor=ex,
                        )
                    cut_set_sp0p9 = CutSet.from_cuts(
                        cut.with_id("sp0.9-" + cut.id) for cut in cut_set_sp0p9
                    )
                    cut_set_sp0p9.to_json(json_path)

                # combine the clean and augmented sets together
                logger.info(f"Combine all the features above")
                cut_set = combine(
                    cut_set_clean, cut_set_rev, cut_set_sp1p1, cut_set_sp0p9
                )
            else:  # no augmentations for dev and test sets
                logger.info(f"extract features for '{part}' set")
                json_path = output_dir / f"cuts_{part}.json.gz"
                if json_path.is_file():
                    logger.info(f"{json_path} exists, skip the extraction (remove it if you want to re-generate it)")
                    cut_set = CutSet.from_json(json_path)
                else:
                    with LilcomFilesWriter(f"{output_dir}/feats_{part}") as storage:
                        cut_set = cut_set.compute_and_store_features(
                            extractor=mfcc,
                            storage=storage,
                            augmenter=None,
                            executor=ex,
                        )

            mobvoihotwords_manifests[part]["cuts"] = cut_set
            cut_set.to_json(output_dir / f"cuts_{part}.json.gz")


def get_positive_durations(sup_set: SupervisionSet) -> List[float]:
    """
    Get duration values of all positive recordings, assuming Supervison.text is
    "FREETEXT" for all negative recordings, and SupervisionSegment.duration
    equals to the corresponding Recording.duration.
    """
    return [sup.duration for sup in sup_set.filter(lambda seg: seg.text != "FREETEXT")]


def keep_positives_and_split_negatives(
    cut_set: CutSet,
    durations: List[float],
    max_remaining_duration: float = 0.3,
    overlap_duration: float = 0.3,
) -> CutSet:
    """
    Returns a new CutSet where all the positives are directly taken from the original
    input cut set, and the negatives are obtained by splitting original negatives
    into shorter chunks of random lengths drawn from the given length distribution
    (here it is the empirical distribution of the positive recordings), There can
    be overlap between chunks.

    Args:
        cut_set (CutSet): original input cut set 
        durations (list[float]): list of durations to sample from
        max_remaining_duration (float, optional): not split if the left-over
            duration is less than this many seconds (default: 0.3).
        overlap_duration (float, optional): overlap between adjacent segments
            (default: None)

    Returns:
        CutSet: a new cut set after split
    """
    assert max_remaining_duration >= 0.0 and overlap_duration >= 0.0
    new_cuts = []
    for cut in cut_set:
        assert len(cut.supervisions) == 1
        if cut.supervisions[0].text != "FREETEXT":  # keep the positive as it is
            new_cuts.append(cut)
        else:
            this_offset = cut.start
            this_offset_relative = this_offset - cut.start
            remaining_duration = cut.duration
            this_dur = durations[np.random.randint(len(durations))]
            while remaining_duration > this_dur + max_remaining_duration:
                new_cut = cut.truncate(
                    offset=this_offset_relative, duration=this_dur, preserve_id=True
                )
                new_cut = new_cut.with_id(
                    "{id}-{s:07d}-{e:07d}".format(
                        id=new_cut.id,
                        s=int(round(100 * this_offset_relative)),
                        e=int(round(100 * (this_offset_relative + this_dur)))
                    )
                )
                new_cuts.append(new_cut)
                this_offset += this_dur - overlap_duration
                this_offset_relative = this_offset - cut.start
                remaining_duration -= this_dur - overlap_duration
                this_dur = durations[np.random.randint(len(durations))]

            new_cut = cut.truncate(offset=this_offset_relative, preserve_id=True)
            new_cut = new_cut.with_id(
                "{id}-{s:07d}-{e:07d}".format(
                    id=new_cut.id,
                    s=int(round(100 * this_offset_relative)),
                    e=int(round(100 * cut.duration))
                )
            )
            new_cuts.append(new_cut)

    return CutSet.from_cuts(new_cuts)


def reverb(sampling_rate: int) -> List[List[str]]:
    return [
        ["reverb", 50, 50, RandomValue(1, 30)],
        ["remix", "-"],  # Merge all channels (reverb changes mono to stereo)
    ]


def speed(sampling_rate: int, factor: float) -> List[List[str]]:
    return [
        # speed perturbation with a factor
        ["speed", factor],
        ["rate", sampling_rate],  # Resample back to the original sampling rate (speed changes it)
    ]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
