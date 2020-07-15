#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dump frame-level posteriors (intepreted as log probabilities) with a trained model
for decoding with Kaldi.
"""

import logging
import sys

import numpy as np

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


def main(args):
    assert args.path is not None, "--path required for decoding!"
    return _main(args, sys.stderr)


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger("espresso.dump_posteriors")

    print_options_meaning_changes(args, logger)

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset split
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Load state prior for cross-entropy trained systems decoding
    if args.state_prior_file is not None:
        prior = torch.from_numpy(kaldi_io.read_vec_flt(args.state_prior_file))
    else:
        prior = []

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        if isinstance(prior, list) and getattr(model, "state_prior", None) is not None:
            prior.append(model.state_prior.unsqueeze(0))

    if isinstance(prior, list) and len(prior) > 0:
        prior = torch.cat(prior, 0).mean(0)  # average priors across models
        prior = prior / prior.sum()  # re-normalize
    elif isinstance(prior, list):
        prior = None

    if prior is not None:
        if args.fp16:
            prior = prior.half()
        if use_cuda:
            prior = prior.cuda()
        log_prior = prior.log()
    else:
        log_prior = None

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() if hasattr(model, "encoder")
              else (None, model.max_positions()) for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(models, args)

    # Generate and dump
    num_sentences = 0
    chunk_width = getattr(task, "chunk_width", None)
    lprobs_wspecifier = "ark:| copy-matrix ark:- ark:-"
    with kaldi_io.open_or_fd(lprobs_wspecifier, "wb") as f:
        if chunk_width is None:  # normal dumping (i.e., no chunking)
            for sample in progress:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if "net_input" not in sample:
                    continue

                gen_timer.start()
                lprobs, padding_mask = task.inference_step(generator, models, sample)
                if log_prior is not None:
                    assert lprobs.size(-1) == log_prior.size(0)
                    lprobs = lprobs - log_prior
                out_lengths = (~padding_mask).long().sum(dim=1).cpu() if padding_mask is not None else None
                num_processed_frames = sample["ntokens"]
                gen_timer.stop(num_processed_frames)
                num_sentences += sample["nsentences"]

                if out_lengths is not None:
                    for i in range(sample["nsentences"]):
                        length = out_lengths[i]
                        kaldi_io.write_mat(f, lprobs[i, :length, :].cpu().numpy(), key=sample["utt_id"][i])
                else:
                    for i in range(sample["nsentences"]):
                        kaldi_io.write_mat(f, lprobs[i, :, :].cpu().numpy(), key=sample["utt_id"][i])
        else:  # dumping chunks within the same utterance from left to right
            for sample in progress:  # sample is actually a list of batches
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                utt_id = sample[0]["utt_id"]
                id = sample[0]["id"]
                whole_lprobs = None
                for i, chunk_sample in enumerate(sample):
                    if "net_input" not in chunk_sample:
                        continue

                    assert chunk_sample["utt_id"] == utt_id and (chunk_sample["id"] == id).all()
                    gen_timer.start()
                    lprobs, _ = task.inference_step(generator, models, chunk_sample)
                    if log_prior is not None:
                        assert lprobs.size(-1) == log_prior.size(0)
                        lprobs = lprobs - log_prior
                    if whole_lprobs is None:
                        whole_lprobs = lprobs.cpu()
                    else:
                        whole_lprobs = torch.cat((whole_lprobs, lprobs.cpu()), 1)
                    num_processed_frames = chunk_sample["ntokens"]
                    gen_timer.stop(num_processed_frames)

                    if i == len(sample) - 1:
                        num_sentences += len(utt_id)
                        for j in range(len(utt_id)):
                            truncated_length = models[0].output_lengths(
                                task.dataset(args.gen_subset).src_sizes[id[j]]
                            )  # length is after possible subsampling by the model
                            mat = whole_lprobs[j, :truncated_length, :]
                            kaldi_io.write_mat(f, mat.numpy(), key=utt_id[j])

    logger.info("Dumped {} utterances ({} frames) in {:.1f}s ({:.2f} sentences/s, {:.2f} frames/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    return


def print_options_meaning_changes(args, logger):
    """Options that have different meanings than those in the translation task
    are explained here.
    """
    logger.info("| --max-tokens is the maximum number of input frames in a batch")


def cli_main():
    parser = options.get_generation_parser(default_task="speech_recognition_hybrid")
    parser.add_argument("--apply-log-softmax", action="store_true",
                        help="Apply log-softmax to the neural network outputs for some "
                        "systems, e.g., Xent. Otherwise use the raw outputs")
    parser.add_argument("--state-prior-file", default=None, type=str, metavar="FILE",
                        help="state prior file. If provided, use this file instead of "
                        "that from the checkpoint")
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
