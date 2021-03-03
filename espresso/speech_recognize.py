#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Recognize pre-processed speech with a trained model.
"""

import ast
from itertools import chain
import logging
import math
import os
import sys

import numpy as np

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel

from espresso.models.external_language_model import MultiLevelLanguageModel
from espresso.models.tensorized_lookahead_language_model import TensorizedLookaheadLanguageModel
from espresso.tools import wer
from espresso.tools.utils import plot_attention, sequence_mask


def main(args):
    assert args.path is not None, "--path required for recognition!"
    assert not args.sampling or args.nbest == args.beam, \
        "--sampling requires --nbest to be equal to --beam"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("espresso.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    print_options_meaning_changes(args, logger)

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
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

    # Set dictionary
    dictionary = task.target_dictionary

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                utils.split_paths(args.lm_path),
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(f"Failed to load language model! Please make sure that the language model dict is the same "
                           f"as target dict and is located in the data dir ({args.data})")
            raise

        assert len(lms) == 1 or len(lms) == 2  # Multi-level LM expects two LMs
    else:
        lms = [None]

    for i, m in enumerate(lms):
        if m is None:
            continue
        if hasattr(m, "is_wordlm") and m.is_wordlm:
            # assume subword LM comes before word LM
            if i > 0 and isinstance(lms[i - 1], FairseqLanguageModel):
                lms[i - 1] = MultiLevelLanguageModel(
                    m, lms[i - 1],
                    subwordlm_weight=args.subwordlm_weight,
                    oov_penalty=args.oov_penalty,
                    open_vocab=not args.disable_open_vocab,
                )
                del lms[i]
                logger.info("LM fusion with Multi-level LM")
            else:
                lms[i] = TensorizedLookaheadLanguageModel(
                    m, dictionary,
                    oov_penalty=args.oov_penalty,
                    open_vocab=not args.disable_open_vocab,
                )
                logger.info("LM fusion with Look-ahead Word LM")
        else:
            assert isinstance(m, FairseqLanguageModel)
            logger.info("LM fusion with Subword LM")
    if args.lm_weight != 0.0:
        logger.info("using LM fusion with lm-weight={:.2f}".format(args.lm_weight))

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
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
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    if args.match_source_len:
        logger.warning(
            "The option match_source_len is not applicable to speech recognition. Ignoring it."
        )
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {
        "lm_model": lms[0],
        "lm_weight": args.lm_weight,
        "eos_factor": args.eos_factor,
    }
    args.score_reference = False  # not applicable for ASR
    temp_val = args.print_alignment
    args.print_alignment = False  # not applicable for ASR
    generator = task.build_generator(models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs)
    args.print_alignment = temp_val

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Generate and compute WER
    scorer = wer.Scorer(dictionary, wer_output_filter=args.wer_output_filter)
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample["target"][:, :args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens=prefix_tokens, constraints=constraints)
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        # obtain nonpad mask of encoder output to plot attentions
        if args.print_alignment:
            net_input = sample["net_input"]
            src_tokens = net_input["src_tokens"]
            output_lengths = models[0].encoder.output_lengths(net_input["src_lengths"])
            nonpad_idxs = sequence_mask(output_lengths, models[0].encoder.output_lengths(src_tokens.size(1)))

        for i in range(len(sample["id"])):
            has_target = sample["target"] is not None
            utt_id = sample["utt_id"][i]

            # Retrieve the original sentences
            if has_target:
                target_str = sample["target_raw_text"][i]
                if not args.quiet:
                    detok_target_str = decode_fn(target_str)
                    print("T-{}\t{}".format(utt_id, detok_target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_str = dictionary.string(
                    hypo["tokens"].int().cpu(),
                    bpe_symbol=None,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )  # not removing bpe at this point
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    print("H-{}\t{}\t{}".format(utt_id, detok_hypo_str, score), file=output_file)

                # Score and obtain attention only the top hypothesis
                if j == 0:
                    # src_len x tgt_len
                    attention = hypo["attention"][nonpad_idxs[i]].float().cpu() \
                        if args.print_alignment and hypo["attention"] is not None else None
                    if args.print_alignment and attention is not None:
                        save_dir = os.path.join(args.results_path, "attn_plots")
                        os.makedirs(save_dir, exist_ok=True)
                        plot_attention(attention, detok_hypo_str, utt_id, save_dir)
                    scorer.add_prediction(utt_id, hypo_str)
                    if has_target:
                        scorer.add_evaluation(utt_id, target_str, hypo_str)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if args.print_alignment:
        logger.info("Saved attention plots in " + save_dir)

    if has_target:
        scorer.add_ordered_utt_list(task.datasets[args.gen_subset].tgt.utt_ids)

    fn = "decoded_char_results.txt"
    with open(os.path.join(args.results_path, fn), "w", encoding="utf-8") as f:
        f.write(scorer.print_char_results())
        logger.info("Decoded char results saved as " + f.name)

    fn = "decoded_results.txt"
    with open(os.path.join(args.results_path, fn), "w", encoding="utf-8") as f:
        f.write(scorer.print_results())
        logger.info("Decoded results saved as " + f.name)

    if has_target:
        header = "Recognize {} with beam={}: ".format(args.gen_subset, args.beam)
        fn = "wer"
        with open(os.path.join(args.results_path, fn), "w", encoding="utf-8") as f:
            res = "WER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%".format(
                *(scorer.wer()))
            logger.info(header + res)
            f.write(res + "\n")
            logger.info("WER saved in " + f.name)

        fn = "cer"
        with open(os.path.join(args.results_path, fn), "w", encoding="utf-8") as f:
            res = "CER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%".format(
                *(scorer.cer()))
            logger.info(" " * len(header) + res)
            f.write(res + "\n")
            logger.info("CER saved in " + f.name)

        fn = "aligned_results.txt"
        with open(os.path.join(args.results_path, fn), "w", encoding="utf-8") as f:
            f.write(scorer.print_aligned_results())
            logger.info("Aligned results saved as " + f.name)
    return scorer


def print_options_meaning_changes(args, logger):
    """Options that have different meanings than those in the translation task
    are explained here.
    """
    logger.info("--max-tokens is the maximum number of input frames in a batch")
    if args.print_alignment:
        logger.info("--print-alignment has been set to plot attentions")


def cli_main():
    parser = options.get_generation_parser(default_task="speech_recognition_espresso")
    parser.add_argument("--eos-factor", default=None, type=float, metavar="F",
                        help="only consider emitting EOS if its score is no less "
                        "than the specified factor of the best candidate score")
    parser.add_argument("--subwordlm-weight", default=0.8, type=float, metavar="W",
                        help="subword LM weight relative to word LM. Only relevant "
                        "to MultiLevelLanguageModel as an external LM")
    parser.add_argument("--oov-penalty", default=1e-4, type=float,
                        help="oov penalty with the pretrained external LM")
    parser.add_argument("--disable-open-vocab", action="store_true",
                        help="whether open vocabulary mode is enabled with the "
                        "pretrained external LM")
    args = options.parse_args_and_arch(parser)
    assert args.results_path is not None, "please specify --results-path"
    main(args)


if __name__ == "__main__":
    cli_main()
