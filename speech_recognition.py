#!/usr/bin/env python3 -u
# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Recognize pre-processed speech with a trained model.
"""

import os

import torch

from fairseq import wer, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter
from fairseq.speech_recognizer import SpeechRecognizer
from fairseq.utils import import_user_module
from speech_tools.utils import plot_attention


def main(args):
    assert args.path is not None, '--path required for recognition!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset split
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset,
        len(task.dataset(args.gen_subset))))

    # Set dictionary
    dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    if args.match_source_len:
        print('| The option match_source_len is not applicable to '
            'speech recognition. Ignoring it.')
    gen_timer = StopwatchMeter()
    recognizer = SpeechRecognizer(
        models, dict, beam_size=args.beam, minlen=args.min_len,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk,
        sampling_temperature=args.sampling_temperature,
        diverse_beam_groups=args.diverse_beam_groups,
        diverse_beam_strength=args.diverse_beam_strength,
        match_source_len=False, no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    if use_cuda:
        recognizer.cuda()

    # Generate and compute WER
    scorer = wer.Scorer(dict)
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        recognitions = recognizer.generate_batched_itr(
            t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
        )

        sps_meter = TimeMeter()
        for sample_id, utt_id, target_tokens, hypos in recognitions:
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Retrieve the original sentences
            if has_target:
                target_str = task.dataset(args.gen_subset).tgt.get_original_tokens(sample_id)
                if not args.quiet:
                    print('T-{}\t{}'.format(utt_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_str = dict.string(hypo['tokens'].int().cpu(), remove_bpe)

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(utt_id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        utt_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))

                # Score and obtain attention only the top hypothesis
                if i == 0:
                    # src_len x tgt_len
                    attention = hypo['attention'].float().cpu() \
                        if hypo['attention'] is not None else None
                    if attention is not None and args.print_alignment:
                        plot_attention(attention, hypo_str, utt_id,
                            os.path.join(args.path, 'attn_plots'))
                        print('| Saved attention plots in ' + \
                            os.path.join(args.path, 'attn_plots'))
                    scorer.add_prediction(utt_id, hypo_str)
                    if has_target:
                        scorer.add_evaluation(utt_id, target_str, hypo_str)

            num_sentences += 1

    print('| Recognized {} utterances in {:.1f}s ({:.2f} utterances/s)'.format(
        num_sentences, gen_timer.sum, 1. / gen_timer.avg))

    scorer.add_ordered_utt_list(*args.test_text_files)

    fn = 'results.txt'
    with open(os.path.join(args.path, fn), 'w', encoding='utf-8') as f:
        f.write(scorer.print_results())
        print('| Decoded results saved as ' + f.name)

    if has_target:
        fn = 'wer'
        with open(os.path.join(args.path, fn), 'w', encoding='utf-8') as f:
            res = 'WER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%'.format(
                *(scorer.wer()))
            print('| Recognize {} with beam={}: '.format(args.gen_subset, args.beam) + res)
            f.write(res + '\n')
            print('| WER saved in ' + f.name)

        fn = 'cer'
        with open(os.path.join(args.path, fn), 'w', encoding='utf-8') as f:
            res = 'CER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%'.format(
                *(scorer.cer()))
            print('|                            ' + res)
            f.write(res + '\n')
            print('| CER saved in ' + f.name)

        fn = 'aligned_results.txt'
        with open(os.path.join(args.path, fn), 'w', encoding='utf-8') as f:
            f.write(scorer.print_aligned_results())
            print('| Aligned results saved as ' + f.name)


def print_options_meaning_changes(args):
    """Options that have different meanings than those in the translation task
    are explained here.
    """
    print('| --max-tokens is the maximum number of input frames in a batch')
    if args.print_alignment:
        print('| --print-alignment is set to plot attentions')


if __name__ == '__main__':
    parser = options.get_generation_parser(default_task='speech_recognition')
    args = options.parse_args_and_arch(parser)
    print_options_meaning_changes(args)
    main(args)
