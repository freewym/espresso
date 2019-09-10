#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Recognize pre-processed speech with a trained model.
"""

import os

import torch

from fairseq import wer, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from fairseq.models.external_language_model import MultiLevelLanguageModel
from fairseq.models.tensorized_lookahead_language_model import TensorizedLookaheadLanguageModel
from fairseq.utils import import_user_module
from speech_tools.utils import plot_attention


def main(args):
    assert args.path is not None, '--path required for recognition!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset split
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionary
    dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    for i, m in enumerate(models):
        if hasattr(m, 'is_wordlm') and m.is_wordlm:
            # assume subword LM comes before word LM
            if isinstance(models[i - 1], FairseqLanguageModel):
                models[i-1] = MultiLevelLanguageModel(m, models[i-1],
                    subwordlm_weight=args.subwordlm_weight,
                    oov_penalty=args.oov_penalty,
                    open_vocab=not args.disable_open_vocab)
                del models[i]
                print('| LM fusion with Multi-level LM')
            else:
                models[i] = TensorizedLookaheadLanguageModel(m, dict,
                    oov_penalty=args.oov_penalty,
                    open_vocab=not args.disable_open_vocab)
                print('| LM fusion with Look-ahead Word LM')
        # assume subword LM comes after E2E models
        elif i == len(models) - 1 and isinstance(m, FairseqLanguageModel):
            print('| LM fusion with Subword LM')
    if args.lm_weight != 0.0:
        print('| using LM fusion with lm-weight={:.2f}'.format(args.lm_weight))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment or args.coverage_weight > 0.,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() if hasattr(model, 'encoder') \
            else (None, model.max_positions()) for model in models]
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
    generator = task.build_generator(args)

    # Generate and compute WER
    scorer = wer.Scorer(dict, wer_output_filter=args.wer_output_filter)
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens,
                lm_weight=args.lm_weight)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None
                utt_id = sample['utt_id'][i]

                # Retrieve the original sentences
                if has_target:
                    target_str = task.dataset(args.gen_subset).tgt.get_original_tokens(sample_id)
                    if not args.quiet:
                        target_sent = dict.tokens_to_sentence(target_str,
                            use_unk_sym=False, bpe_symbol=args.remove_bpe)
                        print('T-{}\t{}'.format(utt_id, target_sent))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_str = dict.string(hypo['tokens'].int().cpu()) # not removing bpe at this point
                    if not args.quiet or i == 0:
                        hypo_sent = dict.tokens_to_sentence(hypo_str, bpe_symbol=args.remove_bpe)

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(utt_id, hypo_sent, hypo['score']))

                    # Score and obtain attention only the top hypothesis
                    if j == 0:
                        # src_len x tgt_len
                        attention = hypo['attention'].float().cpu() \
                            if hypo['attention'] is not None else None
                        if args.print_alignment and attention is not None:
                            save_dir = os.path.join(args.results_path, 'attn_plots')
                            os.makedirs(save_dir, exist_ok=True)
                            plot_attention(attention, hypo_sent, utt_id, save_dir)
                        scorer.add_prediction(utt_id, hypo_str, bpe_symbol=args.remove_bpe)
                        if has_target:
                            scorer.add_evaluation(utt_id, target_str, hypo_str, bpe_symbol=args.remove_bpe)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Recognized {} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if args.print_alignment:
        print('| Saved attention plots in ' + save_dir)

    if has_target:
        assert args.test_text_files is not None
        scorer.add_ordered_utt_list(*args.test_text_files)

    os.makedirs(args.results_path, exist_ok=True)

    fn = 'decoded_char_results.txt'
    with open(os.path.join(args.results_path, fn), 'w', encoding='utf-8') as f:
        f.write(scorer.print_char_results())
        print('| Decoded char results saved as ' + f.name)

    fn = 'decoded_results.txt'
    with open(os.path.join(args.results_path, fn), 'w', encoding='utf-8') as f:
        f.write(scorer.print_results())
        print('| Decoded results saved as ' + f.name)

    if has_target:
        header = ' Recognize {} with beam={}: '.format(args.gen_subset, args.beam)
        fn = 'wer'
        with open(os.path.join(args.results_path, fn), 'w', encoding='utf-8') as f:
            res = 'WER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%'.format(
                *(scorer.wer()))
            print('|' + header + res)
            f.write(res + '\n')
            print('| WER saved in ' + f.name)

        fn = 'cer'
        with open(os.path.join(args.results_path, fn), 'w', encoding='utf-8') as f:
            res = 'CER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%'.format(
                *(scorer.cer()))
            print('|' + ' ' * len(header) + res)
            f.write(res + '\n')
            print('| CER saved in ' + f.name)

        fn = 'aligned_results.txt'
        with open(os.path.join(args.results_path, fn), 'w', encoding='utf-8') as f:
            f.write(scorer.print_aligned_results())
            print('| Aligned results saved as ' + f.name)
    return scorer


def print_options_meaning_changes(args):
    """Options that have different meanings than those in the translation task
    are explained here.
    """
    print('| --max-tokens is the maximum number of input frames in a batch')
    if args.print_alignment:
        print('| --print-alignment has been set to plot attentions')


def cli_main():
    parser = options.get_generation_parser(default_task='speech_recognition')
    parser.add_argument('--coverage-weight', default=0.0, type=float, metavar='W',
                        help='coverage weight in log-prob space, mostly to '
                        'reduce deletion errors while using the pretrained '
                        'external LM for decoding')
    parser.add_argument('--eos-factor', default=None, type=float, metavar='F',
                        help='only consider emitting EOS if its score is no less '
                        'than the specified factor of the best candidate score')
    parser.add_argument('--lm-weight', default=0.0, type=float, metavar='W',
                        help='LM weight in log-prob space, assuming the pretrained '
                        'external LM is specified as the second one in --path')
    parser.add_argument('--subwordlm-weight', default=0.8, type=float, metavar='W',
                        help='subword LM weight relative to word LM. Only relevant '
                        'to MultiLevelLanguageModel as an external LM')
    parser.add_argument('--oov-penalty', default=1e-4, type=float,
                        help='oov penalty with the pretrained external LM')
    parser.add_argument('--disable-open-vocab', action='store_true',
                        help='whether open vocabulary mode is enabled with the '
                        'pretrained external LM')
    args = options.parse_args_and_arch(parser)
    assert args.results_path is not None, 'please specify --results-path'
    print_options_meaning_changes(args)
    main(args)


if __name__ == '__main__':
    cli_main()
