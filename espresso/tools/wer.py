# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, OrderedDict
import logging
import re

import espresso.tools.utils as speech_utils


logger = logging.getLogger(__name__)


class Scorer(object):
    def __init__(self, dictionary, wer_output_filter=None):
        self.dictionary = dictionary
        self.ordered_utt_list = None
        self.word_filters = []
        self.parse_wer_output_filter(wer_output_filter)
        self.reset()

    def reset(self):
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.char_results = OrderedDict()
        self.results = OrderedDict()
        self.aligned_results = OrderedDict()

    def parse_wer_output_filter(self, wer_output_filter):
        if wer_output_filter:
            with open(wer_output_filter, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#!') or line == '':
                        continue
                    elif line.startswith('s/'):
                        m = re.match(r's/(.+)/(.*)/g', line)
                        assert m is not None
                        self.word_filters.append([m.group(1), m.group(2)])
                    elif line.startswith('s:'):
                        m = re.match(r's:(.+):(.*):g', line)
                        assert m is not None
                        self.word_filters.append([m.group(1), m.group(2)])
                    else:
                        logger.warning('Unsupported pattern: "{}". Ignoring it'.format(line))

    def add_prediction(self, utt_id, pred):
        if not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {})'.format(type(utt_id)))
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))

        assert utt_id not in self.char_results, \
            'Duplicated utterance id detected: {}'.format(utt_id)
        self.char_results[utt_id] = pred + '\n'

        pred_words = self.dictionary.wordpiece_decode(pred)
        assert utt_id not in self.results, \
            'Duplicated utterance id detected: {}'.format(utt_id)
        self.results[utt_id] = pred_words + '\n'

    def add_evaluation(self, utt_id, ref, pred):
        if not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {})'.format(type(utt_id)))
        if not isinstance(ref, str):
            raise TypeError('ref must be a string (got {})'.format(type(ref)))
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))

        # filter out any non_lang_syms from ref and pred
        non_lang_syms = getattr(self.dictionary, 'non_lang_syms', None)
        assert non_lang_syms is None or isinstance(non_lang_syms, list)
        if non_lang_syms is not None and len(non_lang_syms) > 0:
            ref_list, pred_list = ref.strip().split(), pred.strip().split()
            ref = ' '.join([x for x in ref_list if x not in non_lang_syms])
            pred = ' '.join([x for x in pred_list if x not in non_lang_syms])

        # char level counts
        _, _, counter = speech_utils.edit_distance(
            ref.strip().split(), pred.strip().split(),
        )
        self.char_counter += counter

        # word level counts
        ref_words = self.dictionary.wordpiece_decode(ref)
        pred_words = self.dictionary.wordpiece_decode(pred)

        # filter words according to self.word_filters (support re.sub only)
        for pattern, repl in self.word_filters:
            ref_words = re.sub(pattern, repl, ref_words)
            pred_words = re.sub(pattern, repl, pred_words)

        ref_word_list, pred_word_list = ref_words.split(), pred_words.split()
        _, steps, counter = speech_utils.edit_distance(
            ref_word_list, pred_word_list,
        )
        self.word_counter += counter
        assert utt_id not in self.aligned_results, \
            'Duplicated utterance id detected: {}'.format(utt_id)
        self.aligned_results[utt_id] = speech_utils.aligned_print(
            ref_word_list, pred_word_list, steps,
        )

    def cer(self):
        assert self.char_counter['words'] > 0
        cer = float(
            self.char_counter['sub'] + self.char_counter['ins'] + self.char_counter['del']
        ) / self.char_counter['words'] * 100
        sub = float(self.char_counter['sub']) / self.char_counter['words'] * 100
        ins = float(self.char_counter['ins']) / self.char_counter['words'] * 100
        dlt = float(self.char_counter['del']) / self.char_counter['words'] * 100
        return cer, sub, ins, dlt

    def wer(self):
        assert self.word_counter['words'] > 0
        wer = float(
            self.word_counter['sub'] + self.word_counter['ins'] + self.word_counter['del']
        ) / self.word_counter['words'] * 100
        sub = float(self.word_counter['sub']) / self.word_counter['words'] * 100
        ins = float(self.word_counter['ins']) / self.word_counter['words'] * 100
        dlt = float(self.word_counter['del']) / self.word_counter['words'] * 100
        return wer, sub, ins, dlt

    def tot_word_error(self):
        return self.word_counter['sub'] + self.word_counter['ins'] + \
            self.word_counter['del']

    def tot_word_count(self):
        return self.word_counter['words']

    def tot_char_error(self):
        return self.char_counter['sub'] + self.char_counter['ins'] + \
            self.char_counter['del']

    def tot_char_count(self):
        return self.char_counter['words']

    def add_ordered_utt_list(self, *args):
        if len(args) == 1 and isinstance(args[0], list):  # aleady a list of utterance ids
            self.ordered_utt_list = args[0]
            return
        self.ordered_utt_list = []
        for text_file in args:
            with open(text_file, 'r', encoding='utf-8') as f:
                one_utt_list = [line.strip().split()[0] for line in f]
                self.ordered_utt_list.extend(one_utt_list)
        if len(self.char_results):
            assert set(self.ordered_utt_list) == set(self.char_results.keys())
        if len(self.results):
            assert set(self.ordered_utt_list) == set(self.results.keys())
        if len(self.aligned_results):
            assert set(self.ordered_utt_list) == set(self.aligned_results.keys())

    def print_char_results(self):
        res = ''
        if self.ordered_utt_list is not None:
            assert set(self.ordered_utt_list) == set(self.char_results.keys())
            for utt_id in self.ordered_utt_list:
                res += utt_id + ' ' + self.char_results[utt_id]
        else:
            for utt_id in self.char_results:
                res += utt_id + ' ' + self.char_results[utt_id]
        return res

    def print_results(self):
        res = ''
        if self.ordered_utt_list is not None:
            assert set(self.ordered_utt_list) == set(self.results.keys())
            for utt_id in self.ordered_utt_list:
                res += utt_id + ' ' + self.results[utt_id]
        else:
            for utt_id in self.results:
                res += utt_id + ' ' + self.results[utt_id]
        return res

    def print_aligned_results(self):
        res = ''
        if self.ordered_utt_list is not None:
            assert set(self.ordered_utt_list) == set(self.aligned_results.keys())
            for utt_id in self.ordered_utt_list:
                res += utt_id + '\n' + self.aligned_results[utt_id]
        else:
            for utt_id in self.aligned_results:
                res += utt_id + '\n' + self.aligned_results[utt_id]
        return res
