# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter, OrderedDict

import speech_tools.utils as speech_utils


class Scorer(object):
    def __init__(self, dict):
        self.dict = dict
        self.ordered_utt_list = None
        self.reset()

    def reset(self):
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.results = OrderedDict()
        self.aligned_results = OrderedDict()

    def add_prediction(self, utt_id, pred):
        if not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {})'.format(type(utt_id)))
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))

        pred_words= speech_utils.Tokenizer.tokens_to_sentence(pred, self.dict)
        assert not utt_id in self.results, \
            'Duplicated utterance id detected: {}'.format(utt_id)
        self.results[utt_id] = pred_words + '\n'

    def add_evaluation(self, utt_id, ref, pred):
        if not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {})'.format(type(utt_id)))
        if not isinstance(ref, str):
            raise TypeError('ref must be a string (got {})'.format(type(ref)))
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))

        # char level counts
        _, _, counter = speech_utils.edit_distance(ref.strip().split(),
            pred.strip().split())
        self.char_counter += counter

        # word level counts
        ref_words = speech_utils.Tokenizer.tokens_to_sentence(ref, self.dict,
            use_unk_sym=False)
        pred_words= speech_utils.Tokenizer.tokens_to_sentence(pred, self.dict)
        ref_word_list, pred_word_list = ref_words.split(), pred_words.split()
        _, steps, counter = speech_utils.edit_distance(ref_word_list,
            pred_word_list)
        self.word_counter += counter
        assert not utt_id in self.aligned_results, \
            'Duplicated utterance id detected: {}'.format(utt_id)
        self.aligned_results[utt_id] = speech_utils.aligned_print(ref_word_list,
            pred_word_list, steps)

    def cer(self):
        assert self.char_counter['words'] > 0
        cer = float(self.char_counter['sub'] + self.char_counter['ins'] + \
            self.char_counter['del']) / self.char_counter['words'] * 100
        sub = float(self.char_counter['sub']) / self.char_counter['words'] * 100
        ins = float(self.char_counter['ins']) / self.char_counter['words'] * 100
        dlt = float(self.char_counter['del']) / self.char_counter['words'] * 100
        return cer, sub, ins, dlt

    def wer(self):
        assert self.word_counter['words'] > 0
        wer = float(self.word_counter['sub'] + self.word_counter['ins'] + \
            self.word_counter['del']) / self.word_counter['words'] * 100
        sub = float(self.word_counter['sub']) / self.word_counter['words'] * 100
        ins = float(self.word_counter['ins']) / self.word_counter['words'] * 100
        dlt = float(self.word_counter['del']) / self.word_counter['words'] * 100
        return wer, sub, ins, dlt

    def tot_word_error(self):
        return self.word_counter['sub'] + self.word_counter['ins'] + \
            self.word_counter['del']

    def tot_word_count(self):
        return self.word_counter['words']

    def add_ordered_utt_list(self, *args):
        self.ordered_utt_list = []
        for text_file in args:
            with open(text_file, 'r', encoding='utf-8') as f:
                one_utt_list = [line.strip().split()[0] for line in f]
                self.ordered_utt_list.extend(one_utt_list)
        if len(self.results):
            assert set(self.ordered_utt_list) == set(self.results.keys())
        if len(self.aligned_results):
            assert set(self.ordered_utt_list) == set(self.aligned_results.keys())

    def print_results(self):
        res = ''
        if self.order_utt_list is not None:
            assert set(self.ordered_utt_list) == set(self.results.keys())
            for utt_id in self.ordered_utt_list:
                res += utt_id + ' ' + self.results[utt_id]
        else:
            for utt_id in self.results:
                res += utt_id + ' ' + self.results[utt_id]
        return res
    
    def print_aligned_results(self):
        res = ''
        if self.order_utt_list is not None:
            assert set(self.ordered_utt_list) == set(self.aligned_results.keys())
            for utt_id in self.ordered_utt_list:
                res += utt_id + '\n' + self.aligned_results[utt_id]
        else:
            for utt_id in self.aligned_results:
                res += utt_id + '\n' + self.aligned_results[utt_id]
        return res

