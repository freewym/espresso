# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import string
import numpy as np
import os

import torch

from espresso.data import (
    AsrDictionary,
    AsrTextDataset,
    ScpCachedDataset,
    ScpInMemoryDataset,
    SpeechDataset,
)

try:
    import kaldi_io
except ImportError:
    raise ImportError('Please install kaldi_io with: pip install kaldi_io')


class TestSpeechDataset(unittest.TestCase):

    @staticmethod
    def make_dictionary():
        """construct dictionary."""
        d = AsrDictionary()
        alphabet = string.ascii_lowercase
        for token in alphabet:
            d.add_symbol(token)
        d.add_symbol('<space>')
        d.finalize(padding_factor=1)  # don't add extra padding symbols
        d.space_index = d.indices.get('<space>', -1)
        return d

    @staticmethod
    def generate_feats(test_dir, num=10, seed=0):
        """generate feature matrices."""
        feats = {}
        np.random.seed(seed)
        with open(
            os.path.join(test_dir, 'feats.scp'), 'w', encoding='utf-8',
        ) as f:
            for i in range(num):
                utt_id = 'utt_id_' + str(i)
                ark_file = os.path.join(test_dir, 'mat_' + str(i) + '.ark')
                f.write(utt_id + ' ' + ark_file + ':0\n')
                length = np.random.randint(200, 800)
                m = np.random.uniform(-10.0, 10.0, (length, 40))
                feats[utt_id] = m
                kaldi_io.write_mat(ark_file, m)
        return feats

    @staticmethod
    def generate_text_tokens(test_dir, num=10, seed=0):
        """generate token text, where utterances are in a (random) different
        order from those in feats.scp."""
        text_tokens = {}
        alphabet = string.ascii_lowercase
        space = '<space>'
        vocab = list(alphabet)
        vocab.append(space)
        np.random.seed(seed)
        with open(
            os.path.join(test_dir, 'text_tokens'), 'w', encoding='utf-8',
        ) as f:
            for i in np.random.permutation(range(num)):
                utt_id = 'utt_id_' + str(i)
                length = np.random.randint(10, 100)
                tokens = [
                    vocab[np.random.randint(0, len(vocab))] for _ in range(length)
                ]
                if tokens[0] == space:
                    tokens[0] = vocab[np.random.randint(0, len(vocab) - 1)]
                if tokens[-1] == space:
                    tokens[-1] = vocab[np.random.randint(0, len(vocab) - 1)]
                text_tokens[utt_id] = tokens
                f.write(utt_id + ' ' + ' '.join(tokens) + '\n')
        return text_tokens

    def setUp(self):
        self.test_dir = './temp'
        os.makedirs(self.test_dir, exist_ok=True)
        self.num_audios = 150
        self.num_transripts = 100
        self.batch_size = 8
        self.cache_size = 16
        self.dictionary = self.make_dictionary()
        self.expected_feats = self.generate_feats(
            self.test_dir, num=self.num_audios, seed=0,
        )
        self.expected_tokens = self.generate_text_tokens(
            self.test_dir, num=self.num_transripts, seed=1,
        )

        self.cuda = torch.cuda.is_available()

    def _speech_dataset_helper(
        self, all_in_memory=False, ordered_prefetch=False,
    ):
        if not all_in_memory:
            src_dataset = ScpCachedDataset(
                path=os.path.join(self.test_dir, 'feats.scp'),
                ordered_prefetch=ordered_prefetch,
                cache_size=self.cache_size,
            )
        else:
            src_dataset = ScpInMemoryDataset(
                path=os.path.join(self.test_dir, 'feats.scp')
            )
        tgt_dataset = AsrTextDataset(
            path=os.path.join(self.test_dir, 'text_tokens'),
            dictionary=self.dictionary,
        )

        dataset = SpeechDataset(
            src_dataset, src_dataset.sizes,
            tgt_dataset, tgt_dataset.sizes, self.dictionary,
            left_pad_source=False,
            left_pad_target=False,
            max_source_positions=1000,
            max_target_positions=200,
        )

        # assume one is a subset of the other
        expected_dataset_size = min(self.num_audios, self.num_transripts)
        self.assertEqual(len(dataset.src), expected_dataset_size)
        self.assertEqual(len(dataset.tgt), expected_dataset_size)

        indices = list(range(expected_dataset_size))
        batch_sampler = []
        for i in range(0, expected_dataset_size, self.batch_size):
            batch_sampler.append(indices[i:i+self.batch_size])

        if not all_in_memory:
            dataset.prefetch(indices)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
        )

        for i, batch in enumerate(iter(dataloader)):
            bsz = batch["nsentences"]
            self.assertEqual(bsz, len(batch_sampler[i]))
            src_frames = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            tgt_tokens = self.dictionary.string(batch["target"]).split('\n')
            tgt_tokens = [line.split(' ') for line in tgt_tokens]
            self.assertEqual(bsz, src_frames.size(0))
            self.assertEqual(bsz, src_lengths.numel())
            self.assertEqual(bsz, len(tgt_tokens))
            for j, utt_id in enumerate(batch["utt_id"]):
                self.assertTensorEqual(
                    torch.from_numpy(self.expected_feats[utt_id]).float(),
                    src_frames[j, :src_lengths[j], :]
                )
                self.assertEqual(
                    self.expected_tokens[utt_id],
                    tgt_tokens[j],
                )

    def test_speech_dataset_cached_no_ordered_prefetch(self):
        self._speech_dataset_helper(all_in_memory=False, ordered_prefetch=False)

    def test_speech_dataset_cached_with_ordered_prefetch(self):
        self._speech_dataset_helper(all_in_memory=False, ordered_prefetch=True)

    def test_speech_dataset_all_in_memory(self):
        self._speech_dataset_helper(all_in_memory=True)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        if (
            (t1.dtype == torch.short or t1.dtype == torch.int or
             t1.dtype == torch.long) and
            (t2.dtype == torch.short or t2.dtype == torch.int or
             t2.dtype == torch.long)
        ):
            self.assertEqual(t1.ne(t2).long().sum(), 0)
        else:
            self.assertEqual(t1.allclose(t2, rtol=1e-05, atol=1e-08), True)


if __name__ == "__main__":
    unittest.main()
