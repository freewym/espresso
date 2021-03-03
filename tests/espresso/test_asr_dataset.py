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
    AsrDataset,
    AsrDictionary,
    AsrTextDataset,
    FeatScpCachedDataset,
    FeatScpInMemoryDataset,
)

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


class TestAsrDataset(unittest.TestCase):

    @staticmethod
    def make_dictionary():
        """construct dictionary."""
        d = AsrDictionary()
        alphabet = string.ascii_lowercase
        for token in alphabet:
            d.add_symbol(token)
        d.add_symbol("<space>")
        d.finalize(padding_factor=1)  # don't add extra padding symbols
        d.space_index = d.indices.get("<space>", -1)
        return d

    @staticmethod
    def generate_feats(test_dir, num=10, seed=0):
        """generate feature matrices."""
        expected_feats = {}
        np.random.seed(seed)
        utt_ids, rxfiles, utt2num_frames = [], [], []
        for i in range(num):
            utt_id = "utt_id_" + str(i)
            ark_file = os.path.join(test_dir, "mat_" + str(i) + ".ark")
            length = np.random.randint(200, 800)
            m = np.random.uniform(-10.0, 10.0, (length, 40))
            expected_feats[utt_id] = m
            kaldi_io.write_mat(ark_file, m)
            utt_ids.append(utt_id)
            rxfiles.append(ark_file + ":0")
            utt2num_frames.append(length)
        return expected_feats, utt_ids, rxfiles, utt2num_frames

    @staticmethod
    def generate_text(test_dir, num=10, seed=0):
        """generate token text, where utterances are in a (random) different
        order from those in feats.scp."""
        expected_texts = {}
        alphabet = string.ascii_lowercase
        space = "<space>"
        vocab = list(alphabet)
        vocab.append(space)
        np.random.seed(seed)
        utt_ids, texts = [], []
        for i in np.random.permutation(range(num)):
            utt_id = "utt_id_" + str(i)
            length = np.random.randint(10, 100)
            tokens = [
                vocab[np.random.randint(0, len(vocab))] for _ in range(length)
            ]
            if tokens[0] == space:
                tokens[0] = vocab[np.random.randint(0, len(vocab) - 1)]
            if tokens[-1] == space:
                tokens[-1] = vocab[np.random.randint(0, len(vocab) - 1)]
            expected_texts[utt_id] = tokens
            utt_ids.append(utt_id)
            texts.append(" ".join(tokens))
        return expected_texts, utt_ids, texts

    def setUp(self):
        self.test_dir = "./temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.num_audios = 150
        self.num_transripts = 100
        self.batch_size = 8
        self.cache_size = 16
        self.dictionary = self.make_dictionary()
        (
            self.expected_feats, self.feats_utt_ids, self.rxfiles, self.utt2num_frames
        ) = self.generate_feats(self.test_dir, num=self.num_audios, seed=0)
        (
            self.expected_texts, self.text_utt_ids, self.texts
        ) = self.generate_text(self.test_dir, num=self.num_transripts, seed=1)

        self.cuda = torch.cuda.is_available()

    def _asr_dataset_helper(
        self, all_in_memory=False, ordered_prefetch=False, has_utt2num_frames=False,
    ):
        if not all_in_memory:
            src_dataset = FeatScpCachedDataset(
                utt_ids=self.feats_utt_ids,
                rxfiles=self.rxfiles,
                utt2num_frames=self.utt2num_frames if has_utt2num_frames else None,
                ordered_prefetch=ordered_prefetch,
                cache_size=self.cache_size,
            )
        else:
            src_dataset = FeatScpInMemoryDataset(
                utt_ids=self.feats_utt_ids,
                rxfiles=self.rxfiles,
                utt2num_frames=self.utt2num_frames if has_utt2num_frames else None,
            )
        tgt_dataset = AsrTextDataset(
            utt_ids=self.text_utt_ids,
            texts=self.texts,
            dictionary=self.dictionary,
        )

        dataset = AsrDataset(
            src_dataset, src_dataset.sizes,
            tgt_dataset, tgt_dataset.sizes, self.dictionary,
            left_pad_source=False,
            left_pad_target=False,
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
            tgt_tokens = self.dictionary.string(
                batch["target"], extra_symbols_to_ignore={self.dictionary.pad()}
            ).split("\n")
            tgt_tokens = [line.split(" ") for line in tgt_tokens]
            self.assertEqual(bsz, src_frames.size(0))
            self.assertEqual(bsz, src_lengths.numel())
            self.assertEqual(bsz, len(tgt_tokens))
            for j, utt_id in enumerate(batch["utt_id"]):
                self.assertTensorEqual(
                    torch.from_numpy(self.expected_feats[utt_id]).float(),
                    src_frames[j, :src_lengths[j], :]
                )
                self.assertEqual(
                    self.expected_texts[utt_id],
                    tgt_tokens[j],
                )

    def test_asr_dataset_cached_no_ordered_prefetch(self):
        self._asr_dataset_helper(all_in_memory=False, ordered_prefetch=False)

    def test_asr_dataset_cached_with_ordered_prefetch(self):
        self._asr_dataset_helper(all_in_memory=False, ordered_prefetch=True)

    def test_asr_dataset_all_in_memory(self):
        self._asr_dataset_helper(all_in_memory=True)

    def test_asr_dataset_has_utt2num_frames(self):
        self._asr_dataset_helper(has_utt2num_frames=True)

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
