"""
test_DataPipeline.py

Unit tests for the TinyStories data pipeline.
"""

import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataPipeline import get_tokenizer, TinyStoriesDataset


class TestGetTokenizer(unittest.TestCase):
    """Tests for get_tokenizer function."""

    def test_returns_tokenizer(self):
        tokenizer = get_tokenizer()
        self.assertIsNotNone(tokenizer)

    def test_vocab_size(self):
        tokenizer = get_tokenizer()
        self.assertEqual(len(tokenizer), 50257)

    def test_encodes_text(self):
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode("Hello world")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_pad_token_set(self):
        tokenizer = get_tokenizer()
        self.assertIsNotNone(tokenizer.pad_token)


class TestTinyStoriesDataset(unittest.TestCase):
    """Tests for TinyStoriesDataset class."""

    @classmethod
    def setUpClass(cls):
        """Load a small validation split for testing."""
        cls.max_length = 32
        try:
            cls.dataset = TinyStoriesDataset(
                split="validation",
                max_length=cls.max_length,
            )
            cls.available = True
        except Exception:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("TinyStories dataset not available")

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.dataset), 0)

    def test_returns_tensor_pair(self):
        input_ids, target_ids = self.dataset[0]
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(target_ids, torch.Tensor)

    def test_sequence_length(self):
        input_ids, target_ids = self.dataset[0]
        self.assertEqual(input_ids.shape[0], self.max_length)
        self.assertEqual(target_ids.shape[0], self.max_length)

    def test_target_is_shifted(self):
        """Target should be input shifted by one position."""
        input_ids, target_ids = self.dataset[0]
        # target[i] should equal the token at position i+1 in the original chunk
        # This means input_ids[1:] should equal target_ids[:-1]
        self.assertTrue(torch.equal(input_ids[1:], target_ids[:-1]))

    def test_dtype_is_long(self):
        input_ids, target_ids = self.dataset[0]
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(target_ids.dtype, torch.long)

    def test_token_ids_in_vocab_range(self):
        input_ids, target_ids = self.dataset[0]
        self.assertTrue(torch.all(input_ids >= 0))
        self.assertTrue(torch.all(input_ids < 50257))
        self.assertTrue(torch.all(target_ids >= 0))
        self.assertTrue(torch.all(target_ids < 50257))


if __name__ == "__main__":
    unittest.main(verbosity=2)
