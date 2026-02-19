"""
test_DataPipeline.py

Unit tests for the TinyStories data pipeline.
"""

import os
import tempfile
import unittest
import torch

from plugins.tidal.tests.timeout import TimedTestCase
from plugins.tidal.DataPipeline import get_tokenizer, TinyStoriesDataset, CACHE_DIR


class TestGetTokenizer(TimedTestCase):
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


class TestTinyStoriesDataset(TimedTestCase):
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

    def test_chunks_stored_as_uint16(self):
        """Internal chunk storage should use uint16 to minimise cache size."""
        # uint16 max (65535) comfortably fits GPT-2 vocab (50257)
        self.assertEqual(self.dataset.chunks.dtype, torch.uint16)

    def test_token_ids_in_vocab_range(self):
        input_ids, target_ids = self.dataset[0]
        self.assertTrue(torch.all(input_ids >= 0))
        self.assertTrue(torch.all(input_ids < 50257))
        self.assertTrue(torch.all(target_ids >= 0))
        self.assertTrue(torch.all(target_ids < 50257))


class TestStaleCacheRejection(TimedTestCase):
    """Verify that stale int64 cache files are rejected on load."""

    def test_stale_int64_cache_raises(self):
        """Loading a pre-existing int64 cache must raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a fake int64 cache file
            fake_chunks = torch.randint(0, 50257, (10, 33), dtype=torch.int64)
            cache_path = os.path.join(tmpdir, "train_ctx32.pt")
            torch.save(fake_chunks, cache_path)

            # Patch CACHE_DIR to point at our temp dir
            import plugins.tidal.DataPipeline as dp
            original_cache_dir = dp.CACHE_DIR
            dp.CACHE_DIR = tmpdir
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    TinyStoriesDataset(split="train", max_length=32)
                self.assertIn("uint16", str(ctx.exception))
            finally:
                dp.CACHE_DIR = original_cache_dir


if __name__ == "__main__":
    unittest.main(verbosity=2)
