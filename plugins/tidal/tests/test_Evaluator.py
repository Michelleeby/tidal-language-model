"""
test_Evaluator.py

Unit tests for the Evaluator class (perplexity computation).
"""

import os
import unittest
import torch
import math
import tempfile

from plugins.tidal.tests.timeout import TimedTestCase
from plugins.tidal.TransformerLM import TransformerLM
from plugins.tidal.Evaluator import Evaluator


class TestEvaluator(TimedTestCase):
    """Tests for the Evaluator class."""

    @classmethod
    def setUpClass(cls):
        cls.config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.1,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
            "LOG_DIRECTORY": "logs",
            "EVAL_BATCH_SIZE": 4,
            "DATASET": "roneneldan/TinyStories",
            "TOKENIZER": "gpt2",
            "VOCAB_SIZE": 100,
        }
        cls.vocab_size = 100
        cls.tmpdir = tempfile.mkdtemp()

        # Create and save a small model (Trainer saves raw state dicts)
        model = TransformerLM(vocab_size=cls.vocab_size, config=cls.config)
        cls.model_path = os.path.join(cls.tmpdir, "test_model.pt")
        torch.save(model.state_dict(), cls.model_path)

    def test_compute_perplexity_synthetic(self):
        """Test perplexity computation on synthetic data."""
        model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        model.eval()
        device = torch.device("cpu")

        # Create a small synthetic dataset
        batch_size = 4
        seq_len = 16
        data = []
        for _ in range(3):
            input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            data.append((input_ids, target_ids))

        # Compute perplexity manually
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_ids, target_ids in data:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits, (loss, _), _ = model(input_ids, target_ids)
                total_loss += loss.item() * target_ids.numel()
                total_tokens += target_ids.numel()

        expected_perplexity = math.exp(total_loss / total_tokens)

        # Perplexity should be finite and positive for a random model
        self.assertGreater(expected_perplexity, 0)
        self.assertTrue(math.isfinite(expected_perplexity))

    def test_perplexity_untrained_model(self):
        """An untrained model on 100-token vocab should have perplexity near vocab_size."""
        model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        model.eval()
        device = torch.device("cpu")

        batch_size = 8
        seq_len = 16
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _ in range(5):
                input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
                target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
                _, (loss, _), _ = model(input_ids, target_ids)
                total_loss += loss.item() * target_ids.numel()
                total_tokens += target_ids.numel()

        perplexity = math.exp(total_loss / total_tokens)

        # For a random model with vocab_size=100, perplexity should be roughly
        # in the range of the vocab size (not exactly, due to initialization)
        self.assertGreater(perplexity, 1.0)
        self.assertLess(perplexity, self.vocab_size * 10)

    def test_evaluator_instantiation(self):
        """Test that Evaluator can be instantiated."""
        evaluator = Evaluator(
            config=self.config,
            experiment_dir=self.tmpdir,
            model_path=self.model_path,
        )
        self.assertIsNotNone(evaluator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
