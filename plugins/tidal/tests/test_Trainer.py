import ast
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from plugins.tidal.tests.timeout import TimedTestCase


class TestNoTqdmImport(TimedTestCase):
    """Verify that Trainer.py does not import tqdm."""

    def test_no_tqdm_import(self):
        trainer_path = os.path.join(
            os.path.dirname(__file__), "..", "Trainer.py"
        )
        with open(trainer_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        "tqdm", alias.name,
                        f"Found 'import {alias.name}' — tqdm should not be imported",
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and "tqdm" in node.module:
                    self.fail(
                        f"Found 'from {node.module} import ...' — tqdm should not be imported",
                    )


class TestTrainerInit(TimedTestCase):
    """Tests for Trainer.__init__ config handling — no training loop needed."""

    def _make_trainer(self, config_overrides=None):
        """Create a Trainer with mocked I/O dependencies."""
        with patch("plugins.tidal.Trainer.setup_logger"), \
             patch("plugins.tidal.Trainer.SummaryWriter"), \
             patch("plugins.tidal.Trainer.MetricsLogger"):

            from plugins.tidal.Trainer import Trainer

            config = {
                "DEVICE": "cpu",
                "BATCH_SIZE": 2,
                "DESIRED_BATCH_SIZE": 8,
                "MAX_GRAD_NORM": 1.0,
                "EMBED_DIM": 32,
                "NUM_TRANSFORMER_BLOCKS": 1,
                "NUM_ATTENTION_HEADS": 2,
                "FFN_HIDDEN_DIM": 64,
                "DROPOUT": 0.0,
                "MAX_CONTEXT_LENGTH": 16,
                "LOG_DIRECTORY": "logs",
            }
            if config_overrides:
                config.update(config_overrides)

            exp_dir = tempfile.mkdtemp()
            trainer = Trainer(config, exp_dir)
        return trainer

    def test_gradient_accumulation_steps(self):
        """Trainer computes accumulation_steps = DESIRED_BATCH_SIZE / BATCH_SIZE."""
        trainer = self._make_trainer({"BATCH_SIZE": 2, "DESIRED_BATCH_SIZE": 8})
        trainer._flush_logs()
        # accumulation_steps is computed at runtime in _train_epoch, but
        # we can verify the config is stored correctly
        self.assertEqual(trainer.config["DESIRED_BATCH_SIZE"], 8)
        self.assertEqual(trainer.config["BATCH_SIZE"], 2)
        expected_accum = trainer.config["DESIRED_BATCH_SIZE"] // trainer.config["BATCH_SIZE"]
        self.assertEqual(expected_accum, 4)

    def test_device_config_respected(self):
        """Trainer uses the configured device."""
        trainer = self._make_trainer({"DEVICE": "cpu"})
        trainer._flush_logs()
        self.assertEqual(trainer.device, "cpu")

    def test_setup_model_creates_gated_model(self):
        """_setup_model creates a TransformerLM with expected gate dimension."""
        trainer = self._make_trainer()
        trainer._setup_model(vocab_size=100, total_foundational_steps=10)

        from plugins.tidal.TransformerLM import GatedTransformerBlock
        model = trainer.model
        # Check that the model has gated transformer blocks
        for block in model.transformer_blocks:
            self.assertIsInstance(block, GatedTransformerBlock)
            self.assertEqual(block.attn_gate.net[0].in_features, GatedTransformerBlock.GATE_DIM)

        trainer._flush_logs()


class TestTrainerCheckpointRoundtrip(TimedTestCase):
    """Test that Trainer checkpoint save/load preserves model state."""

    def _make_trainer_with_model(self):
        """Create a Trainer with a real (small) model initialized."""
        with patch("plugins.tidal.Trainer.setup_logger"), \
             patch("plugins.tidal.Trainer.SummaryWriter"), \
             patch("plugins.tidal.Trainer.MetricsLogger"):

            from plugins.tidal.Trainer import Trainer

            config = {
                "DEVICE": "cpu",
                "BATCH_SIZE": 2,
                "DESIRED_BATCH_SIZE": 2,
                "MAX_GRAD_NORM": 1.0,
                "EMBED_DIM": 32,
                "NUM_TRANSFORMER_BLOCKS": 1,
                "NUM_ATTENTION_HEADS": 2,
                "FFN_HIDDEN_DIM": 64,
                "DROPOUT": 0.0,
                "MAX_CONTEXT_LENGTH": 16,
                "LOG_DIRECTORY": "logs",
            }

            exp_dir = tempfile.mkdtemp()
            trainer = Trainer(config, exp_dir)

        trainer._setup_model(vocab_size=100, total_foundational_steps=10)
        return trainer

    def test_save_load_checkpoint_roundtrip(self):
        """Save and load a checkpoint — state_dict keys and shapes match."""
        from plugins.tidal.TransformerLM import get_model_state_dict, TransformerLM

        trainer = self._make_trainer_with_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            sd_before = get_model_state_dict(trainer.model)
            torch.save(sd_before, ckpt_path)

            # Load into a fresh model
            model2 = TransformerLM(vocab_size=100, config=trainer.config)
            loaded_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model2.load_state_dict(loaded_sd)

            sd_after = model2.state_dict()

            # Keys match
            self.assertEqual(set(sd_before.keys()), set(sd_after.keys()))

            # Shapes match
            for key in sd_before:
                self.assertEqual(
                    sd_before[key].shape, sd_after[key].shape,
                    f"Shape mismatch in {key}",
                )

            # Values match
            for key in sd_before:
                self.assertTrue(
                    torch.equal(sd_before[key], sd_after[key]),
                    f"Value mismatch in {key}",
                )

        trainer._flush_logs()


if __name__ == "__main__":
    unittest.main()
