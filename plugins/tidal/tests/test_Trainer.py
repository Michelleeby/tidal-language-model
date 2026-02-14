import ast
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch
from torch.utils.data import DataLoader, TensorDataset


class TestNoTqdmImport(unittest.TestCase):
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


class TestTrainEpochMetrics(unittest.TestCase):
    """Verify that _train_epoch logs Iterations/Second and Epoch/Progress."""

    def _make_trainer(self):
        """Create a minimal Trainer with mocked internals."""
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

            import tempfile
            exp_dir = tempfile.mkdtemp()
            trainer = Trainer(config, exp_dir)

        # Lightweight model that returns the right shapes
        vocab_size = 100
        mock_model = MagicMock()
        mock_model.train = MagicMock()
        mock_model.parameters = MagicMock(return_value=[torch.zeros(1)])

        def forward_fn(input_seq, target_seq):
            batch_size, seq_len = input_seq.shape
            logits = torch.randn(batch_size, seq_len, vocab_size)
            loss = torch.tensor(2.0, requires_grad=True)
            return logits, (loss, None), None

        mock_model.side_effect = forward_fn
        mock_model.__call__ = forward_fn

        trainer.model = mock_model
        trainer.optimizer = MagicMock()
        trainer.optimizer.param_groups = [{"lr": 0.001}]
        trainer.scheduler = MagicMock()
        trainer.scaler = MagicMock()
        trainer.scaler.scale = MagicMock(return_value=MagicMock(backward=MagicMock()))

        trainer.current_epoch_num = 0

        return trainer

    def _make_loader(self, num_batches=4, batch_size=2, seq_len=16):
        """Create a simple DataLoader with random token data."""
        total = num_batches * batch_size
        input_ids = torch.randint(0, 100, (total, seq_len))
        target_ids = torch.randint(0, 100, (total, seq_len))
        dataset = TensorDataset(input_ids, target_ids)
        return DataLoader(dataset, batch_size=batch_size)

    def test_train_epoch_logs_iterations_per_second(self):
        trainer = self._make_trainer()
        loader = self._make_loader()

        logged_data = []
        original_log_metrics = trainer._log_metrics

        def capture_log_metrics(data, global_step):
            logged_data.append(data)

        trainer._log_metrics = capture_log_metrics

        trainer._train_epoch(loader)

        self.assertTrue(len(logged_data) > 0, "Expected at least one _log_metrics call")

        has_its = any("Iterations/Second" in d for d in logged_data)
        self.assertTrue(has_its, "Expected 'Iterations/Second' in logged metrics")

        for d in logged_data:
            if "Iterations/Second" in d:
                self.assertIsInstance(d["Iterations/Second"], float)
                self.assertGreater(d["Iterations/Second"], 0)

    def test_train_epoch_logs_epoch_progress(self):
        trainer = self._make_trainer()
        loader = self._make_loader()

        logged_data = []

        def capture_log_metrics(data, global_step):
            logged_data.append(data)

        trainer._log_metrics = capture_log_metrics

        trainer._train_epoch(loader)

        self.assertTrue(len(logged_data) > 0, "Expected at least one _log_metrics call")

        has_progress = any("Epoch/Progress" in d for d in logged_data)
        self.assertTrue(has_progress, "Expected 'Epoch/Progress' in logged metrics")

        for d in logged_data:
            if "Epoch/Progress" in d:
                val = d["Epoch/Progress"]
                self.assertIsInstance(val, float)
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
