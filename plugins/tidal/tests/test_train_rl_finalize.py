"""Tests that RL training properly calls MetricsLogger.finalize().

The bug: train_rl.py never called metrics_logger.finalize() after training,
so RL experiments never reached "completed" status. This broke MCP caching
which only caches completed experiments.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestTrainRLFinalize(unittest.TestCase):
    """Verify finalize() is called after successful RL training."""

    def _run_main_with_mocked_training(self, train_side_effect=None):
        """Run train_rl.main() with all heavy dependencies mocked.

        Returns (mock_metrics_logger, mock_trainer) for assertions.
        """
        # Pre-populate sys.modules so train_rl.py's imports resolve without
        # loading transformers / torch / etc.
        stub_modules = {}
        for mod_name in [
            "torch", "torch.nn", "torch.nn.functional", "torch.optim",
            "torch.cuda", "torch.amp", "torch.utils", "torch.utils.data",
            "ruamel", "ruamel.yaml",
            "transformers",
            "datasets",
            "plugins.tidal.TransformerLM",
            "plugins.tidal.DataPipeline",
            "plugins.tidal.GatingPolicyAgent",
            "plugins.tidal.GatingEnvironment",
            "plugins.tidal.GatingModulator",
            "plugins.tidal.RewardComputer",
            "plugins.tidal.RLTrainer",
        ]:
            if mod_name not in sys.modules:
                stub_modules[mod_name] = sys.modules.setdefault(mod_name, MagicMock())

        try:
            # Force reimport with stubs
            if "plugins.tidal.train_rl" in sys.modules:
                del sys.modules["plugins.tidal.train_rl"]

            import plugins.tidal.train_rl as train_rl_mod

            mock_logger = MagicMock()
            mock_trainer = MagicMock()
            if train_side_effect:
                mock_trainer.train.side_effect = train_side_effect
            else:
                mock_trainer.train.return_value = {"rewards": [1.0]}

            mock_model = MagicMock()
            mock_model.vocab_size = 50257
            mock_model.parameters.return_value = []

            with tempfile.TemporaryDirectory() as tmpdir:
                exp_dir = os.path.join(tmpdir, "fake_exp")
                os.makedirs(exp_dir)

                # Create real config files so open() succeeds
                config_path = os.path.join(tmpdir, "fake.yaml")
                rl_config_path = os.path.join(tmpdir, "fake_rl.yaml")
                checkpoint_path = os.path.join(tmpdir, "fake.pth")
                for p in (config_path, rl_config_path, checkpoint_path):
                    with open(p, "w") as f:
                        f.write("")

                # Mock the yaml loader so it returns valid config dicts
                mock_yaml = MagicMock()
                mock_yaml.load.return_value = {
                    "VOCAB_SIZE": 50257,
                    "RL_TOTAL_TIMESTEPS": 100,
                }
                train_rl_mod.yaml = mock_yaml

                with (
                    patch.object(train_rl_mod, "load_model", return_value=mock_model),
                    patch.object(train_rl_mod, "extract_prompt_tokens", return_value=[[1, 2, 3]]),
                    patch.object(train_rl_mod, "create_rl_experiment_dir", return_value=(exp_dir, "fake_exp_id")),
                    patch.object(train_rl_mod, "report_experiment_id_to_job"),
                    patch.object(train_rl_mod, "MetricsLogger", return_value=mock_logger),
                    patch.object(train_rl_mod, "PPOTrainer", return_value=mock_trainer),
                    patch.object(train_rl_mod, "run_ablation_study", return_value={}),
                    patch.object(train_rl_mod, "GatingModulator", return_value=MagicMock()),
                    patch.object(train_rl_mod, "RewardComputer", return_value=MagicMock()),
                    patch.object(train_rl_mod, "GatingEnvironment", return_value=MagicMock()),
                    patch.object(train_rl_mod, "create_agent", return_value=MagicMock()),
                    patch.object(train_rl_mod, "resolve_device", return_value="cpu"),
                    patch("sys.argv", [
                        "train_rl.py",
                        "--config", config_path,
                        "--rl-config", rl_config_path,
                        "--checkpoint", checkpoint_path,
                    ]),
                ):
                    train_rl_mod.main()

            return mock_logger, mock_trainer

        finally:
            # Clean up stub modules
            for mod_name in stub_modules:
                if sys.modules.get(mod_name) is stub_modules[mod_name]:
                    del sys.modules[mod_name]
            if "plugins.tidal.train_rl" in sys.modules:
                del sys.modules["plugins.tidal.train_rl"]

    def test_finalize_called_after_successful_training(self):
        """finalize() must be called after trainer.train() completes."""
        mock_logger, mock_trainer = self._run_main_with_mocked_training()

        mock_trainer.train.assert_called_once()
        mock_logger.finalize.assert_called_once()

    def test_finalize_not_called_on_keyboard_interrupt(self):
        """finalize() must NOT be called when training is interrupted."""
        mock_logger, mock_trainer = self._run_main_with_mocked_training(
            train_side_effect=KeyboardInterrupt(),
        )

        # Training was interrupted — finalize should NOT have been called
        mock_logger.finalize.assert_not_called()
        # But interrupted checkpoint should have been saved
        mock_trainer.save_checkpoint.assert_called_once_with(
            "rl_checkpoint_interrupted.pth"
        )


if __name__ == "__main__":
    unittest.main()
