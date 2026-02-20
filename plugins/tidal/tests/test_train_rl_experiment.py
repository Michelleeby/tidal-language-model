"""Tests for RL training experiment directory creation.

Verifies that RL training always creates an independent experiment directory
with metadata linking back to the source LM experiment.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from experiment_utils import read_experiment_metadata


class TestRLExperimentDirectory(unittest.TestCase):
    """Tests for create_rl_experiment_dir() in train_rl.py."""

    def setUp(self):
        """Import here to avoid import errors before implementation."""
        from plugins.tidal.train_rl import create_rl_experiment_dir
        self.create_rl_experiment_dir = create_rl_experiment_dir

    def test_creates_new_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = os.path.join(tmpdir, "experiments")
            os.makedirs(experiments_root)

            checkpoint_path = os.path.join(
                experiments_root, "20250101-commit_abc-config_def", "model.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path))
            open(checkpoint_path, "w").close()

            rl_config_path = os.path.join(tmpdir, "rl_config.yaml")
            base_config_path = os.path.join(tmpdir, "base_config.yaml")
            with open(rl_config_path, "w") as f:
                f.write("RL_TOTAL_TIMESTEPS: 1000\n")
            with open(base_config_path, "w") as f:
                f.write("VOCAB_SIZE: 50257\n")

            exp_dir, exp_id = self.create_rl_experiment_dir(
                checkpoint_path=checkpoint_path,
                rl_config_path=rl_config_path,
                base_config_path=base_config_path,
                experiments_root=experiments_root,
            )

            self.assertTrue(os.path.isdir(exp_dir))
            # Should be a NEW directory, not the source experiment
            self.assertNotEqual(
                os.path.basename(exp_dir), "20250101-commit_abc-config_def"
            )

    def test_metadata_links_to_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = os.path.join(tmpdir, "experiments")
            source_exp_id = "20250101-commit_abc-config_def"
            source_dir = os.path.join(experiments_root, source_exp_id)
            os.makedirs(source_dir)

            checkpoint_path = os.path.join(source_dir, "model.pth")
            open(checkpoint_path, "w").close()

            rl_config_path = os.path.join(tmpdir, "rl_config.yaml")
            base_config_path = os.path.join(tmpdir, "base_config.yaml")
            with open(rl_config_path, "w") as f:
                f.write("RL_TOTAL_TIMESTEPS: 1000\n")
            with open(base_config_path, "w") as f:
                f.write("VOCAB_SIZE: 50257\n")

            exp_dir, exp_id = self.create_rl_experiment_dir(
                checkpoint_path=checkpoint_path,
                rl_config_path=rl_config_path,
                base_config_path=base_config_path,
                experiments_root=experiments_root,
            )

            meta = read_experiment_metadata(exp_dir)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["type"], "rl")
            self.assertEqual(meta["source_experiment_id"], source_exp_id)
            self.assertIn("model.pth", meta["source_checkpoint"])

    def test_copies_both_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = os.path.join(tmpdir, "experiments")
            source_dir = os.path.join(experiments_root, "src-exp")
            os.makedirs(source_dir)

            checkpoint_path = os.path.join(source_dir, "model.pth")
            open(checkpoint_path, "w").close()

            rl_config_path = os.path.join(tmpdir, "rl_config.yaml")
            base_config_path = os.path.join(tmpdir, "base_config.yaml")
            with open(rl_config_path, "w") as f:
                f.write("RL_TOTAL_TIMESTEPS: 1000\n")
            with open(base_config_path, "w") as f:
                f.write("VOCAB_SIZE: 50257\n")

            exp_dir, _ = self.create_rl_experiment_dir(
                checkpoint_path=checkpoint_path,
                rl_config_path=rl_config_path,
                base_config_path=base_config_path,
                experiments_root=experiments_root,
            )

            self.assertTrue(os.path.exists(os.path.join(exp_dir, "rl_config.yaml")))
            self.assertTrue(os.path.exists(os.path.join(exp_dir, "base_config.yaml")))

    def test_experiment_id_contains_rl_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = os.path.join(tmpdir, "experiments")
            source_dir = os.path.join(experiments_root, "src-exp")
            os.makedirs(source_dir)

            checkpoint_path = os.path.join(source_dir, "model.pth")
            open(checkpoint_path, "w").close()

            rl_config_path = os.path.join(tmpdir, "rl_config.yaml")
            base_config_path = os.path.join(tmpdir, "base_config.yaml")
            with open(rl_config_path, "w") as f:
                f.write("RL_TOTAL_TIMESTEPS: 1000\n")
            with open(base_config_path, "w") as f:
                f.write("VOCAB_SIZE: 50257\n")

            _, exp_id = self.create_rl_experiment_dir(
                checkpoint_path=checkpoint_path,
                rl_config_path=rl_config_path,
                base_config_path=base_config_path,
                experiments_root=experiments_root,
            )

            self.assertIn("-rl_", exp_id)

    def test_checkpoint_outside_experiments_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = os.path.join(tmpdir, "experiments")
            os.makedirs(experiments_root)

            # Checkpoint outside experiments/
            external_dir = os.path.join(tmpdir, "external")
            os.makedirs(external_dir)
            checkpoint_path = os.path.join(external_dir, "model.pth")
            open(checkpoint_path, "w").close()

            rl_config_path = os.path.join(tmpdir, "rl_config.yaml")
            base_config_path = os.path.join(tmpdir, "base_config.yaml")
            with open(rl_config_path, "w") as f:
                f.write("RL_TOTAL_TIMESTEPS: 1000\n")
            with open(base_config_path, "w") as f:
                f.write("VOCAB_SIZE: 50257\n")

            exp_dir, exp_id = self.create_rl_experiment_dir(
                checkpoint_path=checkpoint_path,
                rl_config_path=rl_config_path,
                base_config_path=base_config_path,
                experiments_root=experiments_root,
            )

            self.assertTrue(os.path.isdir(exp_dir))
            meta = read_experiment_metadata(exp_dir)
            self.assertEqual(meta["type"], "rl")
            # source_experiment_id should be None when checkpoint is external
            self.assertIsNone(meta["source_experiment_id"])


if __name__ == "__main__":
    unittest.main()
