"""Tests for manifest-driven worker agent functionality."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from worker_agent import WorkerAgent, RedisTransport


def _make_agent(transport):
    """Create a WorkerAgent with signal handlers suppressed."""
    with patch("signal.signal"):
        return WorkerAgent("job-123", transport)


def _make_redis_transport():
    with patch("worker_agent.redis_lib"):
        return RedisTransport("redis://localhost:6379")


MANIFEST_YAML = """\
name: tidal
displayName: Tidal Gated Transformer LM
version: 1.0.0
description: A test model

trainingPhases:
  - id: lm-training
    displayName: Language Model Pretraining
    entrypoint: Main.py
    configFiles:
      - configs/base_config.yaml
    args:
      config: "--config"
      resume: "--resume"
    concurrency: 1
    gpuTier: standard

  - id: rl-training
    displayName: RL Gating Controller
    entrypoint: train_rl.py
    configFiles:
      - configs/base_config.yaml
      - configs/rl_config.yaml
    args:
      config: "--config"
      rlConfig: "--rl-config"
      checkpoint: "--checkpoint"
      timesteps: "--timesteps"
    concurrency: 1
    gpuTier: standard

checkpointPatterns: []
generation:
  entrypoint: Generator.py
  args: {}
  defaultConfigPath: configs/base_config.yaml
  modes: []
  parameters: []
  modelCheckpointPatterns: []
  rlCheckpointPatterns: []
metrics:
  redisPrefix: "tidal"
  lm:
    directory: dashboard_metrics
    historyFile: metrics.jsonl
    statusFile: status.json
    latestFile: latest.json
    primaryKeys: []
  rl:
    directory: rl_metrics
    metricsFile: rl_training_metrics.json
    primaryKeys: []
redis:
  jobsHash: "tidal:jobs"
  jobsActiveSet: "tidal:jobs:active"
  signalPrefix: "tidal:job:"
  heartbeatPrefix: "tidal:worker:"
  updatesChannel: "tidal:job:updates"
  experimentsSet: "tidal:experiments"
infrastructure:
  pythonEnv: tidal-env
  dockerImage: "pytorch/pytorch:latest"
  requirementsFile: requirements.txt
  gpuTiers:
    standard:
      minGpuRamMb: 16000
      minCpuCores: 16
"""


class TestLoadManifest(unittest.TestCase):
    """Tests for _load_manifest()."""

    def test_loads_valid_manifest(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            plugin_dir = os.path.join(tmpdir, "plugins", "tidal")
            os.makedirs(plugin_dir)
            with open(os.path.join(plugin_dir, "manifest.yaml"), "w") as f:
                f.write(MANIFEST_YAML)

            manifest = agent._load_manifest("tidal")
            self.assertEqual(manifest["name"], "tidal")
            self.assertEqual(len(manifest["trainingPhases"]), 2)

    def test_raises_on_missing_manifest(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            with self.assertRaises(FileNotFoundError):
                agent._load_manifest("nonexistent")


class TestFindPhase(unittest.TestCase):
    """Tests for _find_phase()."""

    def test_finds_lm_training_phase(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            plugin_dir = os.path.join(tmpdir, "plugins", "tidal")
            os.makedirs(plugin_dir)
            with open(os.path.join(plugin_dir, "manifest.yaml"), "w") as f:
                f.write(MANIFEST_YAML)

            manifest = agent._load_manifest("tidal")
            phase = agent._find_phase(manifest, "lm-training")
            self.assertEqual(phase["id"], "lm-training")
            self.assertEqual(phase["entrypoint"], "Main.py")

    def test_finds_rl_training_phase(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            plugin_dir = os.path.join(tmpdir, "plugins", "tidal")
            os.makedirs(plugin_dir)
            with open(os.path.join(plugin_dir, "manifest.yaml"), "w") as f:
                f.write(MANIFEST_YAML)

            manifest = agent._load_manifest("tidal")
            phase = agent._find_phase(manifest, "rl-training")
            self.assertEqual(phase["id"], "rl-training")
            self.assertEqual(phase["entrypoint"], "train_rl.py")

    def test_raises_on_unknown_phase(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            plugin_dir = os.path.join(tmpdir, "plugins", "tidal")
            os.makedirs(plugin_dir)
            with open(os.path.join(plugin_dir, "manifest.yaml"), "w") as f:
                f.write(MANIFEST_YAML)

            manifest = agent._load_manifest("tidal")
            with self.assertRaises(ValueError) as ctx:
                agent._find_phase(manifest, "unknown-type")
            self.assertIn("unknown-type", str(ctx.exception))


class TestBuildCommand(unittest.TestCase):
    """Tests for _build_command()."""

    def test_builds_lm_training_command(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        phase = {
            "id": "lm-training",
            "entrypoint": "Main.py",
            "args": {
                "config": "--config",
                "resume": "--resume",
            },
        }
        config = {
            "type": "lm-training",
            "configPath": "configs/base_config.yaml",
        }
        plugin_dir = "/project/plugins/tidal"

        args = agent._build_command(phase, config, plugin_dir)
        self.assertEqual(args[0], sys.executable)
        self.assertEqual(args[1], os.path.join(plugin_dir, "Main.py"))
        self.assertIn("--config", args)
        idx = args.index("--config")
        self.assertEqual(args[idx + 1], "configs/base_config.yaml")

    def test_builds_rl_training_command(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        phase = {
            "id": "rl-training",
            "entrypoint": "train_rl.py",
            "args": {
                "config": "--config",
                "rlConfig": "--rl-config",
                "checkpoint": "--checkpoint",
                "timesteps": "--timesteps",
            },
        }
        config = {
            "type": "rl-training",
            "configPath": "configs/base_config.yaml",
            "rlConfigPath": "configs/rl_config.yaml",
            "checkpoint": "experiments/exp-1/model.pth",
            "timesteps": 5000,
        }
        plugin_dir = "/project/plugins/tidal"

        args = agent._build_command(phase, config, plugin_dir)
        self.assertEqual(args[0], sys.executable)
        self.assertEqual(args[1], os.path.join(plugin_dir, "train_rl.py"))
        self.assertIn("--config", args)
        self.assertIn("--rl-config", args)
        self.assertIn("--checkpoint", args)
        self.assertIn("--timesteps", args)

        idx = args.index("--timesteps")
        self.assertEqual(args[idx + 1], "5000")

    def test_skips_missing_optional_args(self):
        transport = _make_redis_transport()
        agent = _make_agent(transport)

        phase = {
            "id": "lm-training",
            "entrypoint": "Main.py",
            "args": {
                "config": "--config",
                "resume": "--resume",
            },
        }
        config = {
            "type": "lm-training",
            "configPath": "configs/base_config.yaml",
            # No resume — should be skipped
        }
        plugin_dir = "/project/plugins/tidal"

        args = agent._build_command(phase, config, plugin_dir)
        self.assertNotIn("--resume", args)


if __name__ == "__main__":
    unittest.main()
