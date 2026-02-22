"""
test_experiment_infra.py

Unit tests for ADR 0008 experiment infrastructure:
  - FREEZE_GATES support in Trainer
  - VAL_SUBSET support in TinyStoriesDataset
  - Config overlay merging (--overlay for Main.py)
  - Evaluator VAL_SUBSET threading
"""

import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call

import torch
import torch.utils.data

from plugins.tidal.tests.timeout import TimedTestCase


# ── Shared helpers ────────────────────────────────────────────────────────────

def _base_trainer_config(overrides=None):
    """Return a minimal Trainer config with input_dependent gate mode."""
    config = {
        "DEVICE": "cpu",
        "BATCH_SIZE": 2,
        "DESIRED_BATCH_SIZE": 4,
        "MAX_GRAD_NORM": 1.0,
        "EMBED_DIM": 32,
        "NUM_TRANSFORMER_BLOCKS": 2,
        "NUM_ATTENTION_HEADS": 2,
        "FFN_HIDDEN_DIM": 64,
        "DROPOUT": 0.0,
        "MAX_CONTEXT_LENGTH": 16,
        "LOG_DIRECTORY": "logs",
        "GATE_DIM": 1,
        "GATE_MODE": "input_dependent",
        "GATE_REG_WEIGHT": 0.0,
        "LEARNING_RATE_SCHEDULER": {
            "BASE_LR": 0.001,
            "MIN_LR": 1e-6,
            "WARMUP_RATIO": 0.1,
        },
    }
    if overrides:
        config.update(overrides)
    return config


def _make_trainer(config):
    """Instantiate a Trainer with mocked I/O dependencies."""
    with patch("plugins.tidal.Trainer.setup_logger"), \
         patch("plugins.tidal.Trainer.SummaryWriter"), \
         patch("plugins.tidal.Trainer.MetricsLogger"):
        from plugins.tidal.Trainer import Trainer
        exp_dir = tempfile.mkdtemp()
        return Trainer(config, exp_dir)


def _make_fake_val_cache(tmpdir, num_chunks, max_length=16):
    """Write a synthetic uint16 validation cache to tmpdir and return the tensor."""
    chunk_length = max_length + 1
    fake_chunks = torch.randint(0, 1000, (num_chunks, chunk_length), dtype=torch.uint16)
    cache_path = os.path.join(tmpdir, f"validation_ctx{max_length}.pt")
    torch.save(fake_chunks, cache_path)
    return fake_chunks


# ── FREEZE_GATES tests ────────────────────────────────────────────────────────

class TestFreezeGates(TimedTestCase):
    """Trainer FREEZE_GATES: frozen input-dependent gates stay untrainable."""

    def test_freeze_gates_no_gradient(self):
        """Gate params must have requires_grad=False after _setup_model with FREEZE_GATES=True."""
        config = _base_trainer_config({"FREEZE_GATES": True})
        trainer = _make_trainer(config)
        trainer._setup_model(vocab_size=100, total_foundational_steps=1)
        trainer._flush_logs()

        gate_params = [
            (name, param)
            for name, param in trainer.model.named_parameters()
            if ".attn_gate." in name or ".ffn_gate." in name
        ]
        self.assertGreater(len(gate_params), 0, "No gate parameters found in model")
        for name, param in gate_params:
            self.assertFalse(
                param.requires_grad,
                f"Gate param '{name}' should be frozen (requires_grad=False)",
            )

    def test_freeze_gates_constant_output(self):
        """InputDependentGate at init (zero weight, bias=2.0) produces sigmoid(2.0) ≈ 0.88."""
        from plugins.tidal.TransformerLM import TransformerLM

        config = _base_trainer_config()
        model = TransformerLM(vocab_size=100, config=config)
        model.train(False)  # inference mode — equivalent to model.eval()

        expected = 1.0 / (1.0 + math.exp(-2.0))  # sigmoid(2.0) ≈ 0.8808

        input_ids = torch.randint(0, 100, (2, 16))
        target_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            _, _, viz = model(input_ids, target_ids, return_gate_activations=True)

        gate_acts = viz.get("gate_activations", [])
        self.assertGreater(len(gate_acts), 0, "No gate_activations in viz_data")

        for i, (attn_gate, ffn_gate) in enumerate(gate_acts):
            if attn_gate is not None:
                self.assertAlmostEqual(
                    attn_gate.mean().item(), expected, delta=0.05,
                    msg=f"Layer {i} attn_gate mean should be ≈ {expected:.3f}",
                )
            if ffn_gate is not None:
                self.assertAlmostEqual(
                    ffn_gate.mean().item(), expected, delta=0.05,
                    msg=f"Layer {i} ffn_gate mean should be ≈ {expected:.3f}",
                )

    def test_freeze_gates_requires_input_dependent(self):
        """FREEZE_GATES=True with GATE_MODE='external' must raise ValueError in __init__."""
        config = _base_trainer_config({
            "FREEZE_GATES": True,
            "GATE_MODE": "external",
        })
        with self.assertRaises(ValueError) as ctx:
            _make_trainer(config)
        msg = str(ctx.exception)
        self.assertIn("FREEZE_GATES", msg)
        self.assertIn("input_dependent", msg)

    def test_freeze_gates_grad_clip_excludes_frozen(self):
        """trainer.trainable_params must exist and exclude frozen gate parameters."""
        config = _base_trainer_config({"FREEZE_GATES": True})
        trainer = _make_trainer(config)
        trainer._setup_model(vocab_size=100, total_foundational_steps=1)
        trainer._flush_logs()

        self.assertTrue(
            hasattr(trainer, "trainable_params"),
            "Trainer must expose self.trainable_params",
        )

        frozen_gate_ids = {
            id(param)
            for name, param in trainer.model.named_parameters()
            if ".attn_gate." in name or ".ffn_gate." in name
        }
        trainable_ids = {id(p) for p in trainer.trainable_params}

        overlap = frozen_gate_ids & trainable_ids
        self.assertEqual(
            len(overlap), 0,
            "Frozen gate params must not appear in trainer.trainable_params",
        )


# ── VAL_SUBSET tests ──────────────────────────────────────────────────────────

class TestValSubset(TimedTestCase):
    """TinyStoriesDataset subset parameter for two-stage validation protocol."""

    _timeout_seconds = 30

    MAX_LENGTH = 16
    NUM_CHUNKS = 11  # Odd to test odd-count handling

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.fake_chunks = _make_fake_val_cache(cls.tmpdir, cls.NUM_CHUNKS, cls.MAX_LENGTH)

    def _make_ds(self, subset=None):
        """Load a TinyStoriesDataset backed by the synthetic cache."""
        import plugins.tidal.DataPipeline as dp
        original = dp.CACHE_DIR
        dp.CACHE_DIR = self.tmpdir
        try:
            from plugins.tidal.DataPipeline import TinyStoriesDataset
            return TinyStoriesDataset("validation", max_length=self.MAX_LENGTH, subset=subset)
        finally:
            dp.CACHE_DIR = original

    def test_val_subset_first_half(self):
        """subset='first_half' returns exactly the first n//2 chunks."""
        ds = self._make_ds("first_half")
        expected = self.NUM_CHUNKS // 2
        self.assertEqual(len(ds), expected)

    def test_val_subset_second_half(self):
        """subset='second_half' returns the remaining chunks after n//2."""
        ds = self._make_ds("second_half")
        mid = self.NUM_CHUNKS // 2
        expected = self.NUM_CHUNKS - mid
        self.assertEqual(len(ds), expected)

    def test_val_subsets_disjoint(self):
        """first_half and second_half share no chunk rows."""
        import plugins.tidal.DataPipeline as dp
        original = dp.CACHE_DIR
        dp.CACHE_DIR = self.tmpdir
        try:
            from plugins.tidal.DataPipeline import TinyStoriesDataset
            ds_first = TinyStoriesDataset("validation", max_length=self.MAX_LENGTH, subset="first_half")
            ds_second = TinyStoriesDataset("validation", max_length=self.MAX_LENGTH, subset="second_half")
        finally:
            dp.CACHE_DIR = original

        first_set = {tuple(ds_first.chunks[i].tolist()) for i in range(len(ds_first))}
        second_set = {tuple(ds_second.chunks[i].tolist()) for i in range(len(ds_second))}
        overlap = first_set & second_set
        self.assertEqual(len(overlap), 0, "first_half and second_half must be disjoint")

    def test_val_subsets_cover_full(self):
        """len(first_half) + len(second_half) == len(full dataset)."""
        import plugins.tidal.DataPipeline as dp
        original = dp.CACHE_DIR
        dp.CACHE_DIR = self.tmpdir
        try:
            from plugins.tidal.DataPipeline import TinyStoriesDataset
            ds_full = TinyStoriesDataset("validation", max_length=self.MAX_LENGTH)
            ds_first = TinyStoriesDataset("validation", max_length=self.MAX_LENGTH, subset="first_half")
            ds_second = TinyStoriesDataset("validation", max_length=self.MAX_LENGTH, subset="second_half")
        finally:
            dp.CACHE_DIR = original

        self.assertEqual(
            len(ds_first) + len(ds_second), len(ds_full),
            "first_half + second_half must cover all chunks",
        )

    def test_val_subset_invalid_raises(self):
        """subset='invalid' must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self._make_ds("invalid")
        self.assertIn("invalid", str(ctx.exception).lower())


# ── Config overlay tests ──────────────────────────────────────────────────────

class TestConfigOverlay(TimedTestCase):
    """--overlay merges a second YAML on top of the base config in Main.py."""

    def _write_yaml(self, content):
        """Write content to a temp YAML file; caller must os.unlink it."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        return f.name

    def _load_and_merge(self, base_content, overlay_content):
        """Simulate what Main.py does: load base, load overlay, config.update(overlay)."""
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")

        base_path = self._write_yaml(base_content)
        overlay_path = self._write_yaml(overlay_content)
        try:
            with open(base_path) as f:
                config = yaml.load(f)
            with open(overlay_path) as f:
                overlay = yaml.load(f)
            config.update(overlay)
            return config
        finally:
            os.unlink(base_path)
            os.unlink(overlay_path)

    def test_config_overlay_merge(self):
        """Overlay values replace base values for matching keys."""
        base = "GATE_MODE: external\nGATE_REG_WEIGHT: 0.0\n"
        overlay = "GATE_MODE: input_dependent\n"
        config = self._load_and_merge(base, overlay)
        self.assertEqual(config["GATE_MODE"], "input_dependent")

    def test_config_overlay_preserves_base(self):
        """Keys not present in overlay are preserved from the base config."""
        base = "GATE_MODE: external\nGATE_REG_WEIGHT: 0.0\nRANDOM_SEED: 42\n"
        overlay = "GATE_MODE: input_dependent\n"
        config = self._load_and_merge(base, overlay)
        self.assertEqual(config["GATE_REG_WEIGHT"], 0.0)
        self.assertEqual(config["RANDOM_SEED"], 42)


# ── Evaluator VAL_SUBSET threading tests ──────────────────────────────────────

class _FakeDataset(torch.utils.data.Dataset):
    """Synthetic dataset that works with DataLoader (no HuggingFace needed)."""

    def __init__(self, n=20, seq_len=16, vocab_size=100):
        self.pairs = [
            (torch.randint(0, vocab_size, (seq_len,)),
             torch.randint(0, vocab_size, (seq_len,)))
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class TestEvaluatorValSubset(TimedTestCase):
    """Evaluator must forward VAL_SUBSET to TinyStoriesDataset on every dataset creation."""

    _timeout_seconds = 30

    @classmethod
    def setUpClass(cls):
        from plugins.tidal.TransformerLM import TransformerLM

        cls.seq_len = 16
        cls.vocab_size = 100
        cls.config = {
            "EMBED_DIM": 32,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 2,
            "FFN_HIDDEN_DIM": 64,
            "DROPOUT": 0.0,
            "MAX_CONTEXT_LENGTH": cls.seq_len,
            "DEVICE": "cpu",
            "LOG_DIRECTORY": "logs",
            "EVAL_BATCH_SIZE": 2,
            "NUM_CPU_CORE_WORKERS": 0,
            "VOCAB_SIZE": cls.vocab_size,
            "GATE_MODE": "input_dependent",
        }
        cls.tmpdir = tempfile.mkdtemp()
        model = TransformerLM(vocab_size=cls.vocab_size, config=cls.config)
        cls.model_path = os.path.join(cls.tmpdir, "model.pt")
        torch.save(model.state_dict(), cls.model_path)

    def _make_evaluator(self, config_overrides=None):
        config = dict(self.config)
        if config_overrides:
            config.update(config_overrides)
        from plugins.tidal.Evaluator import Evaluator
        return Evaluator(config, self.tmpdir, self.model_path)

    def _fake_ds(self):
        return _FakeDataset(n=8, seq_len=self.seq_len, vocab_size=self.vocab_size)

    def test_evaluator_compute_perplexity_threads_val_subset(self):
        """compute_perplexity() must pass VAL_SUBSET from config to TinyStoriesDataset."""
        ev = self._make_evaluator({"VAL_SUBSET": "first_half"})

        with patch("plugins.tidal.Evaluator.TinyStoriesDataset", return_value=self._fake_ds()) as MockDS:
            try:
                ev.compute_perplexity()
            except Exception:
                pass  # Don't care if it errors; we only inspect the call args

        calls = MockDS.call_args_list
        self.assertGreater(len(calls), 0, "TinyStoriesDataset was never called")
        found = any(c.kwargs.get("subset") == "first_half" for c in calls)
        self.assertTrue(
            found,
            f"No TinyStoriesDataset call with subset='first_half'. Actual calls: {calls}",
        )

    def test_evaluator_analyze_gate_activations_threads_val_subset(self):
        """analyze_gate_activations() must pass VAL_SUBSET from config to TinyStoriesDataset."""
        ev = self._make_evaluator({"VAL_SUBSET": "second_half"})

        with patch("plugins.tidal.Evaluator.TinyStoriesDataset", return_value=self._fake_ds()) as MockDS:
            try:
                ev.analyze_gate_activations(max_batches=1)
            except Exception:
                pass

        calls = MockDS.call_args_list
        self.assertGreater(len(calls), 0, "TinyStoriesDataset was never called")
        found = any(c.kwargs.get("subset") == "second_half" for c in calls)
        self.assertTrue(
            found,
            f"No TinyStoriesDataset call with subset='second_half'. Actual calls: {calls}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
