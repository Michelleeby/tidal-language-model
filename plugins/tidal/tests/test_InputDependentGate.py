"""
Tests for Direction A: Adaptive Depth via Input-Dependent Learned Gating.

Covers:
- InputDependentGate class (tests 1-4)
- GatedTransformerBlock with input-dependent mode (tests 5-9)
- TransformerLM with input-dependent gates (tests 10-13)
- Checkpoint compatibility (tests 14-15)
- Fixed-gate baseline / Model B (tests 16-17)
- Gate regularization in Trainer (tests 18-19)
- Gate analysis in Evaluator (tests 20-22)
- Hypothesis assessment (tests 23-26)
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from plugins.tidal.tests.timeout import TimedTestCase
from plugins.tidal.TransformerLM import (
    InputDependentGate,
    GatedTransformerBlock,
    TransformerLM,
    get_model_state_dict,
    load_model_state_dict,
)


# ── Shared test config ─────────────────────────────────────────────────────

SMALL_CONFIG_EXTERNAL = {
    "EMBED_DIM": 64,
    "NUM_TRANSFORMER_BLOCKS": 2,
    "NUM_ATTENTION_HEADS": 4,
    "FFN_HIDDEN_DIM": 128,
    "DROPOUT": 0.1,
    "MAX_CONTEXT_LENGTH": 32,
    "DEVICE": "cpu",
    "GATE_MODE": "external",
}

SMALL_CONFIG_INPUT_DEP = {
    **SMALL_CONFIG_EXTERNAL,
    "GATE_MODE": "input_dependent",
}

VOCAB_SIZE = 100


# ── 1-4: InputDependentGate ────────────────────────────────────────────────

class TestInputDependentGate(TimedTestCase):
    """Tests for the InputDependentGate module."""

    def setUp(self):
        self.embed_dim = 64
        self.gate = InputDependentGate(self.embed_dim)

    def test_output_shape(self):
        """1. Output is (batch, seq_len, 1) for input (batch, seq_len, embed_dim)."""
        x = torch.randn(4, 10, self.embed_dim)
        out = self.gate(x)
        self.assertEqual(out.shape, (4, 10, 1))

    def test_neutral_initialization(self):
        """2. Output mean in [0.85, 0.92] at init (sigmoid(2.0) is about 0.88)."""
        x = torch.randn(32, 20, self.embed_dim)
        with torch.no_grad():
            out = self.gate(x)
        mean_val = out.mean().item()
        self.assertGreaterEqual(mean_val, 0.85,
                                f"Init mean {mean_val:.4f} below 0.85")
        self.assertLessEqual(mean_val, 0.92,
                             f"Init mean {mean_val:.4f} above 0.92")

    def test_output_range(self):
        """3. All outputs in [0, 1] (sigmoid guarantee)."""
        x = torch.randn(8, 15, self.embed_dim) * 10  # large inputs
        with torch.no_grad():
            out = self.gate(x)
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_gradient_flow(self):
        """4. Gradients flow through gate params and back to input after weights become non-zero."""
        # At init, weights are zero so d(sigmoid(0*x+2))/dx = 0.
        # Verify that after one gradient step (weights become non-zero),
        # gradients flow back to the input.
        gate = InputDependentGate(self.embed_dim)
        optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)

        # Step 1: update weights so they are non-zero
        x1 = torch.randn(2, 5, self.embed_dim)
        loss1 = gate(x1).sum()
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2: now verify gradients flow to input
        x2 = torch.randn(2, 5, self.embed_dim, requires_grad=True)
        loss2 = gate(x2).sum()
        loss2.backward()
        self.assertIsNotNone(x2.grad)
        self.assertTrue(x2.grad.abs().sum() > 0,
                        "Gradients should flow to input after weights become non-zero")
        self.assertIsNotNone(gate.proj.weight.grad)


# ── 5-9: GatedTransformerBlock with input-dependent mode ───────────────────

class TestGatedTransformerBlockInputDependent(TimedTestCase):
    """Tests for GatedTransformerBlock in input_dependent gate mode."""

    def setUp(self):
        self.embed_dim = 64
        self.block = GatedTransformerBlock(
            embed_dim=self.embed_dim, num_heads=4, ffn_hidden_dim=128,
            dropout=0.0, gate_mode="input_dependent",
        )
        self.block.eval()

    def test_input_dependent_mode_forward(self):
        """5. Forward produces correct output shape in input-dependent mode."""
        x = torch.randn(2, 10, self.embed_dim)
        out = self.block(x)
        self.assertEqual(out.shape, (2, 10, self.embed_dim))

    def test_input_dependent_ignores_gate_signals(self):
        """6. Passing gate_signals has no effect in input-dependent mode."""
        x = torch.randn(2, 10, self.embed_dim)
        with torch.no_grad():
            out_no_signal = self.block(x)
            out_with_signal = self.block(x, gate_signals=torch.tensor([[0.1], [0.9]]))
        self.assertTrue(
            torch.allclose(out_no_signal, out_with_signal, atol=1e-6),
            "gate_signals should be ignored in input_dependent mode"
        )

    def test_external_mode_backward_compat(self):
        """7. External mode still works identically."""
        block_ext = GatedTransformerBlock(
            embed_dim=self.embed_dim, num_heads=4, ffn_hidden_dim=128,
            dropout=0.0, gate_mode="external",
        )
        block_ext.eval()
        x = torch.randn(2, 10, self.embed_dim)
        gate_signals = torch.tensor([[0.5], [0.8]])
        with torch.no_grad():
            out = block_ext(x, gate_signals=gate_signals)
        self.assertEqual(out.shape, (2, 10, self.embed_dim))

    def test_gate_activations_returned(self):
        """8. return_gate_activations=True returns correct structure."""
        x = torch.randn(2, 10, self.embed_dim)
        result = self.block(x, return_gate_activations=True)
        # Should be (output, (attn_gate_vals, ffn_gate_vals))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        output, (attn_gate, ffn_gate) = result
        self.assertEqual(output.shape, (2, 10, self.embed_dim))
        self.assertIsInstance(attn_gate, torch.Tensor)
        self.assertIsInstance(ffn_gate, torch.Tensor)

    def test_gate_activations_shape(self):
        """9. Gate activations are (batch, seq_len, 1) in input-dependent mode."""
        x = torch.randn(3, 8, self.embed_dim)
        _, (attn_gate, ffn_gate) = self.block(x, return_gate_activations=True)
        self.assertEqual(attn_gate.shape, (3, 8, 1))
        self.assertEqual(ffn_gate.shape, (3, 8, 1))


# ── 10-13: TransformerLM with input-dependent gates ────────────────────────

class TestTransformerLMInputDependent(TimedTestCase):
    """Tests for TransformerLM configured with input-dependent gates."""

    def setUp(self):
        self.model = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        self.model.to("cpu")

    def test_model_with_input_dependent_gates(self):
        """10. Full model forward pass works with input-dependent gates."""
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        logits, (loss, _), _ = self.model(input_ids, target_ids)
        self.assertEqual(logits.shape, (2, 10, VOCAB_SIZE))
        self.assertGreater(loss.item(), 0)

    def test_gate_activations_collected(self):
        """11. viz_data has gate_activations with one entry per layer."""
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        _, _, viz_data = self.model(input_ids, return_gate_activations=True)
        self.assertIn("gate_activations", viz_data)
        activations = viz_data["gate_activations"]
        self.assertEqual(len(activations), SMALL_CONFIG_INPUT_DEP["NUM_TRANSFORMER_BLOCKS"])
        for attn_g, ffn_g in activations:
            self.assertEqual(attn_g.shape, (2, 10, 1))
            self.assertEqual(ffn_g.shape, (2, 10, 1))

    def test_gradient_flow_through_gates(self):
        """12. Gate parameters receive gradients during training."""
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        self.model.train()
        _, (loss, _), _ = self.model(input_ids, target_ids)
        loss.backward()
        block = self.model.transformer_blocks[0]
        self.assertIsNotNone(block.attn_gate.proj.weight.grad)
        self.assertTrue(block.attn_gate.proj.weight.grad.abs().sum() > 0)

    def test_backward_compat_external_mode(self):
        """13. Model with GATE_MODE external works as before."""
        model_ext = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_EXTERNAL)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        gate_signals = torch.tensor([[0.5], [0.8]])
        logits, (loss, _), _ = model_ext(input_ids, target_ids, gate_signals=gate_signals)
        self.assertEqual(logits.shape, (2, 10, VOCAB_SIZE))
        self.assertGreater(loss.item(), 0)


# ── 14-15: Checkpoint compatibility ────────────────────────────────────────

class TestCheckpointCompatibility(TimedTestCase):
    """Tests for checkpoint cross-mode and same-mode loading."""

    def test_cross_mode_checkpoint_fails_clearly(self):
        """14. Loading external checkpoint into input_dependent model raises RuntimeError."""
        model_ext = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_EXTERNAL)
        sd_ext = get_model_state_dict(model_ext)

        model_idp = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        with self.assertRaises(RuntimeError):
            load_model_state_dict(model_idp, sd_ext)

        # And vice versa
        sd_idp = get_model_state_dict(model_idp)
        model_ext2 = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_EXTERNAL)
        with self.assertRaises(RuntimeError):
            load_model_state_dict(model_ext2, sd_idp)

    def test_same_mode_checkpoint_roundtrip(self):
        """15. Save/load input_dependent checkpoint works, all keys match."""
        model1 = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        sd1 = get_model_state_dict(model1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            torch.save(sd1, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        model2 = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        load_model_state_dict(model2, loaded)
        sd2 = get_model_state_dict(model2)

        self.assertEqual(set(sd1.keys()), set(sd2.keys()))
        for key in sd1:
            self.assertTrue(
                torch.equal(sd1[key], sd2[key]),
                f"Mismatch in {key} after roundtrip",
            )


# ── 16-17: Fixed-gate baseline (Model B) ──────────────────────────────────

class TestFixedGateBaseline(TimedTestCase):
    """Tests for frozen gates (Model B: constant output near 0.88)."""

    def _make_frozen_model(self):
        model = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        for block in model.transformer_blocks:
            for param in block.attn_gate.parameters():
                param.requires_grad = False
            for param in block.ffn_gate.parameters():
                param.requires_grad = False
        return model

    def test_frozen_gates_produce_constant_output(self):
        """16. Frozen gates give same value near 0.88 regardless of input."""
        model = self._make_frozen_model()
        model.eval()
        x1 = torch.randint(0, VOCAB_SIZE, (2, 10))
        x2 = torch.randint(0, VOCAB_SIZE, (2, 10))
        with torch.no_grad():
            _, _, viz1 = model(x1, return_gate_activations=True)
            _, _, viz2 = model(x2, return_gate_activations=True)
        for (a1, f1), (a2, f2) in zip(viz1["gate_activations"], viz2["gate_activations"]):
            # All values should be close to sigmoid(2.0) which is about 0.88
            self.assertTrue(torch.allclose(a1, a2, atol=1e-5))
            self.assertAlmostEqual(a1.mean().item(), 0.88, places=1)

    def test_frozen_gates_no_gradient(self):
        """17. Frozen gate parameters have requires_grad=False, no grads after backward."""
        model = self._make_frozen_model()
        model.train()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        _, (loss, _), _ = model(input_ids, target_ids)
        loss.backward()
        for block in model.transformer_blocks:
            for param in block.attn_gate.parameters():
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad)
            for param in block.ffn_gate.parameters():
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad)


# ── 18-19: Gate regularization in Trainer ──────────────────────────────────

class TestGateRegularization(TimedTestCase):
    """Tests for gate sparsity regularization loss computation."""

    def test_gate_reg_loss_computed(self):
        """18. Regularization loss is non-zero and retains gradient connectivity."""
        model = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        model.train()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        _, (ce_loss, _), viz_data = model(input_ids, target_ids, return_gate_activations=True)

        gate_activations = viz_data["gate_activations"]
        gate_loss = sum(
            g.mean() for pair in gate_activations for g in pair
        ) / (2 * len(gate_activations))

        self.assertGreater(gate_loss.item(), 0.0,
                           "Gate reg loss should be > 0 with active gates")

        # gate_loss must retain grad connectivity so regularization actually works
        self.assertTrue(gate_loss.requires_grad,
                        "gate_loss must require grad (not detached) for regularization to backpropagate")

    def test_gate_reg_zero_when_disabled(self):
        """19. No regularization when GATE_REG_WEIGHT is 0.0."""
        gate_reg_weight = 0.0
        model = TransformerLM(vocab_size=VOCAB_SIZE, config=SMALL_CONFIG_INPUT_DEP)
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        _, (ce_loss, _), viz_data = model(input_ids, target_ids, return_gate_activations=True)

        gate_activations = viz_data["gate_activations"]
        gate_loss = sum(
            g.mean() for pair in gate_activations for g in pair
        ) / (2 * len(gate_activations))

        total_loss = ce_loss + gate_reg_weight * gate_loss
        # With weight=0, total should equal ce_loss exactly
        self.assertAlmostEqual(total_loss.item(), ce_loss.item(), places=6)

    def test_gate_reg_requires_input_dependent_mode(self):
        """27. GATE_REG_WEIGHT > 0 with GATE_MODE external raises ValueError."""
        from plugins.tidal.Trainer import Trainer

        bad_config = {
            **SMALL_CONFIG_EXTERNAL,
            "GATE_REG_WEIGHT": 0.01,
            "GATE_MODE": "external",
            "BATCH_SIZE": 4,
            "NUM_EPOCHS": 1,
            "PATIENCE": 5,
            "MIN_DELTA": 0.0001,
            "MAX_GRAD_NORM": 1.0,
        }
        with self.assertRaises(ValueError) as ctx:
            Trainer(bad_config, "/tmp/test_exp")
        self.assertIn("GATE_MODE", str(ctx.exception))


# ── 20-22: Gate analysis ──────────────────────────────────────────────────

class TestGateAnalysis(TimedTestCase):
    """Tests for gate activation analysis metrics."""

    def _make_fake_activations(self, pattern="varied"):
        """Create fake gate activations for testing analysis logic."""
        num_layers = 2
        batch, seq_len = 4, 10
        activations = []
        for layer in range(num_layers):
            if pattern == "varied":
                attn = torch.rand(batch, seq_len, 1) * 0.5 + 0.25
                ffn = torch.rand(batch, seq_len, 1) * 0.5 + 0.25
                # Inject some sparse values
                attn[:, :3, :] = 0.05
            elif pattern == "constant":
                attn = torch.full((batch, seq_len, 1), 0.88)
                ffn = torch.full((batch, seq_len, 1), 0.88)
            activations.append((attn, ffn))
        return activations

    def test_gate_analysis_output_structure(self):
        """20. Analysis returns dict with expected keys."""
        from plugins.tidal.evaluate_hypothesis import compute_gate_metrics
        activations = self._make_fake_activations()
        result = compute_gate_metrics(activations)
        self.assertIn("per_gate_stats", result)
        self.assertIn("sparsity_fraction", result)
        self.assertIn("mean_cov", result)
        self.assertEqual(len(result["per_gate_stats"]), 4)  # 2 layers x 2 gates

    def test_sparsity_fraction_range(self):
        """21. Sparsity fraction in [0, 1]."""
        from plugins.tidal.evaluate_hypothesis import compute_gate_metrics
        for pattern in ["varied", "constant"]:
            activations = self._make_fake_activations(pattern)
            result = compute_gate_metrics(activations)
            self.assertGreaterEqual(result["sparsity_fraction"], 0.0)
            self.assertLessEqual(result["sparsity_fraction"], 1.0)

    def test_cov_computation(self):
        """22. CoV correct: constant gates give CoV near 0, varied gives CoV > 0."""
        from plugins.tidal.evaluate_hypothesis import compute_gate_metrics
        constant = self._make_fake_activations("constant")
        result_const = compute_gate_metrics(constant)
        self.assertAlmostEqual(result_const["mean_cov"], 0.0, places=4,
                               msg="Constant gates should have CoV near 0")

        varied = self._make_fake_activations("varied")
        result_var = compute_gate_metrics(varied)
        self.assertGreater(result_var["mean_cov"], 0.0,
                           "Varied gates should have CoV > 0")


# ── 23-26: Hypothesis assessment ──────────────────────────────────────────

class TestHypothesisAssessment(TimedTestCase):
    """Tests for the H1/H0 decision logic."""

    def test_h1_all_pass(self):
        """23. Returns H1 when all conditions met."""
        from plugins.tidal.evaluate_hypothesis import assess_hypothesis
        result = assess_hypothesis(
            ppl_gated=10.0, ppl_ungated=10.0,
            sparsity_fraction=0.15, mean_cov=0.30,
        )
        self.assertTrue(result["h1"])
        self.assertTrue(result["c1"])
        self.assertTrue(result["c2"])
        self.assertTrue(result["c3"])

    def test_h0_quality_fail(self):
        """24. Returns H0 when C1 fails (quality degraded > 5%)."""
        from plugins.tidal.evaluate_hypothesis import assess_hypothesis
        result = assess_hypothesis(
            ppl_gated=11.0, ppl_ungated=10.0,  # ratio = 1.10 > 1.05
            sparsity_fraction=0.15, mean_cov=0.30,
        )
        self.assertFalse(result["h1"])
        self.assertFalse(result["c1"])
        self.assertTrue(result["c2"])
        self.assertTrue(result["c3"])

    def test_h0_sparsity_fail(self):
        """25. Returns H0 when C2 fails (insufficient sparsity)."""
        from plugins.tidal.evaluate_hypothesis import assess_hypothesis
        result = assess_hypothesis(
            ppl_gated=10.0, ppl_ungated=10.0,
            sparsity_fraction=0.05,  # < 0.10
            mean_cov=0.30,
        )
        self.assertFalse(result["h1"])
        self.assertTrue(result["c1"])
        self.assertFalse(result["c2"])
        self.assertTrue(result["c3"])

    def test_h0_adaptivity_fail(self):
        """26. Returns H0 when C3 fails (gates not input-dependent)."""
        from plugins.tidal.evaluate_hypothesis import assess_hypothesis
        result = assess_hypothesis(
            ppl_gated=10.0, ppl_ungated=10.0,
            sparsity_fraction=0.15,
            mean_cov=0.15,  # <= 0.20
        )
        self.assertFalse(result["h1"])
        self.assertTrue(result["c1"])
        self.assertTrue(result["c2"])
        self.assertFalse(result["c3"])


if __name__ == "__main__":
    unittest.main()
