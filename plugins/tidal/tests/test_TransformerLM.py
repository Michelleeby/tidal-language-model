import os
import unittest
import torch

import tempfile

from plugins.tidal.tests.timeout import TimedTestCase
from plugins.tidal.TransformerLM import (
    TransformerLM, GatedTransformerBlock, DynamicGate,
    get_model_state_dict, load_model_state_dict,
    precompute_rope_frequencies, apply_rope,
)
from plugins.tidal.Trainer import Trainer


class TestTransformerLM(TimedTestCase):
    """Unit tests for the TransformerLM class."""

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
        }

    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 100
        os.makedirs(self.config.get("LOG_DIRECTORY", "logs"), exist_ok=True)
        self.model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        self.model.to(self.device)

    def test_initialization(self):
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.embed_dim, 64)
        self.assertEqual(self.model.num_transformer_blocks, 2)

        self.assertEqual(self.model.token_embeddings.weight.shape, (self.vocab_size, 64))
        self.assertEqual(self.model.rope_cos.shape, (32, 16))  # (max_ctx, head_dim=64/4)
        self.assertEqual(self.model.rope_sin.shape, (32, 16))
        self.assertEqual(self.model.output_projection.weight.shape, (self.vocab_size, 64))

    def test_forward_pass_output_shapes(self):
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        logits, (total_loss, loss_components), viz_data = self.model(input_ids, target_ids)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertEqual(total_loss.numel(), 1)
        self.assertIsNone(loss_components)
        self.assertIsInstance(viz_data, dict)
        self.assertEqual(len(viz_data), 0)

    def test_forward_with_gate_signals(self):
        """Test forward pass with gate signals."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        gate_signals = torch.tensor([[0.5], [0.8]], device=self.device)

        logits, (loss, _), _ = self.model(input_ids, target_ids, gate_signals=gate_signals)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertGreater(loss.item(), 0)

    def test_eval_mode_loss(self):
        """Test that loss is computed even in eval mode (the bug fix)."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        self.model.eval()
        with torch.no_grad():
            _, (loss, _), _ = self.model(input_ids, target_ids)

        self.assertGreater(loss.item(), 0, "Loss should be non-zero in eval mode when targets provided")

    def test_forward_pass_without_targets(self):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, (total_loss, _), _ = self.model(input_ids, target_ids=None)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertEqual(total_loss.item(), 0.0)

    def test_generate_method(self):
        prompt_ids = torch.tensor([1, 5, 10], device=self.device)
        max_new_tokens = 5

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_k=10,
        )

        self.assertIsInstance(generated_ids, list)
        self.assertEqual(len(generated_ids), len(prompt_ids) + max_new_tokens)

        for token_id in generated_ids:
            self.assertTrue(0 <= token_id < self.vocab_size)

    def test_gradient_flow(self):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        self.model.train()
        logits, (loss, _), _ = self.model(input_ids, target_ids)
        loss.backward()

        self.assertIsNotNone(self.model.token_embeddings.weight.grad)
        self.assertIsNotNone(self.model.output_projection.weight.grad)

        has_nonzero_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        self.assertTrue(has_nonzero_grad)

    def test_gradient_flow_with_gating(self):
        """Test that gradients flow through gate parameters."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        gate_signals = torch.tensor([[0.7]] * batch_size, device=self.device, requires_grad=True)

        self.model.train()
        logits, (loss, _), _ = self.model(input_ids, target_ids, gate_signals=gate_signals)
        loss.backward()

        # Check that gate module parameters have gradients
        block = self.model.transformer_blocks[0]
        attn_gate_grad = block.attn_gate.net[-1].weight.grad
        self.assertIsNotNone(attn_gate_grad)


class TestDynamicGate(TimedTestCase):
    """Tests for the DynamicGate module."""

    def test_neutral_initialization(self):
        """Gate should produce ~1.0 outputs at initialization."""
        gate = DynamicGate(gate_dim=1, embed_dim=64)
        signals = torch.tensor([[0.5]])
        output = gate(signals)

        # Should be close to 1.0 due to bias initialization
        self.assertTrue(output.mean().item() > 0.8)

    def test_none_gate_signals(self):
        """Gate should return 1.0 when gate_signals is None."""
        gate = DynamicGate(gate_dim=1, embed_dim=64)
        output = gate(None)
        self.assertEqual(output, 1.0)

    def test_output_shape(self):
        gate = DynamicGate(gate_dim=1, embed_dim=64)
        signals = torch.randn(4, 1)
        output = gate(signals)
        self.assertEqual(output.shape, (4, 1, 64))


class TestGatedTransformerBlock(TimedTestCase):
    """Tests for the GatedTransformerBlock."""

    def test_forward_without_gating(self):
        block = GatedTransformerBlock(embed_dim=64, num_heads=4, ffn_hidden_dim=128)
        x = torch.randn(2, 10, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_forward_with_gating(self):
        block = GatedTransformerBlock(embed_dim=64, num_heads=4, ffn_hidden_dim=128)
        x = torch.randn(2, 10, 64)
        gate_signals = torch.tensor([[0.5], [0.8]])
        out = block(x, gate_signals)
        self.assertEqual(out.shape, (2, 10, 64))


class TestTrainerLogThread(TimedTestCase):
    """Tests for Trainer log thread behavior."""

    def test_log_thread_is_not_daemon(self):
        """Log thread should not be daemon so pending metrics flush on exit."""
        config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.1,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
            "BATCH_SIZE": 4,
            "NUM_EPOCHS": 1,
            "LOG_DIRECTORY": "logs",
            "ENABLE_CONSOLE_LOGGING": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, tmpdir)
            self.assertFalse(trainer._log_thread.daemon)
            trainer._flush_logs()


class TestGatedTransformerBlockKVCache(TimedTestCase):
    """Tests for GatedTransformerBlock with manual Q/K/V projections and KV cache."""

    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads
        self.block = GatedTransformerBlock(
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            ffn_hidden_dim=128, dropout=0.0, max_seq_len=32,
        )
        self.block.eval()

    def test_block_has_separate_projections(self):
        """Block should have q_proj, k_proj, v_proj, out_proj; no 'attention' attribute."""
        self.assertTrue(hasattr(self.block, 'q_proj'))
        self.assertTrue(hasattr(self.block, 'k_proj'))
        self.assertTrue(hasattr(self.block, 'v_proj'))
        self.assertTrue(hasattr(self.block, 'out_proj'))
        self.assertFalse(hasattr(self.block, 'attention'))

    def test_forward_without_cache_returns_tensor(self):
        """Default forward returns a single tensor (backward-compatible)."""
        x = torch.randn(2, 10, self.embed_dim)
        cos, sin = precompute_rope_frequencies(self.head_dim, 32)
        out = self.block(x, rope_cos=cos[:10], rope_sin=sin[:10])
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, 10, self.embed_dim))

    def test_forward_with_cache_returns_tuple(self):
        """use_cache=True returns (tensor, (k, v))."""
        x = torch.randn(2, 10, self.embed_dim)
        cos, sin = precompute_rope_frequencies(self.head_dim, 32)
        result = self.block(x, rope_cos=cos[:10], rope_sin=sin[:10], use_cache=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        tensor_out, (cached_k, cached_v) = result
        self.assertEqual(tensor_out.shape, (2, 10, self.embed_dim))
        self.assertEqual(cached_k.shape[0], 2)  # batch
        self.assertEqual(cached_k.shape[2], 10)  # seq_len
        self.assertEqual(cached_k.shape[3], self.head_dim)

    def test_cached_generation_matches_full_forward(self):
        """Last-token output from cached pass matches full pass."""
        seq_len = 10
        x = torch.randn(1, seq_len, self.embed_dim)
        cos, sin = precompute_rope_frequencies(self.head_dim, 32)

        with torch.no_grad():
            # Full forward
            full_out = self.block(x, rope_cos=cos[:seq_len], rope_sin=sin[:seq_len])

            # Prefill: first seq_len-1 tokens with cache
            prefill_out, (cached_k, cached_v) = self.block(
                x[:, :-1, :], rope_cos=cos[:seq_len-1], rope_sin=sin[:seq_len-1], use_cache=True
            )

            # Decode: last token only, using cached KV
            decode_out, _ = self.block(
                x[:, -1:, :], rope_cos=cos[seq_len-1:seq_len], rope_sin=sin[seq_len-1:seq_len],
                layer_past=(cached_k, cached_v), use_cache=True
            )

        self.assertTrue(
            torch.allclose(full_out[:, -1:, :], decode_out, atol=1e-4),
            f"Max diff: {(full_out[:, -1:, :] - decode_out).abs().max().item()}"
        )

    def test_forward_with_gating_and_cache(self):
        """Cache works alongside gate_signals."""
        x = torch.randn(2, 10, self.embed_dim)
        cos, sin = precompute_rope_frequencies(self.head_dim, 32)
        gate_signals = torch.tensor([[0.5], [0.8]])

        result = self.block(
            x, gate_signals=gate_signals,
            rope_cos=cos[:10], rope_sin=sin[:10], use_cache=True,
        )
        tensor_out, (cached_k, cached_v) = result
        self.assertEqual(tensor_out.shape, (2, 10, self.embed_dim))

    def test_layer_past_extends_kv(self):
        """present KV seq_len = past_len + current_len."""
        cos, sin = precompute_rope_frequencies(self.head_dim, 32)

        x1 = torch.randn(1, 5, self.embed_dim)
        _, (k1, v1) = self.block(x1, rope_cos=cos[:5], rope_sin=sin[:5], use_cache=True)
        self.assertEqual(k1.shape[2], 5)

        x2 = torch.randn(1, 3, self.embed_dim)
        _, (k2, v2) = self.block(
            x2, rope_cos=cos[5:8], rope_sin=sin[5:8],
            layer_past=(k1, v1), use_cache=True,
        )
        self.assertEqual(k2.shape[2], 8)  # 5 + 3


class TestRoPE(TimedTestCase):
    """Tests for Rotary Positional Embedding helper functions."""

    def test_precompute_rope_frequencies_shape(self):
        """Returns (cos, sin) each of shape (max_seq_len, head_dim)."""
        head_dim = 16
        max_seq_len = 64
        cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
        self.assertEqual(cos.shape, (max_seq_len, head_dim))
        self.assertEqual(sin.shape, (max_seq_len, head_dim))

    def test_precompute_rope_frequencies_unit_norm(self):
        """cos^2 + sin^2 = 1 for all positions and dimensions."""
        cos, sin = precompute_rope_frequencies(head_dim=32, max_seq_len=128)
        norm_sq = cos ** 2 + sin ** 2
        self.assertTrue(torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5))

    def test_apply_rope_shape_preserved(self):
        """Output shape matches input (batch, heads, seq, head_dim)."""
        batch, heads, seq_len, head_dim = 2, 4, 10, 16
        x = torch.randn(batch, heads, seq_len, head_dim)
        cos, sin = precompute_rope_frequencies(head_dim, max_seq_len=32)
        cos_slice = cos[:seq_len]  # (seq_len, head_dim)
        sin_slice = sin[:seq_len]
        out = apply_rope(x, cos_slice, sin_slice)
        self.assertEqual(out.shape, x.shape)

    def test_apply_rope_equivariance(self):
        """Dot product between RoPE-encoded vectors depends on relative position."""
        head_dim = 16
        cos, sin = precompute_rope_frequencies(head_dim, max_seq_len=64)

        q = torch.randn(1, 1, 1, head_dim).expand(1, 1, 3, head_dim).clone()
        k = q.clone()

        # Apply RoPE at positions [0,1,2]
        q_rot = apply_rope(q, cos[:3], sin[:3])
        k_rot = apply_rope(k, cos[:3], sin[:3])

        # dot(q_pos0, k_pos1) should equal dot(q_pos1, k_pos2) (same relative dist)
        dot_01 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
        dot_12 = (q_rot[0, 0, 1] * k_rot[0, 0, 2]).sum()
        self.assertTrue(torch.allclose(dot_01, dot_12, atol=1e-5))

    def test_apply_rope_different_relative_positions_differ(self):
        """Different relative distances produce different dot products."""
        head_dim = 16
        cos, sin = precompute_rope_frequencies(head_dim, max_seq_len=64)

        q = torch.randn(1, 1, 1, head_dim).expand(1, 1, 4, head_dim).clone()
        k = q.clone()

        q_rot = apply_rope(q, cos[:4], sin[:4])
        k_rot = apply_rope(k, cos[:4], sin[:4])

        # dot(pos0, pos1) — distance 1
        dot_dist1 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
        # dot(pos0, pos3) — distance 3
        dot_dist3 = (q_rot[0, 0, 0] * k_rot[0, 0, 3]).sum()
        self.assertFalse(torch.allclose(dot_dist1, dot_dist3, atol=1e-3))


class TestTransformerLMRoPE(TimedTestCase):
    """Tests for RoPE integration in TransformerLM."""

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
        }
        cls.vocab_size = 100

    def setUp(self):
        self.model = TransformerLM(vocab_size=self.vocab_size, config=self.config)

    def test_no_position_embeddings(self):
        """Model should not have position_embeddings attribute."""
        self.assertFalse(hasattr(self.model, 'position_embeddings'))

    def test_has_rope_buffers(self):
        """Model should have rope_cos and rope_sin of shape (max_context_length, head_dim)."""
        head_dim = self.config["EMBED_DIM"] // self.config["NUM_ATTENTION_HEADS"]
        max_len = self.config["MAX_CONTEXT_LENGTH"]
        self.assertTrue(hasattr(self.model, 'rope_cos'))
        self.assertTrue(hasattr(self.model, 'rope_sin'))
        self.assertEqual(self.model.rope_cos.shape, (max_len, head_dim))
        self.assertEqual(self.model.rope_sin.shape, (max_len, head_dim))

    def test_forward_still_returns_3_tuple(self):
        """Return signature unchanged: (logits, (loss, None), viz_data)."""
        input_ids = torch.randint(0, self.vocab_size, (2, 10))
        target_ids = torch.randint(0, self.vocab_size, (2, 10))
        result = self.model(input_ids, target_ids)
        self.assertEqual(len(result), 3)
        logits, (loss, components), viz_data = result
        self.assertEqual(logits.shape, (2, 10, self.vocab_size))
        self.assertIsNone(components)
        self.assertIsInstance(viz_data, dict)


class TestTransformerLMKVCache(TimedTestCase):
    """Tests for KV cache in TransformerLM."""

    @classmethod
    def setUpClass(cls):
        cls.config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.0,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
        }
        cls.vocab_size = 100

    def setUp(self):
        self.model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        self.model.eval()

    def test_forward_with_cache_returns_past_in_viz_data(self):
        """viz_data['past_key_values'] present when use_cache=True."""
        input_ids = torch.randint(0, self.vocab_size, (1, 10))
        with torch.no_grad():
            _, _, viz_data = self.model(input_ids, use_cache=True)
        self.assertIn("past_key_values", viz_data)
        past = viz_data["past_key_values"]
        self.assertEqual(len(past), self.config["NUM_TRANSFORMER_BLOCKS"])

    def test_forward_without_cache_no_past_in_viz_data(self):
        """No cache key when use_cache=False."""
        input_ids = torch.randint(0, self.vocab_size, (1, 10))
        with torch.no_grad():
            _, _, viz_data = self.model(input_ids)
        self.assertNotIn("past_key_values", viz_data)

    def test_cached_generation_matches_full_forward(self):
        """Cached single-token logits match full-context logits."""
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (1, seq_len))

        with torch.no_grad():
            # Full forward
            full_logits, _, _ = self.model(input_ids)

            # Prefill + single token decode
            _, _, viz_data = self.model(input_ids[:, :-1], use_cache=True)
            past = viz_data["past_key_values"]

            cached_logits, _, _ = self.model(
                input_ids[:, -1:], use_cache=True, past_key_values=past,
            )

        self.assertTrue(
            torch.allclose(full_logits[:, -1:, :], cached_logits, atol=1e-4),
            f"Max diff: {(full_logits[:, -1:, :] - cached_logits).abs().max().item()}"
        )

    def test_generate_produces_valid_output(self):
        """generate() still works correctly with KV cache."""
        prompt_ids = torch.tensor([1, 5, 10])
        generated = self.model.generate(prompt_ids, max_new_tokens=5, top_k=10)
        self.assertIsInstance(generated, list)
        self.assertEqual(len(generated), 8)  # 3 prompt + 5 generated
        for tid in generated:
            self.assertTrue(0 <= tid < self.vocab_size)


class TestTorchCompileForward(TimedTestCase):
    """Tests for torch.compile compatibility with forward pass."""

    @classmethod
    def setUpClass(cls):
        cls.config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.0,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
        }
        cls.vocab_size = 100

    def test_compiled_forward_without_gate_signals(self):
        """torch.compile'd model should handle gate_signals=None without KeyError."""
        model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        try:
            compiled = torch.compile(model)
        except Exception:
            self.skipTest("torch.compile not available")

        input_ids = torch.randint(0, self.vocab_size, (2, 10))
        target_ids = torch.randint(0, self.vocab_size, (2, 10))

        logits, (loss, _), _ = compiled(input_ids, target_ids)
        self.assertEqual(logits.shape, (2, 10, self.vocab_size))
        self.assertGreater(loss.item(), 0)

    def test_compiled_forward_with_gate_signals(self):
        """torch.compile'd model should handle explicit gate signals."""
        model = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        try:
            compiled = torch.compile(model)
        except Exception:
            self.skipTest("torch.compile not available")

        input_ids = torch.randint(0, self.vocab_size, (2, 10))
        target_ids = torch.randint(0, self.vocab_size, (2, 10))
        gate_signals = torch.tensor([[0.5], [0.8]])

        logits, (loss, _), _ = compiled(input_ids, target_ids, gate_signals=gate_signals)
        self.assertEqual(logits.shape, (2, 10, self.vocab_size))
        self.assertGreater(loss.item(), 0)


class TestModelStateDictHelpers(TimedTestCase):
    """Tests for get_model_state_dict / load_model_state_dict helpers."""

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
        }

    def test_get_state_dict_uncompiled(self):
        """Returns clean state_dict from uncompiled model."""
        model = TransformerLM(vocab_size=100, config=self.config)
        sd = get_model_state_dict(model)
        self.assertIsInstance(sd, dict)
        self.assertFalse(any(k.startswith("_orig_mod.") for k in sd))

    def test_get_state_dict_compiled(self):
        """Returns state_dict without _orig_mod prefix from compiled model."""
        model = TransformerLM(vocab_size=100, config=self.config)
        try:
            compiled = torch.compile(model)
        except Exception:
            self.skipTest("torch.compile not available")
        sd = get_model_state_dict(compiled)
        self.assertIsInstance(sd, dict)
        self.assertFalse(any(k.startswith("_orig_mod.") for k in sd))

    def test_load_state_dict_into_compiled(self):
        """Loads clean state_dict into compiled model."""
        model = TransformerLM(vocab_size=100, config=self.config)
        sd = model.state_dict()
        try:
            compiled = torch.compile(model)
        except Exception:
            self.skipTest("torch.compile not available")
        load_model_state_dict(compiled, sd)

    def test_roundtrip_save_load(self):
        """Save from compiled, load into uncompiled, keys match."""
        model1 = TransformerLM(vocab_size=100, config=self.config)
        try:
            compiled = torch.compile(model1)
        except Exception:
            self.skipTest("torch.compile not available")
        saved_sd = get_model_state_dict(compiled)

        model2 = TransformerLM(vocab_size=100, config=self.config)
        load_model_state_dict(model2, saved_sd)

        sd2 = get_model_state_dict(model2)
        self.assertEqual(set(saved_sd.keys()), set(sd2.keys()))


class TestCheckpointShapeContract(TimedTestCase):
    """Shape-contract tests that verify model state_dict shapes match expectations.

    These tests prevent the exact class of bugs that caused the [32, 3] vs [32, 1]
    gate dimension mismatch production failure.
    """

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
        }
        cls.vocab_size = 100

    def setUp(self):
        super().setUp()
        self.model = TransformerLM(vocab_size=self.vocab_size, config=self.config)

    def test_gate_weight_shapes(self):
        """Every DynamicGate first-linear weight must be [hidden, GATE_DIM=1]."""
        gate_dim = GatedTransformerBlock.GATE_DIM
        sd = self.model.state_dict()
        for key, tensor in sd.items():
            if ".attn_gate.net.0.weight" in key or ".ffn_gate.net.0.weight" in key:
                self.assertEqual(
                    tensor.shape[1], gate_dim,
                    f"{key} has input dim {tensor.shape[1]}, expected {gate_dim}",
                )

    def test_state_dict_roundtrip(self):
        """Save state_dict, load into fresh model — no errors, all keys match."""
        sd = self.model.state_dict()

        model2 = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        model2.load_state_dict(sd)

        sd2 = model2.state_dict()
        self.assertEqual(set(sd.keys()), set(sd2.keys()))
        for key in sd:
            self.assertTrue(
                torch.equal(sd[key], sd2[key]),
                f"Mismatch in {key} after roundtrip",
            )

    def test_old_3gate_checkpoint_fails_clearly(self):
        """A synthetic 3-gate state_dict must fail with 'size mismatch'."""
        sd = self.model.state_dict()

        # Replace gate input weights with old 3-gate format
        for key in list(sd.keys()):
            if ".attn_gate.net.0.weight" in key or ".ffn_gate.net.0.weight" in key:
                hidden_dim = sd[key].shape[0]
                sd[key] = torch.randn(hidden_dim, 3)  # old 3-gate format
            elif ".attn_gate.net.0.bias" in key or ".ffn_gate.net.0.bias" in key:
                pass  # bias shape is [hidden_dim], unchanged

        model2 = TransformerLM(vocab_size=self.vocab_size, config=self.config)
        with self.assertRaises(RuntimeError) as ctx:
            model2.load_state_dict(sd, strict=True)
        self.assertIn("size mismatch", str(ctx.exception))

    def test_checkpoint_format_detection(self):
        """Both raw state_dict and wrapped dict formats load correctly."""
        sd = self.model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Format 1: raw state_dict (what Trainer saves)
            raw_path = os.path.join(tmpdir, "raw.pth")
            torch.save(sd, raw_path)

            loaded_raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            model_raw = TransformerLM(vocab_size=self.vocab_size, config=self.config)
            model_raw.load_state_dict(loaded_raw)

            # Format 2: wrapped dict (what some tools save)
            wrapped_path = os.path.join(tmpdir, "wrapped.pth")
            torch.save({"model_state_dict": sd, "epoch": 1}, wrapped_path)

            loaded_wrapped = torch.load(wrapped_path, map_location="cpu", weights_only=True)
            if "model_state_dict" in loaded_wrapped:
                loaded_wrapped = loaded_wrapped["model_state_dict"]
            model_wrapped = TransformerLM(vocab_size=self.vocab_size, config=self.config)
            model_wrapped.load_state_dict(loaded_wrapped)

            # Both should produce identical state dicts
            sd_raw = model_raw.state_dict()
            sd_wrapped = model_wrapped.state_dict()
            self.assertEqual(set(sd_raw.keys()), set(sd_wrapped.keys()))

    def test_gate_dim_matches_class_constant(self):
        """GATE_DIM class constant is 1, and model gates use it."""
        self.assertEqual(GatedTransformerBlock.GATE_DIM, 1)

        for block in self.model.transformer_blocks:
            if isinstance(block, GatedTransformerBlock):
                # First linear layer input dim should match GATE_DIM
                attn_in_features = block.attn_gate.net[0].in_features
                ffn_in_features = block.ffn_gate.net[0].in_features
                self.assertEqual(attn_in_features, GatedTransformerBlock.GATE_DIM)
                self.assertEqual(ffn_in_features, GatedTransformerBlock.GATE_DIM)


if __name__ == "__main__":
    unittest.main()
