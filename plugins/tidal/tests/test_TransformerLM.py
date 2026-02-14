import os
import unittest
import torch

import tempfile

from plugins.tidal.TransformerLM import (
    TransformerLM, GatedTransformerBlock, DynamicGate,
    get_model_state_dict, load_model_state_dict,
)
from plugins.tidal.Trainer import Trainer


class TestTransformerLM(unittest.TestCase):
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
        self.assertEqual(self.model.position_embeddings.weight.shape, (32, 64))
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
        gate_signals = torch.tensor([[0.5, 0.5, 0.5], [0.8, 0.2, 0.9]], device=self.device)

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
        gate_signals = torch.tensor([[0.7, 0.3, 0.5]] * batch_size, device=self.device, requires_grad=True)

        self.model.train()
        logits, (loss, _), _ = self.model(input_ids, target_ids, gate_signals=gate_signals)
        loss.backward()

        # Check that gate module parameters have gradients
        block = self.model.transformer_blocks[0]
        attn_gate_grad = block.attn_gate.net[-1].weight.grad
        self.assertIsNotNone(attn_gate_grad)


class TestDynamicGate(unittest.TestCase):
    """Tests for the DynamicGate module."""

    def test_neutral_initialization(self):
        """Gate should produce ~1.0 outputs at initialization."""
        gate = DynamicGate(gate_dim=3, embed_dim=64)
        signals = torch.tensor([[0.5, 0.5, 0.5]])
        output = gate(signals)

        # Should be close to 1.0 due to bias initialization
        self.assertTrue(output.mean().item() > 0.8)

    def test_none_gate_signals(self):
        """Gate should return 1.0 when gate_signals is None."""
        gate = DynamicGate(gate_dim=3, embed_dim=64)
        output = gate(None)
        self.assertEqual(output, 1.0)

    def test_output_shape(self):
        gate = DynamicGate(gate_dim=3, embed_dim=64)
        signals = torch.randn(4, 3)
        output = gate(signals)
        self.assertEqual(output.shape, (4, 1, 64))


class TestGatedTransformerBlock(unittest.TestCase):
    """Tests for the GatedTransformerBlock."""

    def test_forward_without_gating(self):
        block = GatedTransformerBlock(embed_dim=64, num_heads=4, ffn_hidden_dim=128)
        x = torch.randn(2, 10, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_forward_with_gating(self):
        block = GatedTransformerBlock(embed_dim=64, num_heads=4, ffn_hidden_dim=128)
        x = torch.randn(2, 10, 64)
        gate_signals = torch.tensor([[0.5, 0.5, 0.5], [0.8, 0.2, 0.9]])
        out = block(x, gate_signals)
        self.assertEqual(out.shape, (2, 10, 64))


class TestTrainerLogThread(unittest.TestCase):
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


class TestModelStateDictHelpers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
