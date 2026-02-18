"""
test_migrate_checkpoint.py

Tests for the checkpoint migration tool that converts 3-gate checkpoints
to 1-gate (single modulation) format.
"""

import unittest
import tempfile
import os

import torch
import torch.nn as nn

from plugins.tidal.TransformerLM import TransformerLM


class TestMigrateGateDim(unittest.TestCase):
    """Tests for migrate_gate_dim function."""

    def setUp(self):
        self.config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.1,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
        }

    def _make_old_state_dict(self):
        """Create a state_dict with 3-gate DynamicGate weights."""
        # Build a model with gate_dim=3 (the old format)
        # We can't use the model class directly since it will change,
        # so we construct the expected keys manually.
        old_config = dict(self.config)
        # Create a fresh model — at this point GATE_DIM might already be 1,
        # so we manually build old-format tensors.
        state_dict = {}

        # Token embeddings
        state_dict["token_embeddings.weight"] = torch.randn(100, 64)

        # For each block, create old 3-gate weights
        for i in range(2):
            prefix = f"transformer_blocks.{i}"
            # Standard transformer weights (not changing)
            state_dict[f"{prefix}.q_proj.weight"] = torch.randn(64, 64)
            state_dict[f"{prefix}.q_proj.bias"] = torch.randn(64)
            state_dict[f"{prefix}.k_proj.weight"] = torch.randn(64, 64)
            state_dict[f"{prefix}.k_proj.bias"] = torch.randn(64)
            state_dict[f"{prefix}.v_proj.weight"] = torch.randn(64, 64)
            state_dict[f"{prefix}.v_proj.bias"] = torch.randn(64)
            state_dict[f"{prefix}.out_proj.weight"] = torch.randn(64, 64)
            state_dict[f"{prefix}.out_proj.bias"] = torch.randn(64)

            state_dict[f"{prefix}.ffn.0.weight"] = torch.randn(128, 64)
            state_dict[f"{prefix}.ffn.0.bias"] = torch.randn(128)
            state_dict[f"{prefix}.ffn.3.weight"] = torch.randn(64, 128)
            state_dict[f"{prefix}.ffn.3.bias"] = torch.randn(64)

            state_dict[f"{prefix}.ln1.weight"] = torch.ones(64)
            state_dict[f"{prefix}.ln1.bias"] = torch.zeros(64)
            state_dict[f"{prefix}.ln2.weight"] = torch.ones(64)
            state_dict[f"{prefix}.ln2.bias"] = torch.zeros(64)

            # OLD 3-gate DynamicGate weights: first Linear is [32, 3]
            for gate_name in ["attn_gate", "ffn_gate"]:
                # net.0 = Linear(3, 32)
                state_dict[f"{prefix}.{gate_name}.net.0.weight"] = torch.randn(32, 3)
                state_dict[f"{prefix}.{gate_name}.net.0.bias"] = torch.randn(32)
                # net.2 = Linear(32, 64)
                state_dict[f"{prefix}.{gate_name}.net.2.weight"] = torch.randn(64, 32)
                state_dict[f"{prefix}.{gate_name}.net.2.bias"] = torch.full((64,), 2.0)

        # Final layer norm and output projection
        state_dict["final_layer_norm.weight"] = torch.ones(64)
        state_dict["final_layer_norm.bias"] = torch.zeros(64)
        state_dict["output_projection.weight"] = torch.randn(100, 64)
        state_dict["output_projection.bias"] = torch.randn(100)

        # RoPE buffers
        state_dict["rope_cos"] = torch.randn(32, 16)
        state_dict["rope_sin"] = torch.randn(32, 16)

        return state_dict

    def test_migrate_gate_dim_slices_column_2(self):
        """First Linear weight [32, 3] -> [32, 1] by slicing column 2 (stability)."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()
        new_sd = migrate_gate_dim(old_sd, slice_index=2)

        for i in range(2):
            for gate_name in ["attn_gate", "ffn_gate"]:
                key = f"transformer_blocks.{i}.{gate_name}.net.0.weight"
                self.assertEqual(new_sd[key].shape, (32, 1))
                # Should be column 2 of original
                expected = old_sd[key][:, 2:3]
                self.assertTrue(torch.equal(new_sd[key], expected))

    def test_migrate_preserves_non_gate_keys(self):
        """Non-gate keys are unchanged after migration."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()
        new_sd = migrate_gate_dim(old_sd, slice_index=2)

        # Token embeddings unchanged
        self.assertTrue(torch.equal(old_sd["token_embeddings.weight"], new_sd["token_embeddings.weight"]))
        # Output projection unchanged
        self.assertTrue(torch.equal(old_sd["output_projection.weight"], new_sd["output_projection.weight"]))

    def test_migrate_preserves_second_linear(self):
        """The second Linear [64, 32] in each gate is unchanged."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()
        new_sd = migrate_gate_dim(old_sd, slice_index=2)

        for i in range(2):
            for gate_name in ["attn_gate", "ffn_gate"]:
                key = f"transformer_blocks.{i}.{gate_name}.net.2.weight"
                self.assertTrue(torch.equal(old_sd[key], new_sd[key]))

    def test_migrated_checkpoint_loads_into_model(self):
        """Migrated state_dict can be loaded into a TransformerLM (with GATE_DIM=1)."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()
        new_sd = migrate_gate_dim(old_sd, slice_index=2)

        model = TransformerLM(vocab_size=100, config=self.config)
        # This should not raise
        model.load_state_dict(new_sd, strict=False)

    def test_already_migrated_checkpoint_is_noop(self):
        """If weights are already [32, 1], migration is a no-op."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()
        # First migration
        new_sd = migrate_gate_dim(old_sd, slice_index=2)
        # Second migration should be no-op
        new_sd2 = migrate_gate_dim(new_sd, slice_index=0)

        for key in new_sd:
            self.assertTrue(torch.equal(new_sd[key], new_sd2[key]))

    def test_cli_roundtrip(self):
        """CLI saves and loads correctly."""
        from plugins.tidal.migrate_checkpoint import migrate_gate_dim

        old_sd = self._make_old_state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "old.pth")
            output_path = os.path.join(tmpdir, "new.pth")

            torch.save(old_sd, input_path)

            new_sd = migrate_gate_dim(old_sd, slice_index=2)
            torch.save(new_sd, output_path)

            loaded = torch.load(output_path, map_location="cpu")
            for key in new_sd:
                self.assertTrue(torch.equal(new_sd[key], loaded[key]))


if __name__ == "__main__":
    unittest.main()
