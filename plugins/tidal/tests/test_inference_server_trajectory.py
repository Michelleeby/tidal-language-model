"""
test_inference_server_trajectory.py

Tests that the inference server returns trajectory data for gated generation.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

import torch

from plugins.tidal.TransformerLM import TransformerLM
from plugins.tidal.GatingModulator import GatingModulator, GatingEffects


class TestInferenceServerTrajectory(unittest.TestCase):
    """Tests for trajectory serialization in inference server responses."""

    @classmethod
    def setUpClass(cls):
        cls.model_config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.0,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
            "VOCAB_SIZE": 100,
            "RL_BASE_TEMPERATURE": 1.0,
            "RL_BASE_REPETITION_PENALTY": 1.2,
            "RL_CREATIVITY_TEMP_MIN": 0.3,
            "RL_CREATIVITY_TEMP_MAX": 2.0,
            "RL_STABILITY_PENALTY_MIN": 1.0,
            "RL_STABILITY_PENALTY_MAX": 3.5,
            "RL_FOCUS_TOP_K_MIN": 5,
            "RL_FOCUS_TOP_K_MAX": 100,
            "RL_CREATIVITY_TOP_P_MIN": 0.7,
            "RL_CREATIVITY_TOP_P_MAX": 1.0,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
        }
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()

    def _get_app(self):
        """Create a test client with mocked model and config."""
        from plugins.tidal.inference_server import app
        return app.test_client()

    def _mock_checkpoint(self, tmp_path="/tmp/test_ckpt.pth"):
        """Save a real checkpoint for test use."""
        torch.save(self.model.state_dict(), tmp_path)
        return tmp_path

    @patch("plugins.tidal.inference_server.os.path.exists", return_value=True)
    @patch("plugins.tidal.inference_server._get_config")
    @patch("plugins.tidal.inference_server._get_model")
    @patch("plugins.tidal.inference_server._get_tokenizer")
    def test_gating_response_contains_trajectory(self, mock_tokenizer, mock_model, mock_config, _):
        """POST /generate with gatingMode=fixed returns trajectory key."""
        mock_config.return_value = self.model_config
        mock_model.return_value = self.model

        # Mock tokenizer
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)
        tok.vocab_size = 100
        mock_tokenizer.return_value = tok

        client = self._get_app()
        resp = client.post("/generate", json={
            "checkpoint": "/tmp/fake.pth",
            "prompt": "hello",
            "maxTokens": 5,
            "gatingMode": "fixed",
        })

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("trajectory", data)
        self.assertIsNotNone(data["trajectory"])

    @patch("plugins.tidal.inference_server.os.path.exists", return_value=True)
    @patch("plugins.tidal.inference_server._get_config")
    @patch("plugins.tidal.inference_server._get_model")
    @patch("plugins.tidal.inference_server._get_tokenizer")
    def test_trajectory_has_expected_shape(self, mock_tokenizer, mock_model, mock_config, _):
        """Trajectory has gateSignals (list of 3-float arrays), effects, tokenTexts."""
        mock_config.return_value = self.model_config
        mock_model.return_value = self.model

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids: "tok"
        tok.vocab_size = 100
        mock_tokenizer.return_value = tok

        client = self._get_app()
        resp = client.post("/generate", json={
            "checkpoint": "/tmp/fake.pth",
            "prompt": "hello",
            "maxTokens": 5,
            "gatingMode": "fixed",
        })

        data = json.loads(resp.data)
        traj = data["trajectory"]

        # gateSignals is a list of 3-element arrays
        self.assertIn("gateSignals", traj)
        self.assertEqual(len(traj["gateSignals"]), 5)
        for sig in traj["gateSignals"]:
            self.assertEqual(len(sig), 3)
            for v in sig:
                self.assertIsInstance(v, float)

        # effects is a list of dicts
        self.assertIn("effects", traj)
        self.assertEqual(len(traj["effects"]), 5)
        for eff in traj["effects"]:
            self.assertIn("temperature", eff)
            self.assertIn("repetition_penalty", eff)
            self.assertIn("top_k", eff)
            self.assertIn("top_p", eff)

        # tokenTexts is a list of strings
        self.assertIn("tokenTexts", traj)
        self.assertEqual(len(traj["tokenTexts"]), 5)
        for t in traj["tokenTexts"]:
            self.assertIsInstance(t, str)

        # tokenIds is a list of ints
        self.assertIn("tokenIds", traj)
        self.assertEqual(len(traj["tokenIds"]), 5)

    @patch("plugins.tidal.inference_server.os.path.exists", return_value=True)
    @patch("plugins.tidal.inference_server._get_config")
    @patch("plugins.tidal.inference_server._get_model")
    @patch("plugins.tidal.inference_server._get_tokenizer")
    def test_trajectory_excludes_heavy_data(self, mock_tokenizer, mock_model, mock_config, _):
        """No observations, logits_history, hidden_states in trajectory."""
        mock_config.return_value = self.model_config
        mock_model.return_value = self.model

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids: "tok"
        tok.vocab_size = 100
        mock_tokenizer.return_value = tok

        client = self._get_app()
        resp = client.post("/generate", json={
            "checkpoint": "/tmp/fake.pth",
            "prompt": "hello",
            "maxTokens": 5,
            "gatingMode": "fixed",
        })

        data = json.loads(resp.data)
        traj = data["trajectory"]
        self.assertNotIn("observations", traj)
        self.assertNotIn("logits_history", traj)
        self.assertNotIn("hidden_states", traj)

    @patch("plugins.tidal.inference_server.os.path.exists", return_value=True)
    @patch("plugins.tidal.inference_server._get_config")
    @patch("plugins.tidal.inference_server._get_model")
    @patch("plugins.tidal.inference_server._get_tokenizer")
    def test_no_gating_response_has_no_trajectory(self, mock_tokenizer, mock_model, mock_config, _):
        """gatingMode=none returns no trajectory key."""
        mock_config.return_value = self.model_config
        mock_model.return_value = self.model

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)
        tok.vocab_size = 100
        mock_tokenizer.return_value = tok

        client = self._get_app()
        resp = client.post("/generate", json={
            "checkpoint": "/tmp/fake.pth",
            "prompt": "hello",
            "maxTokens": 5,
            "gatingMode": "none",
        })

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertNotIn("trajectory", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
