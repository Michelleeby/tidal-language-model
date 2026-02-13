"""
test_GatingRL.py

Unit tests for the RL Gating Controller components.
"""

import torch
import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GatingModulator import GatingModulator, GatingEffects, RandomGatingPolicy, FixedGatingPolicy
from RewardComputer import RewardComputer
from GatingPolicyAgent import GatingPolicyAgent, GaussianGatingPolicyAgent, create_agent


class TestGatingModulator(unittest.TestCase):
    """Tests for GatingModulator class."""

    def setUp(self):
        self.config = {
            "RL_BASE_TEMPERATURE": 1.0,
            "RL_BASE_REPETITION_PENALTY": 1.2,
            "RL_CREATIVITY_TEMP_MIN": 0.5,
            "RL_CREATIVITY_TEMP_MAX": 1.5,
            "RL_STABILITY_PENALTY_MIN": 1.0,
            "RL_STABILITY_PENALTY_MAX": 2.5,
            "RL_FOCUS_ATTENTION_STRENGTH": 2.0,
        }
        self.modulator = GatingModulator(self.config)
        self.device = torch.device("cpu")

    def test_temperature_modulation(self):
        temp_low = self.modulator.compute_temperature(0.0)
        self.assertAlmostEqual(temp_low, 0.5, places=4)

        temp_high = self.modulator.compute_temperature(1.0)
        self.assertAlmostEqual(temp_high, 1.5, places=4)

        temp_mid = self.modulator.compute_temperature(0.5)
        self.assertAlmostEqual(temp_mid, 1.0, places=4)

    def test_repetition_penalty_modulation(self):
        penalty_low = self.modulator.compute_repetition_penalty(0.0)
        self.assertAlmostEqual(penalty_low, 1.0, places=4)

        penalty_high = self.modulator.compute_repetition_penalty(1.0)
        self.assertAlmostEqual(penalty_high, 2.5, places=4)

        penalty_mid = self.modulator.compute_repetition_penalty(0.5)
        self.assertAlmostEqual(penalty_mid, 1.75, places=4)

        self.assertGreater(penalty_high, penalty_low)

    def test_attention_bias_shape(self):
        bias = self.modulator.compute_attention_bias(0.5, 10, self.device)
        self.assertEqual(bias.shape, (10,))

    def test_forward_returns_effects(self):
        gate_signals = torch.tensor([0.5, 0.5, 0.5])
        effects = self.modulator(gate_signals, 20, self.device)

        self.assertIsInstance(effects, GatingEffects)
        self.assertIsInstance(effects.temperature, float)
        self.assertIsInstance(effects.repetition_penalty, float)
        self.assertIsInstance(effects.attention_bias, torch.Tensor)

    def test_baseline_policies(self):
        random_policy = RandomGatingPolicy(self.device)
        action = random_policy.get_action()
        self.assertEqual(action.shape, (3,))
        self.assertTrue(torch.all(action >= 0))
        self.assertTrue(torch.all(action <= 1))

        fixed_policy = FixedGatingPolicy(0.3, 0.6, 0.9, self.device)
        action = fixed_policy.get_action()
        self.assertEqual(action.shape, (3,))
        self.assertAlmostEqual(action[0].item(), 0.3, places=4)


class TestRewardComputer(unittest.TestCase):
    """Tests for RewardComputer class."""

    def setUp(self):
        self.config = {
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
        }
        self.vocab_size = 1000
        self.reward_computer = RewardComputer(self.config, self.vocab_size)

    def test_diversity_reward_high_diversity(self):
        tokens = list(range(20))
        diversity = self.reward_computer.compute_diversity_reward(tokens)
        self.assertGreater(diversity, 0.9)

    def test_diversity_reward_low_diversity(self):
        tokens = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        diversity = self.reward_computer.compute_diversity_reward(tokens)
        self.assertLess(diversity, 0.5)

    def test_repetition_penalty(self):
        tokens = [1, 2, 3, 4, 5]
        penalty_new = self.reward_computer.compute_repetition_penalty(tokens, 6)
        penalty_repeat = self.reward_computer.compute_repetition_penalty(tokens, 3)
        self.assertLess(penalty_new, penalty_repeat)

    def test_step_reward_shape(self):
        logits = torch.randn(1000)
        tokens = [1, 2, 3, 4, 5]
        new_token = 6

        reward, components = self.reward_computer.compute_step_reward(
            logits, tokens, new_token, normalize=False,
        )

        self.assertIsInstance(reward, float)
        self.assertIn("perplexity", components)
        self.assertIn("diversity", components)
        self.assertIn("repetition", components)
        self.assertIn("coherence", components)


class TestGatingPolicyAgent(unittest.TestCase):
    """Tests for GatingPolicyAgent class."""

    def setUp(self):
        self.config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
        }
        self.device = torch.device("cpu")
        self.agent = GatingPolicyAgent(self.config, self.device)

    def test_forward_shapes(self):
        obs = torch.randn(4, 64)
        action_dist, value = self.agent.forward(obs)

        self.assertEqual(value.shape, (4, 1))

        actions = action_dist.sample()
        self.assertEqual(actions.shape, (4, 3))
        self.assertTrue(torch.all(actions >= 0))
        self.assertTrue(torch.all(actions <= 1))

    def test_get_action_single(self):
        obs = torch.randn(64)
        action = self.agent.get_action(obs)

        self.assertEqual(action.shape, (3,))
        self.assertTrue(torch.all(action >= 0))
        self.assertTrue(torch.all(action <= 1))

    def test_get_action_deterministic(self):
        obs = torch.randn(64)
        action1 = self.agent.get_action(obs, deterministic=True)
        action2 = self.agent.get_action(obs, deterministic=True)
        self.assertTrue(torch.allclose(action1, action2))

    def test_evaluate_actions(self):
        obs = torch.randn(32, 64)
        actions = torch.rand(32, 3) * 0.98 + 0.01

        log_probs, values, entropy = self.agent.evaluate_actions(obs, actions)

        self.assertEqual(log_probs.shape, (32,))
        self.assertEqual(values.shape, (32,))
        self.assertEqual(entropy.shape, (32,))

    def test_gaussian_agent(self):
        agent = GaussianGatingPolicyAgent(self.config, self.device)
        obs = torch.randn(64)
        action = agent.get_action(obs)

        self.assertEqual(action.shape, (3,))
        self.assertTrue(torch.all(action >= 0))
        self.assertTrue(torch.all(action <= 1))

    def test_create_agent_factory(self):
        self.config["RL_AGENT_TYPE"] = "beta"
        agent_beta = create_agent(self.config, self.device)
        self.assertIsInstance(agent_beta, GatingPolicyAgent)

        self.config["RL_AGENT_TYPE"] = "gaussian"
        agent_gauss = create_agent(self.config, self.device)
        self.assertIsInstance(agent_gauss, GaussianGatingPolicyAgent)


class TestIntegration(unittest.TestCase):
    """Integration tests for RL gating components working together."""

    def setUp(self):
        self.config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_BASE_TEMPERATURE": 1.0,
            "RL_BASE_REPETITION_PENALTY": 1.2,
            "RL_CREATIVITY_TEMP_MIN": 0.5,
            "RL_CREATIVITY_TEMP_MAX": 1.5,
            "RL_STABILITY_PENALTY_MIN": 1.0,
            "RL_STABILITY_PENALTY_MAX": 2.5,
            "RL_FOCUS_ATTENTION_STRENGTH": 2.0,
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
        }
        self.device = torch.device("cpu")

    def test_agent_modulator_integration(self):
        agent = GatingPolicyAgent(self.config, self.device)
        modulator = GatingModulator(self.config)

        obs = torch.randn(64)
        action = agent.get_action(obs)
        effects = modulator(action, seq_len=20, device=self.device)

        self.assertGreater(effects.temperature, 0)
        self.assertGreater(effects.repetition_penalty, 0)
        self.assertEqual(effects.attention_bias.shape[0], 20)

    def test_reward_computation_integration(self):
        reward_computer = RewardComputer(self.config, vocab_size=1000)

        logits_history = [torch.randn(1000) for _ in range(10)]
        tokens = [i % 100 for i in range(10)]

        rewards, components = reward_computer.compute_episode_rewards(
            logits_history, tokens, gamma=0.99,
        )

        self.assertEqual(rewards.shape[0], 10)
        self.assertEqual(len(components["diversity"]), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
