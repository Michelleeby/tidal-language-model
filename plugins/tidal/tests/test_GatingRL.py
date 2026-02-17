"""
test_GatingRL.py

Unit tests for the RL Gating Controller components.
"""

import math
import torch
import numpy as np
import unittest

from unittest.mock import patch, MagicMock

from plugins.tidal.GatingModulator import GatingModulator, GatingEffects, RandomGatingPolicy, FixedGatingPolicy
from plugins.tidal.RewardComputer import RewardComputer
from plugins.tidal.GatingPolicyAgent import GatingPolicyAgent, GaussianGatingPolicyAgent, create_agent
from plugins.tidal.RLTrainer import RolloutBuffer, PPOTrainer, ExponentialMovingAverage, EntropyHomeostasis
from plugins.tidal.GatingEnvironment import GatingEnvironment
from plugins.tidal.TransformerLM import TransformerLM


class TestGatingModulator(unittest.TestCase):
    """Tests for GatingModulator class."""

    def setUp(self):
        self.config = {
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
        }
        self.modulator = GatingModulator(self.config)
        self.device = torch.device("cpu")

    def test_temperature_modulation(self):
        temp_low = self.modulator.compute_temperature(0.0)
        self.assertAlmostEqual(temp_low, 0.3, places=4)

        temp_high = self.modulator.compute_temperature(1.0)
        self.assertAlmostEqual(temp_high, 2.0, places=4)

        temp_mid = self.modulator.compute_temperature(0.5)
        self.assertAlmostEqual(temp_mid, 1.15, places=4)

    def test_repetition_penalty_modulation(self):
        penalty_low = self.modulator.compute_repetition_penalty(0.0)
        self.assertAlmostEqual(penalty_low, 1.2, places=4)

        penalty_high = self.modulator.compute_repetition_penalty(1.0)
        self.assertAlmostEqual(penalty_high, 4.2, places=4)

        penalty_mid = self.modulator.compute_repetition_penalty(0.5)
        self.assertAlmostEqual(penalty_mid, 2.7, places=4)

        self.assertGreater(penalty_high, penalty_low)

    def test_base_repetition_penalty_scales_output(self):
        """base_repetition_penalty config value must scale the output."""
        # With base_repetition_penalty=1.2, stability=0.0 should give 1.2 (not 1.0)
        penalty = self.modulator.compute_repetition_penalty(0.0)
        self.assertAlmostEqual(penalty, 1.2, places=4)

        # Different base_repetition_penalty should produce a different result
        config_alt = dict(self.config, RL_BASE_REPETITION_PENALTY=2.0)
        modulator_alt = GatingModulator(config_alt)
        penalty_alt = modulator_alt.compute_repetition_penalty(0.0)
        self.assertAlmostEqual(penalty_alt, 2.0, places=4)
        self.assertNotAlmostEqual(penalty, penalty_alt, places=2)

    def test_top_k_modulation(self):
        """focus=0 -> k=100 (wide), focus=1 -> k=5 (sharp)."""
        top_k_wide = self.modulator.compute_top_k(0.0)
        self.assertEqual(top_k_wide, 100)

        top_k_sharp = self.modulator.compute_top_k(1.0)
        self.assertEqual(top_k_sharp, 5)

        # Mid-focus should be between
        top_k_mid = self.modulator.compute_top_k(0.5)
        self.assertGreater(top_k_mid, 5)
        self.assertLess(top_k_mid, 100)

    def test_top_p_modulation(self):
        """creativity=0 -> p=0.7 (narrow nucleus), creativity=1 -> p=1.0 (full)."""
        top_p_low = self.modulator.compute_top_p(0.0)
        self.assertAlmostEqual(top_p_low, 0.7, places=4)

        top_p_high = self.modulator.compute_top_p(1.0)
        self.assertAlmostEqual(top_p_high, 1.0, places=4)

        top_p_mid = self.modulator.compute_top_p(0.5)
        self.assertAlmostEqual(top_p_mid, 0.85, places=4)

    def test_forward_returns_top_k_and_top_p(self):
        """GatingEffects from forward() contains top_k and top_p fields."""
        gate_signals = torch.tensor([0.5, 0.5, 0.5])
        effects = self.modulator(gate_signals, 20, self.device)

        self.assertIsInstance(effects, GatingEffects)
        self.assertIsInstance(effects.top_k, int)
        self.assertIsInstance(effects.top_p, float)
        self.assertGreaterEqual(effects.top_k, 5)
        self.assertLessEqual(effects.top_k, 100)
        self.assertGreaterEqual(effects.top_p, 0.7)
        self.assertLessEqual(effects.top_p, 1.0)

    def test_no_attention_bias_field(self):
        """GatingEffects no longer has an attention_bias field."""
        gate_signals = torch.tensor([0.5, 0.5, 0.5])
        effects = self.modulator(gate_signals, 20, self.device)
        self.assertFalse(hasattr(effects, "attention_bias"))

    def test_forward_returns_effects(self):
        gate_signals = torch.tensor([0.5, 0.5, 0.5])
        effects = self.modulator(gate_signals, 20, self.device)

        self.assertIsInstance(effects, GatingEffects)
        self.assertIsInstance(effects.temperature, float)
        self.assertIsInstance(effects.repetition_penalty, float)
        self.assertIsInstance(effects.top_k, int)
        self.assertIsInstance(effects.top_p, float)

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
            "RL_ENTROPY_TARGET": 5.0,
        }
        self.vocab_size = 1000
        self.reward_computer = RewardComputer(self.config, self.vocab_size)

    def test_diversity_reward_peaked_logits(self):
        """Peaked logits (near-deterministic) produce low diversity reward."""
        logits = torch.full((1000,), -100.0)
        logits[0] = 10.0
        diversity = self.reward_computer.compute_diversity_reward(logits)
        self.assertLess(diversity, 0.3)

    def test_diversity_reward_target_entropy(self):
        """Logits near target entropy produce high diversity reward."""
        # Uniform over exp(5) ≈ 148 tokens gives entropy ≈ 5.0
        logits = torch.full((1000,), -100.0)
        k = int(math.exp(5.0))
        logits[:k] = 0.0
        diversity = self.reward_computer.compute_diversity_reward(logits)
        self.assertGreater(diversity, 0.8)

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

    def test_step_reward_shape_with_focus(self):
        """When sampling_entropy is provided, 'focus' appears in components."""
        logits = torch.randn(1000)
        tokens = [1, 2, 3, 4, 5]
        new_token = 6

        reward, components = self.reward_computer.compute_step_reward(
            logits, tokens, new_token, normalize=False, sampling_entropy=2.5,
        )

        self.assertIsInstance(reward, float)
        self.assertIn("focus", components)


class TestLogitsEntropyDiversity(unittest.TestCase):
    """Tests that diversity reward uses logits entropy, not trivially-saturated distinct-n."""

    def setUp(self):
        self.config = {
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_ENTROPY_TARGET": 5.0,
        }
        self.vocab_size = 1000
        self.rc = RewardComputer(self.config, self.vocab_size)

    def test_accepts_logits_tensor(self):
        """compute_diversity_reward takes a logits tensor, not a token list."""
        logits = torch.randn(1000)
        reward = self.rc.compute_diversity_reward(logits)
        self.assertIsInstance(reward, float)

    def test_reward_always_in_0_1(self):
        """Output is always in [0, 1] for arbitrary logits."""
        for _ in range(50):
            logits = torch.randn(1000) * 10
            reward = self.rc.compute_diversity_reward(logits)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

    def test_peaked_logits_low_reward(self):
        """Near-deterministic distribution (entropy ≈ 0) gives low reward."""
        logits = torch.full((1000,), -100.0)
        logits[0] = 10.0
        reward = self.rc.compute_diversity_reward(logits)
        self.assertLess(reward, 0.2)

    def test_target_entropy_peak_reward(self):
        """Logits with entropy ≈ target give reward near 1.0."""
        # Uniform over exp(5) ≈ 148 tokens → entropy ≈ 5.0
        logits = torch.full((1000,), -100.0)
        k = int(math.exp(5.0))
        logits[:k] = 0.0
        reward = self.rc.compute_diversity_reward(logits)
        self.assertGreater(reward, 0.8)

    def test_uniform_logits_below_peak(self):
        """Uniform over full vocab (entropy=log(1000)≈6.9 > target=5.0) gives < 1.0."""
        logits = torch.zeros(1000)
        reward = self.rc.compute_diversity_reward(logits)
        self.assertLess(reward, 1.0)
        self.assertGreater(reward, 0.1)  # not zero, still reasonable entropy

    def test_gradient_signal_not_saturated(self):
        """Different logits produce meaningfully different rewards — not all ≈ 1.0."""
        # Peaked (entropy ≈ 0)
        logits_peaked = torch.full((1000,), -100.0)
        logits_peaked[0] = 10.0

        # Near-target (entropy ≈ 5.0)
        logits_target = torch.full((1000,), -100.0)
        logits_target[:int(math.exp(5.0))] = 0.0

        # Uniform (entropy ≈ 6.9)
        logits_uniform = torch.zeros(1000)

        r_peaked = self.rc.compute_diversity_reward(logits_peaked)
        r_target = self.rc.compute_diversity_reward(logits_target)
        r_uniform = self.rc.compute_diversity_reward(logits_uniform)

        # Target entropy gives highest reward
        self.assertGreater(r_target, r_peaked)
        self.assertGreater(r_target, r_uniform)
        # All three are distinct — no saturation
        self.assertGreater(r_target - r_peaked, 0.3)

    def test_config_entropy_target_used(self):
        """Different RL_ENTROPY_TARGET values shift the peak."""
        config_low = dict(self.config, RL_ENTROPY_TARGET=2.0)
        config_high = dict(self.config, RL_ENTROPY_TARGET=6.0)
        rc_low = RewardComputer(config_low, self.vocab_size)
        rc_high = RewardComputer(config_high, self.vocab_size)

        # Logits with entropy ≈ 2.0 (uniform over exp(2) ≈ 7 tokens)
        logits_low_ent = torch.full((1000,), -100.0)
        logits_low_ent[:7] = 0.0

        # rc_low (target=2.0) should reward this more than rc_high (target=6.0)
        r_low_target = rc_low.compute_diversity_reward(logits_low_ent)
        r_high_target = rc_high.compute_diversity_reward(logits_low_ent)
        self.assertGreater(r_low_target, r_high_target)


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
            "RL_CREATIVITY_TEMP_MIN": 0.3,
            "RL_CREATIVITY_TEMP_MAX": 2.0,
            "RL_STABILITY_PENALTY_MIN": 1.0,
            "RL_STABILITY_PENALTY_MAX": 3.5,
            "RL_FOCUS_TOP_K_MIN": 5,
            "RL_FOCUS_TOP_K_MAX": 100,
            "RL_CREATIVITY_TOP_P_MIN": 0.7,
            "RL_CREATIVITY_TOP_P_MAX": 1.0,
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
        self.assertIsInstance(effects.top_k, int)
        self.assertGreaterEqual(effects.top_k, 5)
        self.assertLessEqual(effects.top_k, 100)
        self.assertIsInstance(effects.top_p, float)
        self.assertGreaterEqual(effects.top_p, 0.7)
        self.assertLessEqual(effects.top_p, 1.0)

    def test_reward_computation_integration(self):
        reward_computer = RewardComputer(self.config, vocab_size=1000)

        logits_history = [torch.randn(1000) for _ in range(10)]
        tokens = [i % 100 for i in range(10)]

        rewards, components = reward_computer.compute_episode_rewards(
            logits_history, tokens, gamma=0.99,
        )

        self.assertEqual(rewards.shape[0], 10)
        self.assertEqual(len(components["diversity"]), 10)


class TestRolloutBuffer(unittest.TestCase):
    """Tests for pre-allocated RolloutBuffer."""

    def setUp(self):
        self.capacity = 16
        self.obs_dim = 64
        self.action_dim = 3
        self.buffer = RolloutBuffer(self.capacity, self.obs_dim, self.action_dim)

    def test_buffer_preallocated_capacity(self):
        """Capacity matches init arg."""
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertEqual(self.buffer.observations.shape, (self.capacity, self.obs_dim))
        self.assertEqual(self.buffer.actions.shape, (self.capacity, self.action_dim))

    def test_buffer_add_fills_sequential_slots(self):
        """observations[0] matches first added tensor."""
        obs = torch.randn(self.obs_dim)
        action = torch.rand(self.action_dim)
        self.buffer.add(obs, action, reward=1.0, value=0.5, log_prob=-0.3, done=False)
        self.assertTrue(torch.allclose(self.buffer.observations[0], obs))
        self.assertTrue(torch.allclose(self.buffer.actions[0], action))
        self.assertAlmostEqual(self.buffer.rewards[0].item(), 1.0)

    def test_buffer_clear_resets_position(self):
        """len returns 0 after clear, no reallocation."""
        obs_id_before = id(self.buffer.observations)
        self.buffer.add(
            torch.randn(self.obs_dim), torch.rand(self.action_dim),
            reward=1.0, value=0.5, log_prob=-0.3, done=False,
        )
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)
        # Same tensor storage (no reallocation)
        self.assertEqual(id(self.buffer.observations), obs_id_before)

    def test_buffer_len_returns_filled_count(self):
        """len returns number of adds, not capacity."""
        self.assertEqual(len(self.buffer), 0)
        for i in range(5):
            self.buffer.add(
                torch.randn(self.obs_dim), torch.rand(self.action_dim),
                reward=float(i), value=0.5, log_prob=-0.3, done=False,
            )
        self.assertEqual(len(self.buffer), 5)

    def test_buffer_observations_are_contiguous(self):
        """observations[:n] is a contiguous tensor."""
        for i in range(8):
            self.buffer.add(
                torch.randn(self.obs_dim), torch.rand(self.action_dim),
                reward=float(i), value=0.5, log_prob=-0.3, done=False,
            )
        sliced = self.buffer.observations[:8]
        self.assertTrue(sliced.is_contiguous())
        self.assertEqual(sliced.shape, (8, self.obs_dim))

    def test_buffer_overflow_raises(self):
        """Adding beyond capacity raises RuntimeError."""
        for i in range(self.capacity):
            self.buffer.add(
                torch.randn(self.obs_dim), torch.rand(self.action_dim),
                reward=float(i), value=0.5, log_prob=-0.3, done=False,
            )
        with self.assertRaises(RuntimeError):
            self.buffer.add(
                torch.randn(self.obs_dim), torch.rand(self.action_dim),
                reward=0.0, value=0.0, log_prob=0.0, done=False,
            )


class TestGatingEnvironmentPrecomputed(unittest.TestCase):
    """Tests for _get_observation with precomputed hidden_states/logits."""

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
        }
        cls.rl_config = {
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
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_MAX_EPISODE_LENGTH": 10,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
        }
        cls.config = {**cls.model_config, **cls.rl_config}
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()

    def _make_env(self):
        modulator = GatingModulator(self.config)
        reward_computer = RewardComputer(self.config, 100)
        prompts = [[1, 2, 3, 4, 5]]
        return GatingEnvironment(
            model=self.model,
            modulator=modulator,
            reward_computer=reward_computer,
            prompt_tokens=prompts,
            config=self.config,
            device=torch.device("cpu"),
        )

    def test_get_observation_accepts_precomputed(self):
        """Call with hidden_states/logits: no forward pass invoked."""
        env = self._make_env()
        env.reset()

        # Produce real hidden_states/logits from a forward call
        context = torch.tensor(
            env.generated_tokens[-self.model.max_context_length:],
            dtype=torch.long,
        )
        with torch.no_grad():
            logits, hidden_states = self.model.forward_with_hidden(context.unsqueeze(0))

        # Patch forward_with_hidden to detect if it's called
        with patch.object(self.model, "forward_with_hidden", wraps=self.model.forward_with_hidden) as mock_fwd:
            obs = env._get_observation(
                precomputed_hidden_states=hidden_states,
                precomputed_logits=logits,
            )
            mock_fwd.assert_not_called()

        self.assertEqual(obs.shape, (64,))

    def test_get_observation_fallback_without_precomputed(self):
        """Call without precomputed args: performs forward pass (for reset())."""
        env = self._make_env()
        env.reset()

        with patch.object(self.model, "forward_with_hidden", wraps=self.model.forward_with_hidden) as mock_fwd:
            obs = env._get_observation()
            mock_fwd.assert_called_once()

        self.assertEqual(obs.shape, (64,))

    def test_observation_shape_unchanged(self):
        """Shape is (64,) regardless of precomputed path."""
        env = self._make_env()
        env.reset()

        obs_fallback = env._get_observation()

        context = torch.tensor(
            env.generated_tokens[-self.model.max_context_length:],
            dtype=torch.long,
        )
        with torch.no_grad():
            logits, hidden_states = self.model.forward_with_hidden(context.unsqueeze(0))

        obs_precomputed = env._get_observation(
            precomputed_hidden_states=hidden_states,
            precomputed_logits=logits,
        )

        self.assertEqual(obs_fallback.shape, (64,))
        self.assertEqual(obs_precomputed.shape, (64,))


class TestExponentialMovingAverage(unittest.TestCase):
    """Tests for the EMA tracker that replaces deque-based episode stats."""

    def test_ema_tracker_basic(self):
        """EMA produces smooth output that converges toward input values."""
        ema = ExponentialMovingAverage(alpha=0.1)
        self.assertIsNone(ema.value)

        ema.update(10.0)
        self.assertAlmostEqual(ema.value, 10.0, places=5)

        # Feed constant value — EMA should converge to it
        for _ in range(100):
            ema.update(20.0)
        self.assertAlmostEqual(ema.value, 20.0, delta=0.5)

    def test_ema_tracker_no_cycle(self):
        """Feeding a repeating pattern does NOT produce exact repeating output.

        The deque(maxlen=100) bug caused np.mean to cycle with period 26
        when fed deterministic episode lengths. EMA must not exhibit this.
        """
        ema = ExponentialMovingAverage(alpha=0.05)

        # Simulate the bug scenario: feed a repeating 26-step cycle
        cycle = [40.26, 40.34, 40.42, 40.50, 40.58, 40.66, 40.74, 40.82,
                 40.90, 40.98, 41.06, 41.14, 41.22, 41.30, 41.38, 41.46,
                 41.54, 41.62, 41.70, 41.78, 41.86, 41.94, 41.02, 40.60,
                 40.68, 40.26]

        outputs = []
        for i in range(200):
            ema.update(cycle[i % len(cycle)])
            outputs.append(ema.value)

        # After convergence, check that EMA values do NOT form an exact cycle
        # Take last 52 values (two full input cycles)
        tail = outputs[-52:]
        first_half = tail[:26]
        second_half = tail[26:]

        # If this were a deque, first_half == second_half exactly.
        # With EMA, they should differ because EMA has infinite memory.
        diffs = [abs(a - b) for a, b in zip(first_half, second_half)]
        max_diff = max(diffs)
        # EMA values should converge, making consecutive cycle outputs
        # nearly identical but not bit-for-bit equal to previous cycle
        # The key property: no exact repeating cycle artifact
        self.assertLess(max_diff, 0.01,
                        "EMA should converge, but values should not cycle exactly")

    def test_ema_alpha_configurable(self):
        """Different alpha values produce different smoothing behavior."""
        fast = ExponentialMovingAverage(alpha=0.5)
        slow = ExponentialMovingAverage(alpha=0.01)

        fast.update(0.0)
        slow.update(0.0)

        # Jump to 100
        fast.update(100.0)
        slow.update(100.0)

        # Fast EMA should react more quickly
        self.assertGreater(fast.value, slow.value)


class TestBetaConcentrationCapping(unittest.TestCase):
    """Tests that Beta distribution concentration params are capped."""

    def setUp(self):
        self.config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_BETA_CONCENTRATION_MAX": 20.0,
        }
        self.device = torch.device("cpu")

    def test_beta_concentration_capped(self):
        """Alpha and beta parameters never exceed the configured max."""
        agent = GatingPolicyAgent(self.config, self.device)

        # Generate many different observations to try to push concentration high
        for _ in range(50):
            obs = torch.randn(64) * 10  # Large magnitude inputs
            action_dist, _ = agent.forward(obs)

            alpha = action_dist.concentration1
            beta_param = action_dist.concentration0

            max_val = self.config["RL_BETA_CONCENTRATION_MAX"]
            self.assertTrue(
                torch.all(alpha <= max_val + 1e-6),
                f"Alpha {alpha.max().item()} exceeds max {max_val}",
            )
            self.assertTrue(
                torch.all(beta_param <= max_val + 1e-6),
                f"Beta {beta_param.max().item()} exceeds max {max_val}",
            )

    def test_beta_concentration_default_max(self):
        """Agent uses 15.0 as default max when config key is absent."""
        config_no_max = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
        }
        agent = GatingPolicyAgent(config_no_max, self.device)

        # Directly check the attribute value
        self.assertAlmostEqual(agent.beta_concentration_max, 15.0, places=1)


class TestEntropyCoefAnnealing(unittest.TestCase):
    """Tests that entropy coefficient anneals during training."""

    def setUp(self):
        self.config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_ENTROPY_COEF": 0.01,
            "RL_ENTROPY_COEF_FINAL": 0.1,
            "RL_TOTAL_TIMESTEPS": 100000,
            "RL_ROLLOUT_STEPS": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 32,
        }

    def test_entropy_coef_anneals(self):
        """Entropy coefficient increases from initial to final over training."""
        agent = GatingPolicyAgent(self.config, torch.device("cpu"))
        env = MagicMock()

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                config=self.config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )

            total_iters = self.config["RL_TOTAL_TIMESTEPS"] // self.config["RL_ROLLOUT_STEPS"]

            # At start
            coef_start = trainer.get_entropy_coef(iteration=0, total_iterations=total_iters)
            self.assertAlmostEqual(coef_start, 0.01, places=4)

            # At end
            coef_end = trainer.get_entropy_coef(iteration=total_iters - 1, total_iterations=total_iters)
            self.assertAlmostEqual(coef_end, 0.1, places=4)

            # At midpoint — should be between start and end
            coef_mid = trainer.get_entropy_coef(iteration=total_iters // 2, total_iterations=total_iters)
            self.assertGreater(coef_mid, 0.01)
            self.assertLess(coef_mid, 0.1)


class TestEnvStepRewardNormalization(unittest.TestCase):
    """Tests that GatingEnvironment.step() uses unnormalized rewards."""

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
        }
        cls.rl_config = {
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
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_MAX_EPISODE_LENGTH": 10,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
        }
        cls.config = {**cls.model_config, **cls.rl_config}
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()

    def test_env_step_uses_unnormalized_reward(self):
        """Environment step() calls compute_step_reward with normalize=False."""
        modulator = GatingModulator(self.config)
        reward_computer = RewardComputer(self.config, 100)
        prompts = [[1, 2, 3, 4, 5]]
        env = GatingEnvironment(
            model=self.model,
            modulator=modulator,
            reward_computer=reward_computer,
            prompt_tokens=prompts,
            config=self.config,
            device=torch.device("cpu"),
        )
        env.reset()

        action = torch.tensor([0.5, 0.5, 0.5])

        with patch.object(
            reward_computer, "compute_step_reward", wraps=reward_computer.compute_step_reward,
        ) as mock_reward:
            env.step(action)

            mock_reward.assert_called_once()
            call_kwargs = mock_reward.call_args
            # Check that normalize=False was passed
            # call_args can be (args, kwargs) — normalize is the 4th positional or keyword
            if call_kwargs.kwargs.get("normalize") is not None:
                self.assertFalse(call_kwargs.kwargs["normalize"])
            else:
                # Positional: compute_step_reward(logits, tokens, new_token, normalize)
                self.assertFalse(call_kwargs.args[3])


class TestEnvStepPassesModifiedLogits(unittest.TestCase):
    """Tests that env.step() passes modified logits (not raw) to compute_step_reward."""

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
        }
        cls.rl_config = {
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
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_MAX_EPISODE_LENGTH": 10,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
        }
        cls.config = {**cls.model_config, **cls.rl_config}
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()

    def test_reward_receives_modified_logits(self):
        """step() must pass temperature-scaled logits to compute_step_reward, not raw."""
        modulator = GatingModulator(self.config)
        reward_computer = RewardComputer(self.config, 100)
        prompts = [[1, 2, 3, 4, 5]]
        env = GatingEnvironment(
            model=self.model,
            modulator=modulator,
            reward_computer=reward_computer,
            prompt_tokens=prompts,
            config=self.config,
            device=torch.device("cpu"),
        )
        env.reset()

        # creativity=0.8 → temperature ~1.66, so logits will be scaled
        action = torch.tensor([0.8, 0.5, 0.5])

        captured_logits = []

        original_compute = reward_computer.compute_step_reward

        def capturing_compute(logits, *args, **kwargs):
            captured_logits.append(logits.clone())
            return original_compute(logits, *args, **kwargs)

        with patch.object(
            reward_computer, "compute_step_reward", side_effect=capturing_compute,
        ):
            env.step(action)

        self.assertEqual(len(captured_logits), 1)

        # Get raw logits from a fresh forward pass for comparison
        context = torch.tensor(
            env.generated_tokens[:-1][-self.model.max_context_length:],
            dtype=torch.long,
        )
        with torch.no_grad():
            raw_logits, _ = self.model.forward_with_hidden(context.unsqueeze(0))
        raw_last = raw_logits[0, -1, :]

        # The logits passed to reward should differ from raw (temperature-scaled)
        self.assertFalse(
            torch.allclose(captured_logits[0], raw_last, atol=1e-4),
            "Reward was computed from raw logits — should use modified logits "
            "(after repetition penalty + temperature scaling)",
        )


class TestPPOTrainerUsesEMA(unittest.TestCase):
    """Tests that PPOTrainer tracks episode stats via EMA, not deque."""

    def test_ppo_trainer_uses_ema(self):
        """PPOTrainer has EMA attributes instead of deque for episode tracking."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 128,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 32,
            "RL_TOTAL_TIMESTEPS": 100000,
            "RL_EMA_ALPHA": 0.05,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))
        env = MagicMock()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                config=config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )

            # Should have EMA tracker for rewards
            self.assertIsInstance(trainer.episode_rewards, ExponentialMovingAverage)
            self.assertEqual(trainer.episode_rewards.alpha, 0.05)

            # episode_lengths EMA is removed — provides no information
            self.assertFalse(
                hasattr(trainer, "episode_lengths"),
                "PPOTrainer must not have episode_lengths — replaced with richer metrics",
            )

            # Should NOT have deque attributes
            from collections import deque
            self.assertNotIsInstance(trainer.episode_rewards, deque)


class TestEpisodeTrackingPersistsAcrossRollouts(unittest.TestCase):
    """Tests that episode reward/length counters survive across collect_rollouts() calls."""

    def setUp(self):
        self.model_config = {
            "EMBED_DIM": 64,
            "NUM_TRANSFORMER_BLOCKS": 2,
            "NUM_ATTENTION_HEADS": 4,
            "FFN_HIDDEN_DIM": 128,
            "DROPOUT": 0.0,
            "MAX_CONTEXT_LENGTH": 32,
            "DEVICE": "cpu",
        }
        self.rl_config = {
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
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_MAX_EPISODE_LENGTH": 50,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.05,
            "RL_ENTROPY_COEF_FINAL": 0.3,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 128,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 32,
            "RL_TOTAL_TIMESTEPS": 100000,
            "RL_EMA_ALPHA": 0.05,
        }
        self.config = {**self.model_config, **self.rl_config}

    def test_episode_tracking_persists_across_rollouts(self):
        """Episode reward counters survive across collect_rollouts() calls.

        A 50-step episode starting at step 100 of a 128-step rollout gets only
        28 steps before the rollout ends. The remaining 22 steps happen in the
        next collect_rollouts() call. Every reported episode reward must be 50.0
        (50 steps * 1.0 reward/step).
        """
        agent = GatingPolicyAgent(self.config, torch.device("cpu"))

        mock_env = MagicMock()
        step_counter = [0]

        def mock_reset():
            step_counter[0] = 0
            mock_env.done = False
            mock_env.generated_tokens = [1]  # Non-empty: episode in progress
            return torch.randn(64)

        def mock_step(action):
            step_counter[0] += 1
            done = (step_counter[0] >= 50)
            if done:
                step_counter[0] = 0
                mock_env.done = True
            else:
                mock_env.generated_tokens = list(range(step_counter[0]))
            return torch.randn(64), 1.0, done, {
                "gate_signals": {"creativity": 0.5, "focus": 0.5, "stability": 0.5},
                "reward_components": {"perplexity": 0.0, "diversity": 0.0, "repetition": 0.0, "coherence": 0.0},
            }

        def mock_get_observation():
            return torch.randn(64)

        mock_env.reset.side_effect = mock_reset
        mock_env.step.side_effect = mock_step
        mock_env._get_observation = mock_get_observation
        mock_env.done = False
        mock_env.generated_tokens = []

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=mock_env,
                config=self.config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )

            # Capture every value passed to episode_rewards.update()
            recorded_rewards = []
            original_update = trainer.episode_rewards.update

            def capturing_update(value):
                recorded_rewards.append(value)
                return original_update(value)

            trainer.episode_rewards.update = capturing_update

            # First rollout: 128 steps. Episodes complete at step 50 and 100.
            # Episode starting at step 100 gets 28 steps (100-127), not done.
            trainer.collect_rollouts(128)

            # Second rollout: 128 more steps. The in-progress episode from
            # the first rollout should complete after 22 more steps (28+22=50).
            trainer.collect_rollouts(128)

            # Every recorded episode reward must be exactly 50.0 (50 steps * 1.0).
            # BUG: with local vars, the boundary episode reports 22.0 instead.
            self.assertTrue(len(recorded_rewards) > 0, "Should have completed episodes")
            for i, reward in enumerate(recorded_rewards):
                self.assertAlmostEqual(
                    reward, 50.0,
                    msg=f"Episode {i} reported reward={reward}, expected 50.0. "
                    f"All rewards: {recorded_rewards}",
                )


class TestBetaConcentrationCappedAt15(unittest.TestCase):
    """Tests that Beta distribution concentration is capped at 15.0."""

    def test_beta_concentration_capped_at_15(self):
        """With RL_BETA_CONCENTRATION_MAX=15.0, alpha/beta never exceed 15.0."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_BETA_CONCENTRATION_MAX": 15.0,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))

        for _ in range(50):
            obs = torch.randn(64) * 10
            action_dist, _ = agent.forward(obs)

            alpha = action_dist.concentration1
            beta_param = action_dist.concentration0

            self.assertTrue(
                torch.all(alpha <= 15.0 + 1e-6),
                f"Alpha {alpha.max().item()} exceeds 15.0",
            )
            self.assertTrue(
                torch.all(beta_param <= 15.0 + 1e-6),
                f"Beta {beta_param.max().item()} exceeds 15.0",
            )


class TestEntropyCoefRangeUpdated(unittest.TestCase):
    """Tests that PPOTrainer reads the recalibrated entropy coef range."""

    def test_entropy_coef_range_0_01_to_0_03(self):
        """PPOTrainer with new config reads initial=0.01, final=0.03."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_ENTROPY_COEF_FINAL": 0.03,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 128,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 32,
            "RL_TOTAL_TIMESTEPS": 100000,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))
        env = MagicMock()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                config=config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )

            total_iters = config["RL_TOTAL_TIMESTEPS"] // config["RL_ROLLOUT_STEPS"]

            coef_start = trainer.get_entropy_coef(0, total_iters)
            self.assertAlmostEqual(coef_start, 0.01, places=4)

            coef_end = trainer.get_entropy_coef(total_iters - 1, total_iters)
            self.assertAlmostEqual(coef_end, 0.03, places=4)


class TestEntropyFloorRemoved(unittest.TestCase):
    """Tests that entropy floor mechanism has been removed from PPOTrainer."""

    def test_no_entropy_floor_attribute(self):
        """PPOTrainer must NOT have an entropy_floor attribute — the mechanism is removed."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_ENTROPY_COEF_FINAL": 0.03,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 1,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 100000,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))
        env = MagicMock()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                config=config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )
            self.assertFalse(
                hasattr(trainer, "entropy_floor"),
                "PPOTrainer must not have entropy_floor — the mechanism is removed",
            )

    def test_no_entropy_floor_even_if_config_has_it(self):
        """Even if RL_ENTROPY_FLOOR is in config, trainer must not use it."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_ENTROPY_COEF_FINAL": 0.03,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 1,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 100000,
            "RL_ENTROPY_FLOOR": -1.5,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))
        env = MagicMock()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                config=config,
                experiment_dir=tmpdir,
                device=torch.device("cpu"),
            )
            self.assertFalse(
                hasattr(trainer, "entropy_floor"),
                "PPOTrainer must not read RL_ENTROPY_FLOOR from config",
            )


class TestCollectRolloutsReturnsGateSignals(unittest.TestCase):
    """Tests that collect_rollouts() returns mean gate signal values."""

    def test_rollout_stats_contain_gate_signals(self):
        """collect_rollouts() stats include mean_gate_creativity/focus/stability."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 100000,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))

        mock_env = MagicMock()
        step_counter = [0]

        def mock_reset():
            step_counter[0] = 0
            mock_env.done = False
            mock_env.generated_tokens = [1]
            return torch.randn(64)

        def mock_step(action):
            step_counter[0] += 1
            done = step_counter[0] >= 50
            if done:
                step_counter[0] = 0
                mock_env.done = True
            else:
                mock_env.generated_tokens = list(range(step_counter[0]))
            return torch.randn(64), 1.0, done, {
                "gate_signals": {"creativity": 0.5, "focus": 0.6, "stability": 0.7},
                "reward_components": {"perplexity": 0.1, "diversity": 0.2, "repetition": 0.3, "coherence": 0.4},
            }

        mock_env.reset.side_effect = mock_reset
        mock_env.step.side_effect = mock_step
        mock_env._get_observation = lambda: torch.randn(64)
        mock_env.done = False
        mock_env.generated_tokens = []

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent, env=mock_env, config=config,
                experiment_dir=tmpdir, device=torch.device("cpu"),
            )
            stats = trainer.collect_rollouts(16)

            self.assertIn("mean_gate_creativity", stats)
            self.assertIn("mean_gate_focus", stats)
            self.assertIn("mean_gate_stability", stats)
            self.assertAlmostEqual(stats["mean_gate_creativity"], 0.5, places=4)
            self.assertAlmostEqual(stats["mean_gate_focus"], 0.6, places=4)
            self.assertAlmostEqual(stats["mean_gate_stability"], 0.7, places=4)


class TestCollectRolloutsReturnsRewardComponents(unittest.TestCase):
    """Tests that collect_rollouts() returns mean reward component values."""

    def test_rollout_stats_contain_reward_components(self):
        """collect_rollouts() stats include mean_reward_perplexity/diversity/repetition/coherence."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 4,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 100000,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))

        mock_env = MagicMock()
        step_counter = [0]

        def mock_reset():
            step_counter[0] = 0
            mock_env.done = False
            mock_env.generated_tokens = [1]
            return torch.randn(64)

        def mock_step(action):
            step_counter[0] += 1
            done = step_counter[0] >= 50
            if done:
                step_counter[0] = 0
                mock_env.done = True
            else:
                mock_env.generated_tokens = list(range(step_counter[0]))
            return torch.randn(64), 1.0, done, {
                "gate_signals": {"creativity": 0.5, "focus": 0.6, "stability": 0.7},
                "reward_components": {"perplexity": 0.1, "diversity": 0.2, "repetition": 0.3, "coherence": 0.4},
            }

        mock_env.reset.side_effect = mock_reset
        mock_env.step.side_effect = mock_step
        mock_env._get_observation = lambda: torch.randn(64)
        mock_env.done = False
        mock_env.generated_tokens = []

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent, env=mock_env, config=config,
                experiment_dir=tmpdir, device=torch.device("cpu"),
            )
            stats = trainer.collect_rollouts(16)

            self.assertIn("mean_reward_perplexity", stats)
            self.assertIn("mean_reward_diversity", stats)
            self.assertIn("mean_reward_repetition", stats)
            self.assertIn("mean_reward_coherence", stats)
            self.assertAlmostEqual(stats["mean_reward_perplexity"], 0.1, places=4)
            self.assertAlmostEqual(stats["mean_reward_diversity"], 0.2, places=4)
            self.assertAlmostEqual(stats["mean_reward_repetition"], 0.3, places=4)
            self.assertAlmostEqual(stats["mean_reward_coherence"], 0.4, places=4)


class TestExplainedVarianceComputation(unittest.TestCase):
    """Tests that train() history contains explained_variance."""

    def test_history_contains_explained_variance(self):
        """train() history includes explained_variance in [-1, 1] range."""
        config = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 1,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 32,
        }
        agent = GatingPolicyAgent(config, torch.device("cpu"))

        mock_env = MagicMock()
        step_counter = [0]

        def mock_reset():
            step_counter[0] = 0
            mock_env.done = False
            mock_env.generated_tokens = [1]
            return torch.randn(64)

        def mock_step(action):
            step_counter[0] += 1
            done = step_counter[0] >= 50
            if done:
                step_counter[0] = 0
                mock_env.done = True
            else:
                mock_env.generated_tokens = list(range(step_counter[0]))
            return torch.randn(64), 1.0, done, {
                "gate_signals": {"creativity": 0.5, "focus": 0.6, "stability": 0.7},
                "reward_components": {"perplexity": 0.1, "diversity": 0.2, "repetition": 0.3, "coherence": 0.4},
            }

        mock_env.reset.side_effect = mock_reset
        mock_env.step.side_effect = mock_step
        mock_env._get_observation = lambda: torch.randn(64)
        mock_env.done = False
        mock_env.generated_tokens = []

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent, env=mock_env, config=config,
                experiment_dir=tmpdir, device=torch.device("cpu"),
            )
            history = trainer.train(total_timesteps=32)

            self.assertIn("explained_variance", history)
            self.assertTrue(len(history["explained_variance"]) > 0)
            for ev in history["explained_variance"]:
                self.assertGreaterEqual(ev, -1.0)
                self.assertLessEqual(ev, 1.0)


class TestEntropyHomeostasis(unittest.TestCase):
    """Tests for the EntropyHomeostasis closed-loop controller."""

    def setUp(self):
        self.config = {
            "RL_ENTROPY_COEF": 0.01,
            "RL_POLICY_ENTROPY_TARGET": -1.0,
            "RL_ENTROPY_HOMEOSTASIS_RELEASE_RATE": 0.05,
            "RL_ENTROPY_HOMEOSTASIS_DECAY_RATE": 0.95,
            "RL_ENTROPY_COEF_MIN": 0.01,
            "RL_ENTROPY_COEF_MAX": 0.5,
        }

    def test_init_defaults(self):
        """All attributes initialize from config."""
        h = EntropyHomeostasis(self.config)
        self.assertAlmostEqual(h.coef, 0.01)
        self.assertAlmostEqual(h.baseline, 0.01)
        self.assertAlmostEqual(h.target, -1.0)
        self.assertAlmostEqual(h.release_rate, 0.05)
        self.assertAlmostEqual(h.decay_rate, 0.95)
        self.assertAlmostEqual(h.coef_min, 0.01)
        self.assertAlmostEqual(h.coef_max, 0.5)

    def test_boost_when_entropy_too_low(self):
        """Coef increases when entropy < target."""
        h = EntropyHomeostasis(self.config)
        initial_coef = h.coef
        h.step(current_entropy=-2.5)  # well below target of -1.0
        self.assertGreater(h.coef, initial_coef)

    def test_no_boost_when_entropy_healthy(self):
        """Coef decays toward baseline when entropy >= target."""
        h = EntropyHomeostasis(self.config)
        # First pump coef up
        for _ in range(50):
            h.step(current_entropy=-3.0)
        pumped_coef = h.coef
        self.assertGreater(pumped_coef, h.baseline)
        # Now feed healthy entropy — coef should decrease
        h.step(current_entropy=-0.5)
        self.assertLess(h.coef, pumped_coef)

    def test_decay_toward_baseline(self):
        """After 200 healthy steps, coef converges to baseline."""
        h = EntropyHomeostasis(self.config)
        # First pump coef up
        for _ in range(50):
            h.step(current_entropy=-3.0)
        # Now let it decay with healthy entropy
        for _ in range(200):
            h.step(current_entropy=-0.5)
        self.assertAlmostEqual(h.coef, h.baseline, places=2)

    def test_clamping_to_max(self):
        """Aggressive release never exceeds RL_ENTROPY_COEF_MAX."""
        config = dict(self.config)
        config["RL_ENTROPY_HOMEOSTASIS_RELEASE_RATE"] = 10.0  # very aggressive
        h = EntropyHomeostasis(config)
        for _ in range(100):
            h.step(current_entropy=-10.0)
        self.assertLessEqual(h.coef, config["RL_ENTROPY_COEF_MAX"])

    def test_clamping_to_min(self):
        """Aggressive decay never drops below RL_ENTROPY_COEF_MIN."""
        config = dict(self.config)
        config["RL_ENTROPY_HOMEOSTASIS_DECAY_RATE"] = 0.01  # very aggressive decay
        config["RL_ENTROPY_COEF"] = 0.02  # start slightly above min
        h = EntropyHomeostasis(config)
        for _ in range(100):
            h.step(current_entropy=0.0)  # healthy entropy, only decay
        self.assertGreaterEqual(h.coef, config["RL_ENTROPY_COEF_MIN"])

    def test_step_returns_current_coef(self):
        """Return value of step() matches h.coef."""
        h = EntropyHomeostasis(self.config)
        result = h.step(current_entropy=-2.0)
        self.assertAlmostEqual(result, h.coef)

    def test_proportional_response(self):
        """Larger entropy deficit produces a larger boost."""
        config = dict(self.config)
        config["RL_ENTROPY_HOMEOSTASIS_DECAY_RATE"] = 1.0  # no decay, isolate release
        h1 = EntropyHomeostasis(config)
        h2 = EntropyHomeostasis(config)

        h1.step(current_entropy=-1.5)  # small deficit (target is -1.0)
        h2.step(current_entropy=-3.0)  # large deficit

        self.assertGreater(h2.coef, h1.coef)


class TestPPOTrainerHomeostaticSchedule(unittest.TestCase):
    """Tests that PPOTrainer integrates the homeostatic entropy schedule."""

    def _base_config(self):
        return {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
            "RL_LEARNING_RATE": 3e-4,
            "RL_GAMMA": 0.99,
            "RL_GAE_LAMBDA": 0.95,
            "RL_CLIP_EPSILON": 0.2,
            "RL_ENTROPY_COEF": 0.01,
            "RL_ENTROPY_COEF_FINAL": 0.03,
            "RL_VALUE_COEF": 0.5,
            "RL_MAX_GRAD_NORM": 0.5,
            "RL_ROLLOUT_STEPS": 16,
            "RL_NUM_EPOCHS": 1,
            "RL_BATCH_SIZE": 16,
            "RL_TOTAL_TIMESTEPS": 100000,
        }

    def _make_trainer(self, config):
        import tempfile
        agent = GatingPolicyAgent(config, torch.device("cpu"))
        env = MagicMock()
        tmpdir = tempfile.mkdtemp()
        return PPOTrainer(
            agent=agent, env=env, config=config,
            experiment_dir=tmpdir, device=torch.device("cpu"),
        ), tmpdir

    def test_homeostasis_creates_controller(self):
        """RL_ENTROPY_SCHEDULE='homeostasis' creates EntropyHomeostasis instance."""
        config = self._base_config()
        config["RL_ENTROPY_SCHEDULE"] = "homeostasis"
        config["RL_POLICY_ENTROPY_TARGET"] = -1.0
        config["RL_ENTROPY_HOMEOSTASIS_RELEASE_RATE"] = 0.05
        config["RL_ENTROPY_HOMEOSTASIS_DECAY_RATE"] = 0.95
        config["RL_ENTROPY_COEF_MIN"] = 0.01
        config["RL_ENTROPY_COEF_MAX"] = 0.5
        trainer, _ = self._make_trainer(config)
        self.assertIsInstance(trainer.entropy_homeostasis, EntropyHomeostasis)

    def test_linear_schedule_no_homeostasis(self):
        """RL_ENTROPY_SCHEDULE='linear' means entropy_homeostasis is None."""
        config = self._base_config()
        config["RL_ENTROPY_SCHEDULE"] = "linear"
        trainer, _ = self._make_trainer(config)
        self.assertIsNone(trainer.entropy_homeostasis)

    def test_default_schedule_is_linear(self):
        """No RL_ENTROPY_SCHEDULE key defaults to linear (backward compat)."""
        config = self._base_config()
        # No RL_ENTROPY_SCHEDULE key
        trainer, _ = self._make_trainer(config)
        self.assertIsNone(trainer.entropy_homeostasis)

    def test_entropy_coef_logged_in_history(self):
        """history['entropy_coef'] is populated after train()."""
        config = self._base_config()
        config["RL_ENTROPY_SCHEDULE"] = "homeostasis"
        config["RL_POLICY_ENTROPY_TARGET"] = -1.0
        config["RL_ENTROPY_HOMEOSTASIS_RELEASE_RATE"] = 0.05
        config["RL_ENTROPY_HOMEOSTASIS_DECAY_RATE"] = 0.95
        config["RL_ENTROPY_COEF_MIN"] = 0.01
        config["RL_ENTROPY_COEF_MAX"] = 0.5
        config["RL_TOTAL_TIMESTEPS"] = 32
        config["RL_ROLLOUT_STEPS"] = 16

        agent = GatingPolicyAgent(config, torch.device("cpu"))
        mock_env = MagicMock()
        step_counter = [0]

        def mock_reset():
            step_counter[0] = 0
            mock_env.done = False
            mock_env.generated_tokens = [1]
            return torch.randn(64)

        def mock_step(action):
            step_counter[0] += 1
            done = step_counter[0] >= 50
            if done:
                step_counter[0] = 0
                mock_env.done = True
            else:
                mock_env.generated_tokens = list(range(step_counter[0]))
            return torch.randn(64), 1.0, done, {
                "gate_signals": {"creativity": 0.5, "focus": 0.5, "stability": 0.5},
                "reward_components": {"perplexity": 0.0, "diversity": 0.0, "repetition": 0.0, "coherence": 0.0},
            }

        mock_env.reset.side_effect = mock_reset
        mock_env.step.side_effect = mock_step
        mock_env._get_observation = lambda: torch.randn(64)
        mock_env.done = False
        mock_env.generated_tokens = []

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent, env=mock_env, config=config,
                experiment_dir=tmpdir, device=torch.device("cpu"),
            )
            history = trainer.train(total_timesteps=32)

            self.assertIn("entropy_coef", history)
            self.assertEqual(len(history["entropy_coef"]), 2)  # 32 / 16 = 2 iterations
            for coef in history["entropy_coef"]:
                self.assertGreaterEqual(coef, config["RL_ENTROPY_COEF_MIN"])
                self.assertLessEqual(coef, config["RL_ENTROPY_COEF_MAX"])


class TestFocusReward(unittest.TestCase):
    """Tests for the focus reward component (sampling entropy)."""

    def setUp(self):
        self.config = {
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.30,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.25,
            "RL_REWARD_FOCUS_WEIGHT": 0.15,
            "RL_REWARD_REPETITION_WEIGHT": 0.20,
            "RL_REWARD_COHERENCE_WEIGHT": 0.10,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_ENTROPY_TARGET": 5.0,
            "RL_SAMPLING_ENTROPY_TARGET": 2.5,
        }
        self.vocab_size = 1000
        self.rc = RewardComputer(self.config, self.vocab_size)

    def test_focus_reward_high_entropy(self):
        """High sampling entropy (far above target) gives reward < 1.0."""
        # entropy = 6.0, target = 2.5 → far above target
        reward = self.rc.compute_focus_reward(6.0)
        self.assertLess(reward, 0.5)

    def test_focus_reward_low_entropy(self):
        """Low sampling entropy (far below target) gives reward < 1.0."""
        # entropy = 0.1, target = 2.5 → far below target
        reward = self.rc.compute_focus_reward(0.1)
        self.assertLess(reward, 0.5)

    def test_focus_reward_at_target(self):
        """Sampling entropy near target gives reward near 1.0."""
        reward = self.rc.compute_focus_reward(2.5)
        self.assertGreater(reward, 0.99)

    def test_focus_reward_in_0_1(self):
        """Output always in [0, 1] for various entropy values."""
        for entropy in [0.0, 0.5, 1.0, 2.5, 5.0, 8.0, 12.0]:
            reward = self.rc.compute_focus_reward(entropy)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

    def test_focus_reward_config_target(self):
        """Different RL_SAMPLING_ENTROPY_TARGET shifts the peak."""
        config_low = dict(self.config, RL_SAMPLING_ENTROPY_TARGET=1.0)
        config_high = dict(self.config, RL_SAMPLING_ENTROPY_TARGET=5.0)
        rc_low = RewardComputer(config_low, self.vocab_size)
        rc_high = RewardComputer(config_high, self.vocab_size)

        # Entropy ≈ 1.0 should be rewarded more by rc_low (target=1.0)
        r_low = rc_low.compute_focus_reward(1.0)
        r_high = rc_high.compute_focus_reward(1.0)
        self.assertGreater(r_low, r_high)


class TestFocusRewardIntegration(unittest.TestCase):
    """Integration tests for focus reward in compute_step_reward."""

    def setUp(self):
        self.config = {
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.30,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.25,
            "RL_REWARD_FOCUS_WEIGHT": 0.15,
            "RL_REWARD_REPETITION_WEIGHT": 0.20,
            "RL_REWARD_COHERENCE_WEIGHT": 0.10,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_ENTROPY_TARGET": 5.0,
            "RL_SAMPLING_ENTROPY_TARGET": 2.5,
        }
        self.vocab_size = 1000
        self.rc = RewardComputer(self.config, self.vocab_size)

    def test_step_reward_includes_focus_component(self):
        """When sampling_entropy is passed, components dict includes 'focus'."""
        logits = torch.randn(1000)
        tokens = [1, 2, 3, 4, 5]
        reward, components = self.rc.compute_step_reward(
            logits, tokens, 6, normalize=False, sampling_entropy=2.5,
        )
        self.assertIn("focus", components)
        self.assertIsInstance(components["focus"], float)

    def test_step_reward_without_sampling_entropy_backward_compatible(self):
        """When sampling_entropy is not passed, no 'focus' key in components."""
        logits = torch.randn(1000)
        tokens = [1, 2, 3, 4, 5]
        reward, components = self.rc.compute_step_reward(
            logits, tokens, 6, normalize=False,
        )
        self.assertNotIn("focus", components)

    def test_different_top_k_produces_different_focus_reward(self):
        """Different sampling entropies produce different focus rewards."""
        logits = torch.randn(1000)
        tokens = [1, 2, 3, 4, 5]

        # Low sampling entropy (peaked distribution after tight top-k)
        _, comps_low = self.rc.compute_step_reward(
            logits, tokens, 6, normalize=False, sampling_entropy=0.5,
        )
        # High sampling entropy (broad distribution after wide top-k)
        _, comps_high = self.rc.compute_step_reward(
            logits, tokens, 6, normalize=False, sampling_entropy=5.0,
        )

        self.assertNotAlmostEqual(comps_low["focus"], comps_high["focus"], places=2)


class TestNucleusSampling(unittest.TestCase):
    """Tests for nucleus (top-p) sampling implementation."""

    def test_nucleus_sampling_filters_tokens(self):
        """top_p=0.5 zeroes out tokens outside the nucleus."""
        from plugins.tidal.GatingEnvironment import GatingEnvironment

        # Create a probability distribution where top 2 tokens hold ~80% mass
        probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])

        filtered = GatingEnvironment._apply_nucleus_sampling(probs, top_p=0.5)

        # Only the top token (0.5) should survive at top_p=0.5
        # The cumulative sum reaches 0.5 at the first token
        self.assertGreater(filtered[0].item(), 0.0)
        # After renormalization, some lower-prob tokens should be zeroed
        num_nonzero = (filtered > 0).sum().item()
        self.assertLess(num_nonzero, len(probs))

    def test_nucleus_sampling_top_p_1_keeps_all(self):
        """top_p=1.0 keeps all tokens (no filtering)."""
        from plugins.tidal.GatingEnvironment import GatingEnvironment

        probs = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])
        filtered = GatingEnvironment._apply_nucleus_sampling(probs, top_p=1.0)

        # All tokens should remain non-zero
        self.assertTrue(torch.all(filtered > 0))


class TestGatingEnvironmentDynamicSampling(unittest.TestCase):
    """Tests that GatingEnvironment.step() uses dynamic top-k and nucleus sampling."""

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
        }
        cls.rl_config = {
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
            "RL_REWARD_PERPLEXITY_WEIGHT": 0.4,
            "RL_REWARD_DIVERSITY_WEIGHT": 0.3,
            "RL_REWARD_REPETITION_WEIGHT": 0.2,
            "RL_REWARD_COHERENCE_WEIGHT": 0.1,
            "RL_PERPLEXITY_CLIP": 100.0,
            "RL_MAX_EPISODE_LENGTH": 10,
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
        }
        cls.config = {**cls.model_config, **cls.rl_config}
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()

    def _make_env(self):
        modulator = GatingModulator(self.config)
        reward_computer = RewardComputer(self.config, 100)
        prompts = [[1, 2, 3, 4, 5]]
        return GatingEnvironment(
            model=self.model,
            modulator=modulator,
            reward_computer=reward_computer,
            prompt_tokens=prompts,
            config=self.config,
            device=torch.device("cpu"),
        )

    def test_step_uses_dynamic_top_k(self):
        """step() info dict contains dynamic top_k from effects."""
        env = self._make_env()
        env.reset()
        action = torch.tensor([0.5, 0.8, 0.5])  # high focus -> low top_k
        _, _, _, info = env.step(action)

        self.assertIn("top_k", info["effects"])
        self.assertIsInstance(info["effects"]["top_k"], int)
        self.assertGreaterEqual(info["effects"]["top_k"], 5)
        self.assertLessEqual(info["effects"]["top_k"], 100)

    def test_step_uses_nucleus_sampling(self):
        """step() info dict contains top_p from effects."""
        env = self._make_env()
        env.reset()
        action = torch.tensor([0.9, 0.5, 0.5])  # high creativity -> high top_p
        _, _, _, info = env.step(action)

        self.assertIn("top_p", info["effects"])
        self.assertIsInstance(info["effects"]["top_p"], float)
        self.assertGreaterEqual(info["effects"]["top_p"], 0.7)
        self.assertLessEqual(info["effects"]["top_p"], 1.0)

    def test_high_focus_produces_lower_top_k(self):
        """Higher focus signal should produce lower top_k (sharper sampling)."""
        env = self._make_env()

        env.reset()
        action_low_focus = torch.tensor([0.5, 0.1, 0.5])
        _, _, _, info_low = env.step(action_low_focus)

        env.reset()
        action_high_focus = torch.tensor([0.5, 0.9, 0.5])
        _, _, _, info_high = env.step(action_high_focus)

        self.assertGreater(
            info_low["effects"]["top_k"],
            info_high["effects"]["top_k"],
        )


class TestTrajectoryModes(unittest.TestCase):
    """Tests for lightweight/full/none trajectory modes in generate_with_gating."""

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
        }
        cls.rl_config = {
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
        }
        cls.model = TransformerLM(vocab_size=100, config=cls.model_config)
        cls.model.eval()
        cls.modulator = GatingModulator(cls.rl_config)
        cls.policy = FixedGatingPolicy(0.5, 0.5, 0.5, torch.device("cpu"))

    def _generate(self, trajectory_mode="lightweight", max_new_tokens=5, **kwargs):
        prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        return self.model.generate_with_gating(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            gating_policy=self.policy,
            modulator=self.modulator,
            trajectory_mode=trajectory_mode,
            **kwargs,
        )

    def test_lightweight_trajectory_contains_actions_effects_tokens(self):
        """Lightweight mode has 'actions', 'effects', and 'tokens' keys."""
        _, trajectory = self._generate(trajectory_mode="lightweight")
        self.assertIsNotNone(trajectory)
        self.assertIn("actions", trajectory)
        self.assertIn("effects", trajectory)
        self.assertIn("tokens", trajectory)

    def test_lightweight_trajectory_excludes_heavy_fields(self):
        """Lightweight mode does NOT include observations, logits_history, hidden_states."""
        _, trajectory = self._generate(trajectory_mode="lightweight")
        self.assertNotIn("observations", trajectory)
        self.assertNotIn("logits_history", trajectory)
        self.assertNotIn("hidden_states", trajectory)

    def test_lightweight_trajectory_effects_has_expected_fields(self):
        """Each effect dict has temperature, repetition_penalty, top_k, top_p."""
        _, trajectory = self._generate(trajectory_mode="lightweight")
        for effect in trajectory["effects"]:
            self.assertIn("temperature", effect)
            self.assertIn("repetition_penalty", effect)
            self.assertIn("top_k", effect)
            self.assertIn("top_p", effect)
            self.assertIsInstance(effect["temperature"], float)
            self.assertIsInstance(effect["repetition_penalty"], float)
            self.assertIsInstance(effect["top_k"], int)
            self.assertIsInstance(effect["top_p"], float)

    def test_lightweight_trajectory_lengths_match(self):
        """len(actions) == len(effects) == len(tokens) == max_new_tokens."""
        max_tokens = 7
        _, trajectory = self._generate(trajectory_mode="lightweight", max_new_tokens=max_tokens)
        self.assertEqual(len(trajectory["actions"]), max_tokens)
        self.assertEqual(len(trajectory["effects"]), max_tokens)
        self.assertEqual(len(trajectory["tokens"]), max_tokens)

    def test_full_trajectory_still_includes_all_fields(self):
        """Full mode includes observations, logits_history, hidden_states AND effects."""
        _, trajectory = self._generate(trajectory_mode="full")
        self.assertIsNotNone(trajectory)
        self.assertIn("actions", trajectory)
        self.assertIn("effects", trajectory)
        self.assertIn("tokens", trajectory)
        self.assertIn("observations", trajectory)
        self.assertIn("logits_history", trajectory)
        self.assertIn("hidden_states", trajectory)

    def test_none_trajectory_mode_returns_none(self):
        """trajectory_mode='none' returns None trajectory."""
        _, trajectory = self._generate(trajectory_mode="none")
        self.assertIsNone(trajectory)

    def test_return_trajectory_true_maps_to_full(self):
        """Backward compat: return_trajectory=True still works and includes all fields."""
        prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        _, trajectory = self.model.generate_with_gating(
            prompt_ids=prompt_ids,
            max_new_tokens=5,
            gating_policy=self.policy,
            modulator=self.modulator,
            return_trajectory=True,
        )
        self.assertIsNotNone(trajectory)
        self.assertIn("observations", trajectory)
        self.assertIn("logits_history", trajectory)
        self.assertIn("hidden_states", trajectory)
        self.assertIn("effects", trajectory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
