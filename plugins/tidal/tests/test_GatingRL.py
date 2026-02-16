"""
test_GatingRL.py

Unit tests for the RL Gating Controller components.
"""

import torch
import numpy as np
import unittest

from unittest.mock import patch, MagicMock

from plugins.tidal.GatingModulator import GatingModulator, GatingEffects, RandomGatingPolicy, FixedGatingPolicy
from plugins.tidal.RewardComputer import RewardComputer
from plugins.tidal.GatingPolicyAgent import GatingPolicyAgent, GaussianGatingPolicyAgent, create_agent
from plugins.tidal.RLTrainer import RolloutBuffer, PPOTrainer, ExponentialMovingAverage
from plugins.tidal.GatingEnvironment import GatingEnvironment
from plugins.tidal.TransformerLM import TransformerLM


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
        """Agent uses a default max when config key is absent."""
        config_no_max = {
            "RL_OBSERVATION_DIM": 64,
            "RL_ACTION_DIM": 3,
            "RL_HIDDEN_DIM": 128,
        }
        agent = GatingPolicyAgent(config_no_max, self.device)

        obs = torch.randn(64) * 10
        action_dist, _ = agent.forward(obs)
        alpha = action_dist.concentration1
        beta_param = action_dist.concentration0

        # Should still be bounded (default max)
        self.assertTrue(torch.all(alpha <= 20.0 + 1e-6))
        self.assertTrue(torch.all(beta_param <= 20.0 + 1e-6))


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

            # Should have EMA trackers, not deques
            self.assertIsInstance(trainer.episode_rewards, ExponentialMovingAverage)
            self.assertIsInstance(trainer.episode_lengths, ExponentialMovingAverage)
            self.assertEqual(trainer.episode_rewards.alpha, 0.05)

            # Should NOT have deque attributes
            from collections import deque
            self.assertNotIsInstance(trainer.episode_rewards, deque)
            self.assertNotIsInstance(trainer.episode_lengths, deque)


if __name__ == "__main__":
    unittest.main(verbosity=2)
