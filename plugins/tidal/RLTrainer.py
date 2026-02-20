"""
RLTrainer.py

PPO (Proximal Policy Optimization) training loop for gate signal control.

Implements the standard PPO algorithm with:
- Rollout collection from environments
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration
- Value function loss

The trainer keeps the language model frozen and only trains the RL agent
to control gate signal levels for improved generation quality.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
from datetime import datetime
from .GatingPolicyAgent import GatingPolicyAgent, create_agent
from .GatingEnvironment import GatingEnvironment, VectorizedGatingEnvironment
from .GatingModulator import GatingModulator
from .RewardComputer import RewardComputer


class ExponentialMovingAverage:
    """Exponential moving average tracker for episode statistics.

    Replaces deque(maxlen=N) + np.mean() which creates deterministic
    sliding-window cycling artifacts once the deque is full.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value


class EntropyHomeostasis:
    """Closed-loop homeostatic controller for the entropy coefficient.

    Monitors policy entropy and reactively adjusts the entropy coefficient
    to prevent entropy collapse. When entropy drops below the target, the
    coefficient is boosted (release). Between boosts, it decays toward a
    baseline. Inspired by the SemanticEndocrineSystem's trigger-based
    release + exponential decay pattern.

    Loop:
        1. Release: if entropy < target → coef += release_rate * (target - entropy)
        2. Decay:   coef = decay_rate * coef + (1 - decay_rate) * baseline
        3. Clamp:   coef = clamp(coef, min, max)
    """

    def __init__(self, config: dict):
        self.baseline = config.get("RL_ENTROPY_COEF", 0.01)
        self.coef = self.baseline
        self.target = config.get("RL_POLICY_ENTROPY_TARGET", -0.35)
        self.release_rate = config.get("RL_ENTROPY_HOMEOSTASIS_RELEASE_RATE", 0.05)
        self.decay_rate = config.get("RL_ENTROPY_HOMEOSTASIS_DECAY_RATE", 0.95)
        self.coef_min = config.get("RL_ENTROPY_COEF_MIN", 0.01)
        self.coef_max = config.get("RL_ENTROPY_COEF_MAX", 0.5)

    def step(self, current_entropy: float) -> float:
        """Update coef based on observed policy entropy. Returns new coef."""
        # Release: boost when entropy is too low
        if current_entropy < self.target:
            self.coef += self.release_rate * (self.target - current_entropy)

        # Decay toward baseline
        self.coef = self.decay_rate * self.coef + (1 - self.decay_rate) * self.baseline

        # Clamp
        self.coef = max(self.coef_min, min(self.coef_max, self.coef))

        return self.coef


class DiversityHomeostasis:
    """Closed-loop homeostatic controller for the diversity reward weight.

    Monitors mean diversity reward and reactively adjusts the diversity weight
    to prevent diversity collapse. When diversity drops below the target, the
    weight is boosted (release). Between boosts, it decays toward a baseline.
    Mirrors the EntropyHomeostasis pattern.

    Loop:
        1. Release: if diversity < target → weight += release_rate * (target - diversity)
        2. Decay:   weight = decay_rate * weight + (1 - decay_rate) * baseline
        3. Clamp:   weight = clamp(weight, min, max)
    """

    def __init__(self, config: dict):
        self.baseline = config.get("RL_REWARD_DIVERSITY_WEIGHT", 0.15)
        self.weight = self.baseline
        self.target = config.get("RL_DIVERSITY_HOMEOSTASIS_TARGET", 0.55)
        self.release_rate = config.get("RL_DIVERSITY_HOMEOSTASIS_RELEASE_RATE", 0.03)
        self.decay_rate = config.get("RL_DIVERSITY_HOMEOSTASIS_DECAY_RATE", 0.95)
        self.weight_min = config.get("RL_DIVERSITY_WEIGHT_MIN", 0.15)
        self.weight_max = config.get("RL_DIVERSITY_WEIGHT_MAX", 0.35)

    def step(self, current_diversity: float) -> float:
        """Update weight based on observed diversity. Returns new weight."""
        # Release: boost when diversity is too low
        if current_diversity < self.target:
            self.weight += self.release_rate * (self.target - current_diversity)

        # Decay toward baseline
        self.weight = self.decay_rate * self.weight + (1 - self.decay_rate) * self.baseline

        # Clamp
        self.weight = max(self.weight_min, min(self.weight_max, self.weight))

        return self.weight


class LagrangeMultiplier:
    """Learned Lagrange multiplier for PPO-Lagrangian.

    Parameterized via softplus for non-negativity.
    Updated by gradient ascent on the dual variable:
        λ ← λ + lr * mean_cost

    The cost is max(0, threshold - diversity_reward), so positive cost
    means the diversity constraint is violated (diversity too low).
    """

    def __init__(self, config: dict):
        self.threshold = config.get("RL_DIVERSITY_CONSTRAINT_THRESHOLD", 0.55)
        init_val = config.get("RL_LAGRANGE_MULTIPLIER_INIT", 1.0)
        lr = config.get("RL_LAGRANGE_MULTIPLIER_LR", 0.05)
        # Inverse softplus: raw = log(exp(val) - 1)
        raw_init = math.log(math.exp(init_val) - 1) if init_val > 0 else 0.0
        self.raw_param = torch.nn.Parameter(torch.tensor(raw_init))
        # Weight decay pulls λ toward 0 when constraint is satisfied (gradient=0)
        self.optimizer = optim.Adam([self.raw_param], lr=lr, weight_decay=0.01)

    def value(self) -> float:
        """Current multiplier value (non-negative via softplus)."""
        return torch.nn.functional.softplus(self.raw_param).item()

    def compute_cost(self, diversity_reward: float) -> float:
        """Per-step cost: max(0, threshold - diversity_reward)."""
        return max(0.0, self.threshold - diversity_reward)

    def update(self, mean_cost: float):
        """Dual gradient ascent: increase λ when constraint is violated.

        We maximize λ * (mean_cost - 0) = λ * mean_cost, which means
        gradient ascent on the raw parameter with loss = -λ * mean_cost.
        When mean_cost > 0 (violated), λ increases.
        When mean_cost = 0 (satisfied), λ decreases toward 0.
        """
        self.optimizer.zero_grad()
        lam = torch.nn.functional.softplus(self.raw_param)
        # Dual loss: we want to maximize λ * mean_cost
        # So we minimize -λ * mean_cost
        dual_loss = -lam * mean_cost
        dual_loss.backward()
        self.optimizer.step()

    def state_dict(self) -> dict:
        return {
            "raw_param": self.raw_param.data.clone(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.raw_param.data.copy_(state["raw_param"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])


class RolloutBuffer:
    """Pre-allocated buffer for storing rollout data.

    All tensors are allocated once at init and reused across rollouts.
    ``clear()`` resets the write position without reallocating.
    """

    def __init__(self, capacity: int, obs_dim: int = 64, action_dim: int = 1,
                 store_costs: bool = False):
        self.capacity = capacity
        self.observations = torch.zeros(capacity, obs_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.log_probs = torch.zeros(capacity)
        self.dones = torch.zeros(capacity)
        if store_costs:
            self.costs = torch.zeros(capacity)
            self.cost_values = torch.zeros(capacity)
        self._pos = 0

    def clear(self):
        self._pos = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        cost: float = None,
        cost_value: float = None,
    ):
        if self._pos >= self.capacity:
            raise RuntimeError(
                f"RolloutBuffer overflow: capacity={self.capacity}, pos={self._pos}"
            )
        self.observations[self._pos] = observation
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.values[self._pos] = value
        self.log_probs[self._pos] = log_prob
        self.dones[self._pos] = float(done)
        if cost is not None and hasattr(self, "costs"):
            self.costs[self._pos] = cost
        if cost_value is not None and hasattr(self, "cost_values"):
            self.cost_values[self._pos] = cost_value
        self._pos += 1

    def __len__(self):
        return self._pos


class PPOTrainer:
    """
    PPO trainer for gate signal control.

    Implements the PPO-Clip algorithm for training the gating RL agent
    while keeping the language model frozen.
    """

    def __init__(
        self,
        agent: nn.Module,
        env: GatingEnvironment,
        config: dict,
        experiment_dir: str,
        device: torch.device = None,
        metrics_logger=None,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_logger = metrics_logger

        # PPO hyperparameters
        self.lr = config.get("RL_LEARNING_RATE", 3e-4)
        self.gamma = config.get("RL_GAMMA", 0.99)
        self.gae_lambda = config.get("RL_GAE_LAMBDA", 0.95)
        self.clip_epsilon = config.get("RL_CLIP_EPSILON", 0.2)
        self.entropy_coef = config.get("RL_ENTROPY_COEF", 0.01)
        self.value_coef = config.get("RL_VALUE_COEF", 0.5)
        self.max_grad_norm = config.get("RL_MAX_GRAD_NORM", 0.5)

        self.rollout_steps = config.get("RL_ROLLOUT_STEPS", 128)
        self.num_epochs = config.get("RL_NUM_EPOCHS", 4)
        self.batch_size = config.get("RL_BATCH_SIZE", 32)
        self.total_timesteps = config.get("RL_TOTAL_TIMESTEPS", 100000)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)

        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.total_timesteps // self.rollout_steps,
        )

        self.writer = SummaryWriter(os.path.join(experiment_dir, "tensorboard_logs"))
        self.metrics_dir = os.path.join(experiment_dir, "rl_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.entropy_coef_final = config.get("RL_ENTROPY_COEF_FINAL", self.entropy_coef)

        entropy_schedule = config.get("RL_ENTROPY_SCHEDULE", "linear")
        if entropy_schedule == "homeostasis":
            self.entropy_homeostasis = EntropyHomeostasis(config)
        else:
            self.entropy_homeostasis = None

        # Constraint mode: "weighted" (default) or "lagrangian"
        self.constraint_mode = config.get("RL_CONSTRAINT_MODE", "weighted")

        if self.constraint_mode == "lagrangian":
            self.lagrange_multiplier = LagrangeMultiplier(config)
            # Lagrangian mode supersedes diversity homeostasis
            self.diversity_homeostasis = None
            # Zero out diversity and repetition weights; renormalize remaining
            self.env.reward_computer.diversity_weight = 0.0
            self.env.reward_computer.repetition_weight = 0.0
            remaining = (
                self.env.reward_computer.perplexity_weight
                + self.env.reward_computer.sampling_weight
                + self.env.reward_computer.coherence_weight
            )
            if remaining > 0:
                self.env.reward_computer.perplexity_weight /= remaining
                self.env.reward_computer.sampling_weight /= remaining
                self.env.reward_computer.coherence_weight /= remaining
        else:
            self.lagrange_multiplier = None
            if "RL_DIVERSITY_HOMEOSTASIS_TARGET" in config:
                self.diversity_homeostasis = DiversityHomeostasis(config)
            else:
                self.diversity_homeostasis = None

        self.global_step = 0
        ema_alpha = config.get("RL_EMA_ALPHA", 0.05)
        self.episode_rewards = ExponentialMovingAverage(alpha=ema_alpha)
        obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        action_dim = config.get("RL_ACTION_DIM", 1)
        store_costs = (self.constraint_mode == "lagrangian")
        self.buffer = RolloutBuffer(self.rollout_steps, obs_dim, action_dim,
                                    store_costs=store_costs)

        # Persist episode counters across rollout boundaries so episodes
        # that span two collect_rollouts() calls get their true length.
        self._episode_reward = 0.0
        self._episode_length = 0

    def get_entropy_coef(self, iteration: int, total_iterations: int) -> float:
        """Linear interpolation from initial to final entropy coefficient."""
        fraction = iteration / max(total_iterations - 1, 1)
        return self.entropy_coef + fraction * (self.entropy_coef_final - self.entropy_coef)

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        self.buffer.clear()
        self.agent.eval()

        if not hasattr(self.env, "generated_tokens") or len(self.env.generated_tokens) == 0 or self.env.done:
            obs = self.env.reset()
            self._episode_reward = 0.0
            self._episode_length = 0
        else:
            obs = self.env._get_observation()

        rollout_rewards = []
        rollout_values = []
        rollout_costs = []

        gate_signals_sum = {"modulation": 0.0}
        reward_components_sum = {"perplexity": 0.0, "diversity": 0.0, "sampling": 0.0, "repetition": 0.0, "coherence": 0.0}

        for step in range(num_steps):
            with torch.no_grad():
                obs_tensor = obs.to(self.device)

                if self.constraint_mode == "lagrangian":
                    action_dist, value, cost_value = self.agent.forward_with_cost(obs_tensor)
                else:
                    action_dist, value = self.agent.forward(obs_tensor)
                    cost_value = None

                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()

                if obs_tensor.dim() == 1:
                    action = action.squeeze(0)
                    value = value.squeeze()
                    if cost_value is not None:
                        cost_value = cost_value.squeeze()

            next_obs, reward, done, info = self.env.step(action)

            # Compute per-step cost for Lagrangian mode
            step_cost = None
            step_cost_value = None
            if self.constraint_mode == "lagrangian" and self.lagrange_multiplier is not None:
                diversity_reward = info.get("reward_components", {}).get("diversity", 0.0)
                step_cost = self.lagrange_multiplier.compute_cost(diversity_reward)
                step_cost_value = cost_value.item() if isinstance(cost_value, torch.Tensor) else cost_value
                rollout_costs.append(step_cost)

            self.buffer.add(
                observation=obs_tensor.cpu(),
                action=action.cpu(),
                reward=reward,
                value=value.item() if isinstance(value, torch.Tensor) else value,
                log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                done=done,
                cost=step_cost,
                cost_value=step_cost_value,
            )

            rollout_rewards.append(reward)
            rollout_values.append(value.item() if isinstance(value, torch.Tensor) else value)

            if "gate_signals" in info:
                for key in gate_signals_sum:
                    gate_signals_sum[key] += info["gate_signals"].get(key, 0.0)
            if "reward_components" in info:
                for key in reward_components_sum:
                    reward_components_sum[key] += info["reward_components"].get(key, 0.0)

            self._episode_reward += reward
            self._episode_length += 1

            if done:
                self.episode_rewards.update(self._episode_reward)
                self._episode_reward = 0.0
                self._episode_length = 0
                obs = self.env.reset()
            else:
                obs = next_obs

            self.global_step += 1

        self.agent.train()

        stats = {
            "mean_reward": np.mean(rollout_rewards),
            "mean_value": np.mean(rollout_values),
            "std_reward": np.std(rollout_rewards),
            "mean_gate_modulation": gate_signals_sum["modulation"] / num_steps,
            "mean_reward_perplexity": reward_components_sum["perplexity"] / num_steps,
            "mean_reward_diversity": reward_components_sum["diversity"] / num_steps,
            "mean_reward_sampling": reward_components_sum["sampling"] / num_steps,
            "mean_reward_repetition": reward_components_sum["repetition"] / num_steps,
            "mean_reward_coherence": reward_components_sum["coherence"] / num_steps,
        }
        if rollout_costs:
            stats["mean_cost"] = np.mean(rollout_costs)
        return stats

    def compute_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(self.buffer)
        rewards = self.buffer.rewards[:n]
        values = self.buffer.values[:n]
        dones = self.buffer.dones[:n]

        with torch.no_grad():
            last_obs = self.buffer.observations[n - 1].to(self.device)
            _, last_value = self.agent.forward(last_obs)
            last_value = last_value.squeeze().cpu()

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_cost_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE-Lambda over cost values. Does NOT normalize cost advantages."""
        n = len(self.buffer)
        costs = self.buffer.costs[:n]
        cost_values = self.buffer.cost_values[:n]
        dones = self.buffer.dones[:n]

        with torch.no_grad():
            last_obs = self.buffer.observations[n - 1].to(self.device)
            last_cost_value = self.agent.get_cost_value(last_obs).cpu()

        advantages = torch.zeros_like(costs)
        last_gae = 0.0

        for t in reversed(range(len(costs))):
            if t == len(costs) - 1:
                next_value = last_cost_value
            else:
                next_value = cost_values[t + 1]

            delta = costs[t] + self.gamma * next_value * (1 - dones[t]) - cost_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + cost_values
        # Cost advantages are NOT normalized (different from reward advantages)
        return advantages, returns

    def update_policy(
        self, advantages: torch.Tensor, returns: torch.Tensor,
        current_entropy_coef: float = None,
        cost_advantages: torch.Tensor = None,
        cost_returns: torch.Tensor = None,
    ) -> Dict[str, float]:
        n = len(self.buffer)
        observations = self.buffer.observations[:n].to(self.device)
        actions = self.buffer.actions[:n].to(self.device)
        old_log_probs = self.buffer.log_probs[:n].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        lagrangian = (self.constraint_mode == "lagrangian"
                      and cost_advantages is not None
                      and cost_returns is not None)
        if lagrangian:
            cost_advantages = cost_advantages.to(self.device)
            cost_returns = cost_returns.to(self.device)
            lam = self.lagrange_multiplier.value()

        entropy_coef = current_entropy_coef if current_entropy_coef is not None else self.entropy_coef

        num_samples = len(self.buffer)
        indices = np.arange(num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_cost_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                if lagrangian:
                    log_probs, values, entropy, cost_values = (
                        self.agent.evaluate_actions_with_cost(batch_obs, batch_actions)
                    )
                    batch_cost_advantages = cost_advantages[batch_indices]
                    batch_cost_returns = cost_returns[batch_indices]
                    # Combined advantage: (A_r - λ * A_c) / (1 + λ)
                    combined_advantages = (
                        (batch_advantages - lam * batch_cost_advantages) / (1 + lam)
                    )
                else:
                    log_probs, values, entropy = self.agent.evaluate_actions(
                        batch_obs, batch_actions,
                    )
                    combined_advantages = batch_advantages

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * combined_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * combined_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_coef * entropy_loss
                )

                if lagrangian:
                    cost_value_loss = nn.functional.mse_loss(cost_values, batch_cost_returns)
                    loss = loss + self.value_coef * cost_value_loss
                    total_cost_value_loss += cost_value_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        result = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }
        if lagrangian:
            result["cost_value_loss"] = total_cost_value_loss / num_updates
        return result

    def train(self, total_timesteps: int = None) -> Dict[str, List[float]]:
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        history = {
            "episode_rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "entropy_coef": [],
            "gate_modulation": [],
            "explained_variance": [],
            "reward_perplexity": [],
            "reward_diversity": [],
            "reward_sampling": [],
            "reward_repetition": [],
            "reward_coherence": [],
            "diversity_weight": [],
            "lagrange_multiplier": [],
            "mean_cost": [],
            "cost_value_loss": [],
        }

        num_iterations = total_timesteps // self.rollout_steps
        print(f"Starting RL training for {total_timesteps} steps ({num_iterations} iterations)")

        for iteration in range(num_iterations):
            rollout_stats = self.collect_rollouts(self.rollout_steps)

            if self.diversity_homeostasis is not None:
                current_diversity_weight = self.diversity_homeostasis.step(
                    rollout_stats["mean_reward_diversity"]
                )
                self.env.reward_computer.diversity_weight = current_diversity_weight
            else:
                current_diversity_weight = None

            advantages, returns = self.compute_advantages()

            # Lagrangian: compute cost advantages and update dual variable
            cost_adv = None
            cost_ret = None
            if self.constraint_mode == "lagrangian" and self.lagrange_multiplier is not None:
                cost_adv, cost_ret = self.compute_cost_advantages()
                mean_cost = rollout_stats.get("mean_cost", 0.0)
                self.lagrange_multiplier.update(mean_cost)

            if self.entropy_homeostasis is not None:
                current_entropy_coef = self.entropy_homeostasis.coef
            else:
                current_entropy_coef = self.get_entropy_coef(iteration, num_iterations)

            update_stats = self.update_policy(
                advantages, returns, current_entropy_coef,
                cost_advantages=cost_adv, cost_returns=cost_ret,
            )

            if self.entropy_homeostasis is not None:
                current_entropy_coef = self.entropy_homeostasis.step(update_stats["entropy"])

            self.lr_scheduler.step()

            # Explained variance: 1 - Var(returns - values) / Var(returns)
            n = len(self.buffer)
            values = self.buffer.values[:n]
            var_returns = returns.var().item()
            if var_returns < 1e-8:
                explained_var = 0.0
            else:
                explained_var = 1.0 - (returns - values).var().item() / var_returns
            explained_var = max(-1.0, min(1.0, explained_var))

            mean_ep_reward = self.episode_rewards.value if self.episode_rewards.value is not None else 0.0

            history["episode_rewards"].append(mean_ep_reward)
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])
            history["entropy_coef"].append(current_entropy_coef)
            history["gate_modulation"].append(rollout_stats.get("mean_gate_modulation", 0.0))
            history["explained_variance"].append(explained_var)
            history["reward_perplexity"].append(rollout_stats.get("mean_reward_perplexity", 0.0))
            history["reward_diversity"].append(rollout_stats.get("mean_reward_diversity", 0.0))
            history["reward_sampling"].append(rollout_stats.get("mean_reward_sampling", 0.0))
            history["reward_repetition"].append(rollout_stats.get("mean_reward_repetition", 0.0))
            history["reward_coherence"].append(rollout_stats.get("mean_reward_coherence", 0.0))
            if current_diversity_weight is not None:
                history["diversity_weight"].append(current_diversity_weight)
            if self.lagrange_multiplier is not None:
                history["lagrange_multiplier"].append(self.lagrange_multiplier.value())
                history["mean_cost"].append(rollout_stats.get("mean_cost", 0.0))
                history["cost_value_loss"].append(update_stats.get("cost_value_loss", 0.0))

            self.writer.add_scalar("RL/episode_reward", mean_ep_reward, self.global_step)
            self.writer.add_scalar("RL/policy_loss", update_stats["policy_loss"], self.global_step)
            self.writer.add_scalar("RL/value_loss", update_stats["value_loss"], self.global_step)
            self.writer.add_scalar("RL/entropy", update_stats["entropy"], self.global_step)
            self.writer.add_scalar("RL/entropy_coef", current_entropy_coef, self.global_step)
            self.writer.add_scalar("RL/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("RL/gate_modulation", rollout_stats.get("mean_gate_modulation", 0.0), self.global_step)
            self.writer.add_scalar("RL/explained_variance", explained_var, self.global_step)
            self.writer.add_scalar("RL/reward_perplexity", rollout_stats.get("mean_reward_perplexity", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_diversity", rollout_stats.get("mean_reward_diversity", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_sampling", rollout_stats.get("mean_reward_sampling", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_repetition", rollout_stats.get("mean_reward_repetition", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_coherence", rollout_stats.get("mean_reward_coherence", 0.0), self.global_step)
            if current_diversity_weight is not None:
                self.writer.add_scalar("RL/diversity_weight", current_diversity_weight, self.global_step)
            if self.lagrange_multiplier is not None:
                self.writer.add_scalar("RL/lagrange_multiplier", self.lagrange_multiplier.value(), self.global_step)
                self.writer.add_scalar("RL/mean_cost", rollout_stats.get("mean_cost", 0.0), self.global_step)
                if "cost_value_loss" in update_stats:
                    self.writer.add_scalar("RL/cost_value_loss", update_stats["cost_value_loss"], self.global_step)

            if (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{num_iterations} | "
                    f"Steps: {self.global_step} | "
                    f"Reward: {mean_ep_reward:.3f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                    f"Value Loss: {update_stats['value_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.4f} | "
                    f"Entropy Coef: {current_entropy_coef:.4f}"
                )

            if (iteration + 1) % 50 == 0:
                self._save_metrics(history)

            if (iteration + 1) % 100 == 0:
                self.save_checkpoint(f"rl_checkpoint_iter_{iteration + 1}.pth")

        self._save_metrics(history)
        self.save_checkpoint("rl_checkpoint_final.pth")
        self.writer.close()

        return history

    def _save_metrics(self, history: Dict[str, List[float]]):
        metrics_file = os.path.join(self.metrics_dir, "rl_training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "history": history,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )
        if self.metrics_logger is not None:
            self.metrics_logger.log_rl_metrics(history, self.global_step)

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        checkpoint_dict = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        if self.entropy_homeostasis is not None:
            checkpoint_dict["entropy_homeostasis_coef"] = self.entropy_homeostasis.coef
        if self.diversity_homeostasis is not None:
            checkpoint_dict["diversity_homeostasis_weight"] = self.diversity_homeostasis.weight
        if self.lagrange_multiplier is not None:
            checkpoint_dict["lagrange_multiplier_state"] = self.lagrange_multiplier.state_dict()
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        if self.metrics_logger is not None:
            self.metrics_logger.upload_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        if self.entropy_homeostasis is not None and "entropy_homeostasis_coef" in checkpoint:
            self.entropy_homeostasis.coef = checkpoint["entropy_homeostasis_coef"]
        if self.diversity_homeostasis is not None and "diversity_homeostasis_weight" in checkpoint:
            self.diversity_homeostasis.weight = checkpoint["diversity_homeostasis_weight"]
            self.env.reward_computer.diversity_weight = checkpoint["diversity_homeostasis_weight"]
        if self.lagrange_multiplier is not None and "lagrange_multiplier_state" in checkpoint:
            self.lagrange_multiplier.load_state_dict(checkpoint["lagrange_multiplier_state"])
        print(f"Loaded checkpoint from step {self.global_step}")


def run_ablation_study(
    model: Any,
    prompt_tokens: List[List[int]],
    config: dict,
    experiment_dir: str,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study comparing learned vs baseline gating policies.
    """
    # Lazy import: only needed for optional ablation study path
    from .GatingModulator import RandomGatingPolicy, FixedGatingPolicy, NeutralGatingPolicy

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    policies = {
        "random": RandomGatingPolicy(device),
        "fixed": FixedGatingPolicy(device=device),
        "neutral_gating": NeutralGatingPolicy(device),
    }

    trained_checkpoint = os.path.join(experiment_dir, "rl_checkpoint_final.pth")
    if os.path.exists(trained_checkpoint):
        agent = create_agent(config, device)
        checkpoint = torch.load(trained_checkpoint, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()
        policies["learned"] = agent
    else:
        print("Warning: No trained RL checkpoint found. Skipping learned policy evaluation.")

    modulator = GatingModulator(config)
    reward_computer = RewardComputer(config, model.vocab_size)

    for policy_name, policy in policies.items():
        print(f"\nEvaluating {policy_name} policy...")

        total_rewards = []
        total_diversity = []
        total_perplexity = []

        num_eval_episodes = config.get("RL_EVAL_EPISODES", 50)

        for ep in range(num_eval_episodes):
            prompt_idx = ep % len(prompt_tokens)
            prompt = prompt_tokens[prompt_idx][:5]

            generated, trajectory = model.generate_with_gating(
                prompt_ids=torch.tensor(prompt, device=device),
                max_new_tokens=config.get("RL_MAX_EPISODE_LENGTH", 50),
                gating_policy=policy,
                modulator=modulator,
                return_trajectory=True,
            )

            if trajectory:
                ep_rewards = []
                ep_diversity = []
                ep_perplexity = []
                generated_so_far = []

                sampling_entropies = trajectory.get("sampling_entropy", [])
                for i, (logits, token) in enumerate(zip(trajectory["logits_history"], trajectory["tokens"])):
                    se = sampling_entropies[i] if i < len(sampling_entropies) else None
                    reward, comps = reward_computer.compute_step_reward(
                        logits, generated_so_far, token, normalize=False,
                        sampling_entropy=se,
                    )
                    ep_rewards.append(reward)
                    ep_diversity.append(comps["diversity"])
                    ep_perplexity.append(comps["perplexity"])
                    generated_so_far.append(token)

                total_rewards.append(np.mean(ep_rewards))
                total_diversity.append(np.mean(ep_diversity))
                total_perplexity.append(np.mean(ep_perplexity))

        results[policy_name] = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_diversity": np.mean(total_diversity),
            "mean_perplexity": np.mean(total_perplexity),
        }

        print(f"  Mean reward: {results[policy_name]['mean_reward']:.4f} +/- {results[policy_name]['std_reward']:.4f}")

    results_file = os.path.join(experiment_dir, "ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {results_file}")

    return results
