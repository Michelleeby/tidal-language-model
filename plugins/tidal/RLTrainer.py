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
        self.target = config.get("RL_POLICY_ENTROPY_TARGET", -1.0)
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


class RolloutBuffer:
    """Pre-allocated buffer for storing rollout data.

    All tensors are allocated once at init and reused across rollouts.
    ``clear()`` resets the write position without reallocating.
    """

    def __init__(self, capacity: int, obs_dim: int = 64, action_dim: int = 3):
        self.capacity = capacity
        self.observations = torch.zeros(capacity, obs_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.log_probs = torch.zeros(capacity)
        self.dones = torch.zeros(capacity)
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

        self.global_step = 0
        ema_alpha = config.get("RL_EMA_ALPHA", 0.05)
        self.episode_rewards = ExponentialMovingAverage(alpha=ema_alpha)
        obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        action_dim = config.get("RL_ACTION_DIM", 3)
        self.buffer = RolloutBuffer(self.rollout_steps, obs_dim, action_dim)

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

        gate_signals_sum = {"creativity": 0.0, "focus": 0.0, "stability": 0.0}
        reward_components_sum = {"perplexity": 0.0, "diversity": 0.0, "focus": 0.0, "repetition": 0.0, "coherence": 0.0}

        for step in range(num_steps):
            with torch.no_grad():
                obs_tensor = obs.to(self.device)
                action_dist, value = self.agent.forward(obs_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()

                if obs_tensor.dim() == 1:
                    action = action.squeeze(0)
                    value = value.squeeze()

            next_obs, reward, done, info = self.env.step(action)

            self.buffer.add(
                observation=obs_tensor.cpu(),
                action=action.cpu(),
                reward=reward,
                value=value.item() if isinstance(value, torch.Tensor) else value,
                log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                done=done,
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

        return {
            "mean_reward": np.mean(rollout_rewards),
            "mean_value": np.mean(rollout_values),
            "std_reward": np.std(rollout_rewards),
            "mean_gate_creativity": gate_signals_sum["creativity"] / num_steps,
            "mean_gate_focus": gate_signals_sum["focus"] / num_steps,
            "mean_gate_stability": gate_signals_sum["stability"] / num_steps,
            "mean_reward_perplexity": reward_components_sum["perplexity"] / num_steps,
            "mean_reward_diversity": reward_components_sum["diversity"] / num_steps,
            "mean_reward_focus": reward_components_sum["focus"] / num_steps,
            "mean_reward_repetition": reward_components_sum["repetition"] / num_steps,
            "mean_reward_coherence": reward_components_sum["coherence"] / num_steps,
        }

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

    def update_policy(
        self, advantages: torch.Tensor, returns: torch.Tensor,
        current_entropy_coef: float = None,
    ) -> Dict[str, float]:
        n = len(self.buffer)
        observations = self.buffer.observations[:n].to(self.device)
        actions = self.buffer.actions[:n].to(self.device)
        old_log_probs = self.buffer.log_probs[:n].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        entropy_coef = current_entropy_coef if current_entropy_coef is not None else self.entropy_coef

        num_samples = len(self.buffer)
        indices = np.arange(num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
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

                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_obs, batch_actions,
                )

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def train(self, total_timesteps: int = None) -> Dict[str, List[float]]:
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        history = {
            "episode_rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "entropy_coef": [],
            "gate_creativity": [],
            "gate_focus": [],
            "gate_stability": [],
            "explained_variance": [],
            "reward_perplexity": [],
            "reward_diversity": [],
            "reward_focus": [],
            "reward_repetition": [],
            "reward_coherence": [],
        }

        num_iterations = total_timesteps // self.rollout_steps
        print(f"Starting RL training for {total_timesteps} steps ({num_iterations} iterations)")

        for iteration in range(num_iterations):
            rollout_stats = self.collect_rollouts(self.rollout_steps)
            advantages, returns = self.compute_advantages()

            if self.entropy_homeostasis is not None:
                current_entropy_coef = self.entropy_homeostasis.coef
            else:
                current_entropy_coef = self.get_entropy_coef(iteration, num_iterations)

            update_stats = self.update_policy(advantages, returns, current_entropy_coef)

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
            history["gate_creativity"].append(rollout_stats.get("mean_gate_creativity", 0.0))
            history["gate_focus"].append(rollout_stats.get("mean_gate_focus", 0.0))
            history["gate_stability"].append(rollout_stats.get("mean_gate_stability", 0.0))
            history["explained_variance"].append(explained_var)
            history["reward_perplexity"].append(rollout_stats.get("mean_reward_perplexity", 0.0))
            history["reward_diversity"].append(rollout_stats.get("mean_reward_diversity", 0.0))
            history["reward_focus"].append(rollout_stats.get("mean_reward_focus", 0.0))
            history["reward_repetition"].append(rollout_stats.get("mean_reward_repetition", 0.0))
            history["reward_coherence"].append(rollout_stats.get("mean_reward_coherence", 0.0))

            self.writer.add_scalar("RL/episode_reward", mean_ep_reward, self.global_step)
            self.writer.add_scalar("RL/policy_loss", update_stats["policy_loss"], self.global_step)
            self.writer.add_scalar("RL/value_loss", update_stats["value_loss"], self.global_step)
            self.writer.add_scalar("RL/entropy", update_stats["entropy"], self.global_step)
            self.writer.add_scalar("RL/entropy_coef", current_entropy_coef, self.global_step)
            self.writer.add_scalar("RL/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("RL/gate_creativity", rollout_stats.get("mean_gate_creativity", 0.0), self.global_step)
            self.writer.add_scalar("RL/gate_focus", rollout_stats.get("mean_gate_focus", 0.0), self.global_step)
            self.writer.add_scalar("RL/gate_stability", rollout_stats.get("mean_gate_stability", 0.0), self.global_step)
            self.writer.add_scalar("RL/explained_variance", explained_var, self.global_step)
            self.writer.add_scalar("RL/reward_perplexity", rollout_stats.get("mean_reward_perplexity", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_diversity", rollout_stats.get("mean_reward_diversity", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_focus", rollout_stats.get("mean_reward_focus", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_repetition", rollout_stats.get("mean_reward_repetition", 0.0), self.global_step)
            self.writer.add_scalar("RL/reward_coherence", rollout_stats.get("mean_reward_coherence", 0.0), self.global_step)

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
                max_new_tokens=30,
                gating_policy=policy,
                modulator=modulator,
                return_trajectory=True,
            )

            if trajectory:
                ep_rewards = []
                ep_diversity = []
                ep_perplexity = []
                generated_so_far = []

                for logits, token in zip(trajectory["logits_history"], trajectory["tokens"]):
                    reward, comps = reward_computer.compute_step_reward(
                        logits, generated_so_far, token, normalize=False,
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
