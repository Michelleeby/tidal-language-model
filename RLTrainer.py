"""
RLTrainer.py

PPO (Proximal Policy Optimization) training loop for hormone control.

Implements the standard PPO algorithm with:
- Rollout collection from vectorized environments
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration
- Value function loss

The trainer keeps the language model frozen and only trains the RL agent
to control hormone levels for improved generation quality.
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
from collections import deque

from HormoneRLAgent import HormoneRLAgent, create_agent
from HormoneEnvironment import HormoneEnvironment, VectorizedHormoneEnvironment
from HormoneModulator import HormoneModulator
from RewardComputer import RewardComputer


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[torch.Tensor]
    actions: List[torch.Tensor]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]

    def __init__(self):
        self.clear()

    def clear(self):
        """Clear all stored data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO trainer for hormone control.

    Implements the PPO-Clip algorithm for training the hormone RL agent
    while keeping the language model frozen.
    """

    def __init__(
        self,
        agent: nn.Module,
        env: HormoneEnvironment,
        config: dict,
        experiment_dir: str,
        device: torch.device = None
    ):
        """
        Initialize the PPO trainer.

        Args:
            agent: HormoneRLAgent to train
            env: HormoneEnvironment or VectorizedHormoneEnvironment
            config: Configuration dictionary with PPO hyperparameters
            experiment_dir: Directory for saving checkpoints and logs
            device: Device for tensor operations
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PPO hyperparameters
        self.lr = config.get("RL_LEARNING_RATE", 3e-4)
        self.gamma = config.get("RL_GAMMA", 0.99)
        self.gae_lambda = config.get("RL_GAE_LAMBDA", 0.95)
        self.clip_epsilon = config.get("RL_CLIP_EPSILON", 0.2)
        self.entropy_coef = config.get("RL_ENTROPY_COEF", 0.01)
        self.value_coef = config.get("RL_VALUE_COEF", 0.5)
        self.max_grad_norm = config.get("RL_MAX_GRAD_NORM", 0.5)

        # Training parameters
        self.rollout_steps = config.get("RL_ROLLOUT_STEPS", 128)
        self.num_epochs = config.get("RL_NUM_EPOCHS", 4)
        self.batch_size = config.get("RL_BATCH_SIZE", 32)
        self.total_timesteps = config.get("RL_TOTAL_TIMESTEPS", 100000)

        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)

        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.total_timesteps // self.rollout_steps
        )

        # Logging
        self.writer = SummaryWriter(os.path.join(experiment_dir, "tensorboard_logs"))
        self.metrics_dir = os.path.join(experiment_dir, "rl_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """
        Collect rollout data from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dict with rollout statistics
        """
        self.buffer.clear()
        self.agent.eval()

        # Always reset if the env is done or hasn't been started yet
        if not hasattr(self.env, 'generated_tokens') or len(self.env.generated_tokens) == 0 or self.env.done:
            obs = self.env.reset()
        else:
            obs = self.env._get_observation()

        episode_reward = 0.0
        episode_length = 0

        rollout_rewards = []
        rollout_values = []

        for step in range(num_steps):
            with torch.no_grad():
                obs_tensor = obs.to(self.device)
                action_dist, value = self.agent.forward(obs_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()

                if obs_tensor.dim() == 1:
                    action = action.squeeze(0)
                    value = value.squeeze()

            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)

            # Store transition
            self.buffer.add(
                observation=obs_tensor.cpu(),
                action=action.cpu(),
                reward=reward,
                value=value.item() if isinstance(value, torch.Tensor) else value,
                log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                done=done
            )

            rollout_rewards.append(reward)
            rollout_values.append(value.item() if isinstance(value, torch.Tensor) else value)

            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_reward = 0.0
                episode_length = 0
                obs = self.env.reset()
            else:
                obs = next_obs

            self.global_step += 1

        self.agent.train()

        return {
            "mean_reward": np.mean(rollout_rewards),
            "mean_value": np.mean(rollout_values),
            "std_reward": np.std(rollout_rewards)
        }

    def compute_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Returns:
            advantages: Tensor of shape (num_steps,)
            returns: Tensor of shape (num_steps,)
        """
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        values = torch.tensor(self.buffer.values, dtype=torch.float32)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)

        # Get bootstrap value for last state
        with torch.no_grad():
            last_obs = self.buffer.observations[-1].to(self.device)
            _, last_value = self.agent.forward(last_obs)
            last_value = last_value.squeeze().cpu()

        # Compute GAE
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

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO policy update.

        Args:
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            Dict with training metrics
        """
        # Prepare data
        observations = torch.stack(self.buffer.observations).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, device=self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

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

                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Compute policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Compute entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Gradient update
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
            "entropy": total_entropy / num_updates
        }

    def train(self, total_timesteps: int = None) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            total_timesteps: Total environment steps to train for

        Returns:
            Dict with training history
        """
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }

        num_iterations = total_timesteps // self.rollout_steps
        print(f"Starting RL training for {total_timesteps} steps ({num_iterations} iterations)")

        for iteration in range(num_iterations):
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.rollout_steps)

            # Compute advantages
            advantages, returns = self.compute_advantages()

            # Update policy
            update_stats = self.update_policy(advantages, returns)

            # Update learning rate
            self.lr_scheduler.step()

            # Logging
            if len(self.episode_rewards) > 0:
                mean_ep_reward = np.mean(self.episode_rewards)
                mean_ep_length = np.mean(self.episode_lengths)
            else:
                mean_ep_reward = 0.0
                mean_ep_length = 0.0

            # Record history
            history["episode_rewards"].append(mean_ep_reward)
            history["episode_lengths"].append(mean_ep_length)
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])

            # TensorBoard logging
            self.writer.add_scalar("RL/episode_reward", mean_ep_reward, self.global_step)
            self.writer.add_scalar("RL/episode_length", mean_ep_length, self.global_step)
            self.writer.add_scalar("RL/policy_loss", update_stats["policy_loss"], self.global_step)
            self.writer.add_scalar("RL/value_loss", update_stats["value_loss"], self.global_step)
            self.writer.add_scalar("RL/entropy", update_stats["entropy"], self.global_step)
            self.writer.add_scalar("RL/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)

            # Console logging
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations} | "
                      f"Steps: {self.global_step} | "
                      f"Reward: {mean_ep_reward:.3f} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f}")

            # Save metrics for dashboard
            if (iteration + 1) % 50 == 0:
                self._save_metrics(history)

            # Checkpoint
            if (iteration + 1) % 100 == 0:
                self.save_checkpoint(f"rl_checkpoint_iter_{iteration + 1}.pth")

        # Final save
        self._save_metrics(history)
        self.save_checkpoint("rl_checkpoint_final.pth")
        self.writer.close()

        return history

    def _save_metrics(self, history: Dict[str, List[float]]):
        """Save metrics to JSON for dashboard."""
        metrics_file = os.path.join(self.metrics_dir, "rl_training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                "global_step": self.global_step,
                "history": history,
                "timestamp": datetime.now().isoformat()
            }, f)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        torch.save({
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint from step {self.global_step}")


def run_ablation_study(
    model: Any,
    prompt_tokens: List[List[int]],
    config: dict,
    experiment_dir: str,
    device: torch.device = None
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study comparing learned vs baseline policies.

    Compares:
    1. Learned policy (trained RL agent)
    2. Random hormones (uniform [0, 1])
    3. Fixed hormones (constant 0.5)
    4. No hormones (base model)

    Args:
        model: ConstantLanguageModel
        prompt_tokens: List of prompt sequences for evaluation
        config: Configuration dictionary
        experiment_dir: Directory for results
        device: Device for computation

    Returns:
        Dict with results for each policy type
    """
    from HormoneModulator import RandomHormonePolicy, FixedHormonePolicy, NoHormonePolicy

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # Policies to evaluate
    policies = {
        "random": RandomHormonePolicy(device),
        "fixed": FixedHormonePolicy(device=device),
        "no_hormone": NoHormonePolicy(device)
    }

    # Load trained agent if available
    trained_checkpoint = os.path.join(experiment_dir, "rl_checkpoint_final.pth")
    if os.path.exists(trained_checkpoint):
        agent = create_agent(config, device)
        checkpoint = torch.load(trained_checkpoint, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()
        policies["learned"] = agent
    else:
        print("Warning: No trained RL checkpoint found. Skipping learned policy evaluation.")

    modulator = HormoneModulator(config)
    reward_computer = RewardComputer(config, model.vocab_size)

    # Evaluate each policy
    for policy_name, policy in policies.items():
        print(f"\nEvaluating {policy_name} policy...")

        total_rewards = []
        total_diversity = []
        total_perplexity = []

        num_eval_episodes = config.get("RL_EVAL_EPISODES", 50)

        for ep in range(num_eval_episodes):
            prompt_idx = ep % len(prompt_tokens)
            prompt = prompt_tokens[prompt_idx][:5]

            # Generate with this policy
            generated, trajectory = model.generate_with_hormones(
                prompt_ids=torch.tensor(prompt, device=device),
                max_new_tokens=30,
                hormone_policy=policy,
                modulator=modulator,
                return_trajectory=True
            )

            if trajectory:
                # Compute raw (unnormalized) rewards for fair comparison
                ep_rewards = []
                ep_diversity = []
                ep_perplexity = []
                generated_so_far = []

                for logits, token in zip(trajectory["logits_history"], trajectory["tokens"]):
                    reward, comps = reward_computer.compute_step_reward(
                        logits, generated_so_far, token, normalize=False
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
            "mean_perplexity": np.mean(total_perplexity)
        }

        print(f"  Mean reward: {results[policy_name]['mean_reward']:.4f} +/- {results[policy_name]['std_reward']:.4f}")

    # Save results
    results_file = os.path.join(experiment_dir, "ablation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {results_file}")

    return results
