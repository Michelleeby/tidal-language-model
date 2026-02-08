"""
HormoneEnvironment.py

Gym-style environment for RL hormone control of the ConstantLanguageModel.
Wraps the language model and provides a standard RL interface with:
- Observation space: 64D vector with token statistics, hidden states, context
- Action space: 3D continuous [catalyst, stress, inhibitor] in [0, 1]
- Rewards: Multi-component reward from RewardComputer

The environment follows the OpenAI Gym interface conventions for compatibility
with standard RL libraries.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random

from HormoneModulator import HormoneModulator, HormoneEffects
from RewardComputer import RewardComputer


@dataclass
class EnvConfig:
    """Configuration for the HormoneEnvironment."""
    max_episode_length: int = 50
    prompt_min_length: int = 3
    prompt_max_length: int = 10
    observation_dim: int = 64
    action_dim: int = 3
    top_k: int = 40
    base_temperature: float = 1.0


class HormoneEnvironment:
    """
    Gym-style environment for RL-controlled hormone generation.

    The environment wraps a ConstantLanguageModel and allows an RL agent
    to control hormone levels at each generation step. The agent receives
    observations about the current state (token statistics, hidden states)
    and outputs hormone actions that modulate generation behavior.

    Observation Space (64D):
        - Token statistics (10D): repetition ratio, entropy, n-gram diversity
        - Hidden state summary (48D): mean/std pooling from final layer
        - Context (6D): step progress, base temperature, previous hormones

    Action Space (3D continuous):
        - [catalyst, stress, inhibitor] each in [0, 1]

    Reward:
        Multi-component reward from RewardComputer (perplexity, diversity,
        repetition, coherence)
    """

    def __init__(
        self,
        model: Any,  # ConstantLanguageModel
        modulator: HormoneModulator,
        reward_computer: RewardComputer,
        prompt_tokens: List[List[int]],
        config: dict,
        device: torch.device = None
    ):
        """
        Initialize the HormoneEnvironment.

        Args:
            model: ConstantLanguageModel instance (frozen for RL training)
            modulator: HormoneModulator for applying hormone effects
            reward_computer: RewardComputer for computing rewards
            prompt_tokens: List of token sequences to use as prompts
            config: Configuration dictionary
            device: Device for tensor operations
        """
        self.model = model
        self.modulator = modulator
        self.reward_computer = reward_computer
        self.prompt_tokens = prompt_tokens
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment configuration
        self.env_config = EnvConfig(
            max_episode_length=config.get("RL_MAX_EPISODE_LENGTH", 50),
            prompt_min_length=config.get("RL_PROMPT_MIN_LENGTH", 3),
            prompt_max_length=config.get("RL_PROMPT_MAX_LENGTH", 10),
            observation_dim=config.get("RL_OBSERVATION_DIM", 64),
            action_dim=config.get("RL_ACTION_DIM", 3),
            top_k=config.get("RL_TOP_K", 40),
            base_temperature=config.get("RL_BASE_TEMPERATURE", 1.0)
        )

        # Episode state
        self.current_prompt: List[int] = []
        self.generated_tokens: List[int] = []
        self.step_count: int = 0
        self.done: bool = False

        # Previous action for observation
        self.prev_action = torch.zeros(3, device=self.device)

        # Ensure model is in eval mode
        self.model.eval()

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Return observation space shape."""
        return (self.env_config.observation_dim,)

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Return action space shape."""
        return (self.env_config.action_dim,)

    def reset(self, prompt_idx: int = None) -> torch.Tensor:
        """
        Reset the environment to start a new episode.

        Args:
            prompt_idx: Optional index of specific prompt to use.
                       If None, randomly samples a prompt.

        Returns:
            Initial observation tensor
        """
        # Select prompt
        if prompt_idx is not None:
            prompt = self.prompt_tokens[prompt_idx % len(self.prompt_tokens)]
        else:
            prompt = random.choice(self.prompt_tokens)

        # Truncate prompt to random length within bounds
        prompt_len = random.randint(
            min(self.env_config.prompt_min_length, len(prompt)),
            min(self.env_config.prompt_max_length, len(prompt))
        )
        self.current_prompt = prompt[:prompt_len]
        self.generated_tokens = list(self.current_prompt)

        # Reset episode state
        self.step_count = 0
        self.done = False
        self.prev_action = torch.zeros(3, device=self.device)

        # Get initial observation
        observation = self._get_observation()

        return observation

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Hormone action tensor of shape (3,) with values in [0, 1]

        Returns:
            observation: Next observation tensor
            reward: Step reward
            done: Whether episode is finished
            info: Additional information dict
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Ensure action is in valid range
        action = action.clamp(0, 1)

        # Get current context
        context = torch.tensor(
            self.generated_tokens[-self.model.max_context_length:],
            dtype=torch.long, device=self.device
        )

        # Forward pass to get logits and hidden states
        with torch.no_grad():
            logits, hidden_states = self.model.forward_with_hidden(context.unsqueeze(0))

        # Apply hormone effects
        effects = self.modulator(action, len(context), self.device)

        # Get next token logits
        next_token_logits = logits[0, -1, :].clone()

        # Apply repetition penalty
        next_token_logits = self.modulator.apply_repetition_penalty_to_logits(
            next_token_logits,
            torch.tensor(self.generated_tokens, dtype=torch.long, device=self.device),
            effects.repetition_penalty
        )

        # Apply temperature
        next_token_logits = next_token_logits / effects.temperature

        # Sample next token
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_k = min(self.env_config.top_k, self.model.vocab_size)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
        new_token = top_k_indices[next_token_idx].item()

        # Compute reward
        reward, reward_components = self.reward_computer.compute_step_reward(
            logits[0, -1, :],
            self.generated_tokens,
            new_token,
            normalize=True
        )

        # Update state
        self.generated_tokens.append(new_token)
        self.step_count += 1
        self.prev_action = action.clone()

        # Check if done
        self.done = self.step_count >= self.env_config.max_episode_length

        # Get next observation
        observation = self._get_observation()

        # Build info dict
        info = {
            "reward_components": reward_components,
            "hormones": {
                "catalyst": float(action[0]),
                "stress": float(action[1]),
                "inhibitor": float(action[2])
            },
            "effects": {
                "temperature": effects.temperature,
                "repetition_penalty": effects.repetition_penalty
            },
            "new_token": new_token,
            "step": self.step_count,
            "sequence_length": len(self.generated_tokens)
        }

        return observation, reward, self.done, info

    def _get_observation(self) -> torch.Tensor:
        """
        Build observation vector from current state.

        Returns:
            Observation tensor of shape (64,)
        """
        # Get context for model forward pass
        context = torch.tensor(
            self.generated_tokens[-self.model.max_context_length:],
            dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            logits, hidden_states = self.model.forward_with_hidden(context.unsqueeze(0))

        # Build observation using model's method
        observation = self.model._build_rl_observation(
            torch.tensor(self.generated_tokens, dtype=torch.long, device=self.device),
            hidden_states,
            logits,
            self.step_count,
            self.env_config.max_episode_length
        )

        # Inject previous action into context features (last 6D of observation)
        # Observation layout: token_stats(10) + hidden_mean(24) + hidden_std(24) + context(6)
        observation[-4:-1] = self.prev_action

        return observation

    def get_generated_text(self, idx_to_vocab: Dict[int, str]) -> str:
        """
        Convert generated tokens to text.

        Args:
            idx_to_vocab: Mapping from token IDs to words

        Returns:
            Generated text string
        """
        tokens = [idx_to_vocab.get(idx, '[UNK]') for idx in self.generated_tokens]
        return ' '.join(tokens)

    def render(self, idx_to_vocab: Dict[int, str] = None) -> str:
        """
        Render current state as string.

        Args:
            idx_to_vocab: Optional mapping for text conversion

        Returns:
            State description string
        """
        state_str = f"Step: {self.step_count}/{self.env_config.max_episode_length}\n"
        state_str += f"Tokens generated: {len(self.generated_tokens) - len(self.current_prompt)}\n"
        state_str += f"Previous hormones: catalyst={self.prev_action[0]:.3f}, "
        state_str += f"stress={self.prev_action[1]:.3f}, inhibitor={self.prev_action[2]:.3f}\n"

        if idx_to_vocab:
            state_str += f"Text: {self.get_generated_text(idx_to_vocab)}\n"

        return state_str


class VectorizedHormoneEnvironment:
    """
    Vectorized environment for parallel episode collection.

    Runs multiple HormoneEnvironment instances in parallel for
    efficient batch sampling during RL training.
    """

    def __init__(
        self,
        model: Any,
        modulator: HormoneModulator,
        reward_computer: RewardComputer,
        prompt_tokens: List[List[int]],
        config: dict,
        num_envs: int = 4,
        device: torch.device = None
    ):
        """
        Initialize vectorized environment.

        Args:
            model: Shared ConstantLanguageModel
            modulator: Shared HormoneModulator
            reward_computer: Shared RewardComputer
            prompt_tokens: List of prompt sequences
            config: Configuration dictionary
            num_envs: Number of parallel environments
            device: Device for tensor operations
        """
        self.num_envs = num_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create individual environments
        self.envs = [
            HormoneEnvironment(
                model=model,
                modulator=modulator,
                reward_computer=reward_computer,
                prompt_tokens=prompt_tokens,
                config=config,
                device=device
            )
            for _ in range(num_envs)
        ]

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            Stacked observations of shape (num_envs, obs_dim)
        """
        observations = [env.reset() for env in self.envs]
        return torch.stack(observations)

    def step(
        self,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Take a step in all environments.

        Args:
            actions: Tensor of shape (num_envs, 3)

        Returns:
            observations: Tensor of shape (num_envs, obs_dim)
            rewards: Tensor of shape (num_envs,)
            dones: Tensor of shape (num_envs,)
            infos: List of info dicts
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.envs):
            if env.done:
                obs = env.reset()
                reward = 0.0
                done = False
                info = {"reset": True}
            else:
                obs, reward, done, info = env.step(actions[i])

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            torch.stack(observations),
            torch.tensor(rewards, device=self.device),
            torch.tensor(dones, device=self.device),
            infos
        )

    def reset_if_done(self) -> torch.Tensor:
        """
        Reset only environments that are done.

        Returns:
            Current observations for all environments
        """
        observations = []
        for env in self.envs:
            if env.done:
                obs = env.reset()
            else:
                obs = env._get_observation()
            observations.append(obs)
        return torch.stack(observations)
