"""
GatingEnvironment.py

Gym-style environment for RL gate signal control of the TransformerLM.
Wraps the language model and provides a standard RL interface with:
- Observation space: 64D vector with token statistics, hidden states, context
- Action space: 3D continuous [creativity, focus, stability] in [0, 1]
- Rewards: Multi-component reward from RewardComputer

Supports both inference-time (generation) and training-time (backprop through model)
modes.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random

from .GatingModulator import GatingModulator, GatingEffects
from .RewardComputer import RewardComputer


@dataclass
class EnvConfig:
    """Configuration for the GatingEnvironment."""
    max_episode_length: int = 50
    prompt_min_length: int = 3
    prompt_max_length: int = 10
    observation_dim: int = 64
    action_dim: int = 3
    top_k: int = 40
    base_temperature: float = 1.0


class GatingEnvironment:
    """
    Gym-style environment for RL-controlled gated generation.

    The environment wraps a TransformerLM and allows an RL agent to control
    gate signal levels at each generation step. The agent receives observations
    about the current state (token statistics, hidden states) and outputs
    gate signal actions that modulate generation behavior.

    Observation Space (64D):
        - Token statistics (10D): repetition ratio, entropy, n-gram diversity
        - Hidden state summary (48D): mean/std pooling from final layer
        - Context (6D): step progress, base temperature, previous gate signals

    Action Space (3D continuous):
        - [creativity, focus, stability] each in [0, 1]

    Reward:
        Multi-component reward from RewardComputer (perplexity, diversity,
        repetition, coherence)
    """

    def __init__(
        self,
        model: Any,  # TransformerLM
        modulator: GatingModulator,
        reward_computer: RewardComputer,
        prompt_tokens: List[List[int]],
        config: dict,
        device: torch.device = None,
        training_mode: bool = False,
    ):
        """
        Args:
            model: TransformerLM instance (frozen for RL training)
            modulator: GatingModulator for applying gate signal effects
            reward_computer: RewardComputer for computing rewards
            prompt_tokens: List of token sequences to use as prompts
            config: Configuration dictionary
            device: Device for tensor operations
            training_mode: If True, backprop through model during step()
        """
        self.model = model
        self.modulator = modulator
        self.reward_computer = reward_computer
        self.prompt_tokens = prompt_tokens
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_mode = training_mode

        self.env_config = EnvConfig(
            max_episode_length=config.get("RL_MAX_EPISODE_LENGTH", 50),
            prompt_min_length=config.get("RL_PROMPT_MIN_LENGTH", 3),
            prompt_max_length=config.get("RL_PROMPT_MAX_LENGTH", 10),
            observation_dim=config.get("RL_OBSERVATION_DIM", 64),
            action_dim=config.get("RL_ACTION_DIM", 3),
            top_k=config.get("RL_TOP_K", 40),
            base_temperature=config.get("RL_BASE_TEMPERATURE", 1.0),
        )

        self.current_prompt: List[int] = []
        self.generated_tokens: List[int] = []
        self.step_count: int = 0
        self.done: bool = False
        self.prev_action = torch.zeros(3, device=self.device)

        self.model.eval()

    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (self.env_config.observation_dim,)

    @property
    def action_space_shape(self) -> Tuple[int]:
        return (self.env_config.action_dim,)

    def reset(self, prompt_idx: int = None) -> torch.Tensor:
        if prompt_idx is not None:
            prompt = self.prompt_tokens[prompt_idx % len(self.prompt_tokens)]
        else:
            prompt = random.choice(self.prompt_tokens)

        prompt_len = random.randint(
            min(self.env_config.prompt_min_length, len(prompt)),
            min(self.env_config.prompt_max_length, len(prompt)),
        )
        self.current_prompt = prompt[:prompt_len]
        self.generated_tokens = list(self.current_prompt)

        self.step_count = 0
        self.done = False
        self.prev_action = torch.zeros(3, device=self.device)

        observation = self._get_observation()
        return observation

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        action = action.clamp(0, 1)

        context = torch.tensor(
            self.generated_tokens[-self.model.max_context_length:],
            dtype=torch.long, device=self.device,
        )

        context_manager = torch.enable_grad() if self.training_mode else torch.no_grad()
        with context_manager:
            logits, hidden_states = self.model.forward_with_hidden(
                context.unsqueeze(0),
                gate_signals=action.unsqueeze(0) if self.training_mode else None,
            )

        effects = self.modulator(action, len(context), self.device)

        next_token_logits = logits[0, -1, :].clone()

        next_token_logits = self.modulator.apply_repetition_penalty_to_logits(
            next_token_logits,
            torch.tensor(self.generated_tokens, dtype=torch.long, device=self.device),
            effects.repetition_penalty,
        )

        next_token_logits = next_token_logits / effects.temperature

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_k = min(self.env_config.top_k, self.model.vocab_size)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
        new_token = top_k_indices[next_token_idx].item()

        reward, reward_components = self.reward_computer.compute_step_reward(
            logits[0, -1, :], self.generated_tokens, new_token, normalize=True,
        )

        # Optionally include training loss in reward
        if self.training_mode:
            target = torch.tensor([new_token], device=self.device)
            training_loss = torch.nn.functional.cross_entropy(
                logits[0, -1:, :], target,
            ).item()
            reward_components["training_loss"] = training_loss

        self.generated_tokens.append(new_token)
        self.step_count += 1
        self.prev_action = action.clone()

        self.done = self.step_count >= self.env_config.max_episode_length

        observation = self._get_observation(
            precomputed_hidden_states=hidden_states,
            precomputed_logits=logits,
        )

        info = {
            "reward_components": reward_components,
            "gate_signals": {
                "creativity": float(action[0]),
                "focus": float(action[1]),
                "stability": float(action[2]),
            },
            "effects": {
                "temperature": effects.temperature,
                "repetition_penalty": effects.repetition_penalty,
            },
            "new_token": new_token,
            "step": self.step_count,
            "sequence_length": len(self.generated_tokens),
        }

        return observation, reward, self.done, info

    def _get_observation(
        self,
        precomputed_hidden_states=None,
        precomputed_logits=None,
    ) -> torch.Tensor:
        if precomputed_hidden_states is not None and precomputed_logits is not None:
            hidden_states = precomputed_hidden_states
            logits = precomputed_logits
        else:
            context = torch.tensor(
                self.generated_tokens[-self.model.max_context_length:],
                dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                logits, hidden_states = self.model.forward_with_hidden(context.unsqueeze(0))

        observation = self.model._build_rl_observation(
            torch.tensor(self.generated_tokens, dtype=torch.long, device=self.device),
            hidden_states, logits,
            self.step_count, self.env_config.max_episode_length,
        )

        # Inject previous action into context features
        observation[-4:-1] = self.prev_action

        return observation

    def get_generated_text(self, idx_to_vocab: Dict[int, str]) -> str:
        tokens = [idx_to_vocab.get(idx, "[UNK]") for idx in self.generated_tokens]
        return " ".join(tokens)

    def render(self, idx_to_vocab: Dict[int, str] = None) -> str:
        state_str = f"Step: {self.step_count}/{self.env_config.max_episode_length}\n"
        state_str += f"Tokens generated: {len(self.generated_tokens) - len(self.current_prompt)}\n"
        state_str += f"Previous gate signals: creativity={self.prev_action[0]:.3f}, "
        state_str += f"focus={self.prev_action[1]:.3f}, stability={self.prev_action[2]:.3f}\n"

        if idx_to_vocab:
            state_str += f"Text: {self.get_generated_text(idx_to_vocab)}\n"

        return state_str


class VectorizedGatingEnvironment:
    """
    Vectorized environment for parallel episode collection.
    """

    def __init__(
        self,
        model: Any,
        modulator: GatingModulator,
        reward_computer: RewardComputer,
        prompt_tokens: List[List[int]],
        config: dict,
        num_envs: int = 4,
        device: torch.device = None,
    ):
        self.num_envs = num_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = [
            GatingEnvironment(
                model=model,
                modulator=modulator,
                reward_computer=reward_computer,
                prompt_tokens=prompt_tokens,
                config=config,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def reset(self) -> torch.Tensor:
        observations = [env.reset() for env in self.envs]
        return torch.stack(observations)

    def step(
        self, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
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
            infos,
        )

    def reset_if_done(self) -> torch.Tensor:
        observations = []
        for env in self.envs:
            if env.done:
                obs = env.reset()
            else:
                obs = env._get_observation()
            observations.append(obs)
        return torch.stack(observations)
