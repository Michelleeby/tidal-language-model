"""
GatingPolicyAgent.py

Actor-Critic network for PPO-based gate signal control.

The agent learns to control a single modulation gate signal on a
conservative-to-exploratory axis, based on observations from the
generation process. Uses a shared feature extractor with separate
actor and critic heads.

Architecture:
    Observation (64D) -> Shared MLP (128, 64) -> Actor Head -> Action (1D)
                                              -> Critic Head -> Value (1D)

The actor outputs a Beta distribution over [0, 1] for the modulation signal,
which naturally bounds the action to the valid range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from typing import Tuple, Dict, Optional
import numpy as np


class GatingPolicyAgent(nn.Module):
    """
    Actor-Critic agent for single modulation gate control.

    Uses a shared feature extractor followed by separate heads for:
    - Actor: Outputs parameters for a Beta distribution over [0, 1]
    - Critic: Outputs state value estimate

    The Beta distribution is used because it naturally supports bounded
    continuous actions in [0, 1], which matches the gate signal range.
    """

    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        self.action_dim = config.get("RL_ACTION_DIM", 1)
        self.hidden_dim = config.get("RL_HIDDEN_DIM", 128)
        self.beta_concentration_max = config.get("RL_BETA_CONCENTRATION_MAX", 15.0)

        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim * 2),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.constraint_mode = config.get("RL_CONSTRAINT_MODE", "weighted")
        if self.constraint_mode == "lagrangian":
            self.cost_critic_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        output_heads = [self.actor_head[-1], self.critic_head[-1]]
        if self.constraint_mode == "lagrangian":
            output_heads.append(self.cost_critic_head[-1])
        for layer in output_heads:
            nn.init.orthogonal_(layer.weight, gain=0.01)

    def forward(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        features = self.feature_extractor(observation)

        actor_output = self.actor_head(features)
        alpha = F.softplus(actor_output[:, :self.action_dim]) + 1.0
        beta = F.softplus(actor_output[:, self.action_dim:]) + 1.0
        alpha = alpha.clamp(max=self.beta_concentration_max)
        beta = beta.clamp(max=self.beta_concentration_max)
        action_dist = Beta(alpha, beta)

        value = self.critic_head(features)

        return action_dist, value

    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            action_dist, _ = self.forward(observation)

            if deterministic:
                alpha = action_dist.concentration1
                beta = action_dist.concentration0
                action = (alpha - 1) / (alpha + beta - 2)
                action = action.clamp(0.01, 0.99)
            else:
                action = action_dist.sample()

        if observation.dim() == 1:
            action = action.squeeze(0)

        return action

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, values = self.forward(observations)

        actions = actions.clamp(0.001, 0.999)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(observation)
        return value.squeeze(-1)

    def forward_with_cost(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution, value, and cost value.

        Reuses the shared feature extractor, calling all three heads once.
        Only available when constraint_mode == 'lagrangian'.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        features = self.feature_extractor(observation)

        actor_output = self.actor_head(features)
        alpha = F.softplus(actor_output[:, :self.action_dim]) + 1.0
        beta = F.softplus(actor_output[:, self.action_dim:]) + 1.0
        alpha = alpha.clamp(max=self.beta_concentration_max)
        beta = beta.clamp(max=self.beta_concentration_max)
        action_dist = Beta(alpha, beta)

        value = self.critic_head(features)
        cost_value = self.cost_critic_head(features)

        return action_dist, value, cost_value

    def evaluate_actions_with_cost(
        self, observations: torch.Tensor, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions returning log_probs, values, entropy, and cost values.

        Only available when constraint_mode == 'lagrangian'.
        """
        action_dist, value, cost_value = self.forward_with_cost(observations)

        actions = actions.clamp(0.001, 0.999)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, value.squeeze(-1), entropy, cost_value.squeeze(-1)

    def get_cost_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get cost value estimate for an observation."""
        with torch.no_grad():
            _, _, cost_value = self.forward_with_cost(observation)
        return cost_value.squeeze(-1)


class GaussianGatingPolicyAgent(nn.Module):
    """
    Alternative agent using Gaussian distribution with sigmoid squashing to [0, 1].
    """

    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        self.action_dim = config.get("RL_ACTION_DIM", 1)
        self.hidden_dim = config.get("RL_HIDDEN_DIM", 128)

        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, observation: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        features = self.feature_extractor(observation)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        action_dist = Normal(mean, std)
        value = self.critic_head(features)

        return action_dist, value

    def get_action(
        self, observation: torch.Tensor, deterministic: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            action_dist, _ = self.forward(observation)

            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.rsample()

            action = torch.sigmoid(action)

        if observation.dim() == 1:
            action = action.squeeze(0)

        return action

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, values = self.forward(observations)

        actions_unsquashed = torch.log(actions / (1 - actions + 1e-8) + 1e-8)
        log_probs = action_dist.log_prob(actions_unsquashed).sum(dim=-1)
        log_probs -= torch.log(actions * (1 - actions) + 1e-8).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(observation)
        return value.squeeze(-1)


def create_agent(config: dict, device: torch.device = None) -> nn.Module:
    """Factory function to create the appropriate agent type."""
    agent_type = config.get("RL_AGENT_TYPE", "beta")

    if agent_type == "beta":
        return GatingPolicyAgent(config, device)
    elif agent_type == "gaussian":
        return GaussianGatingPolicyAgent(config, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
