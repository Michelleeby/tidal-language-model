"""
HormoneRLAgent.py

Actor-Critic network for PPO-based hormone control.

The agent learns to control three hormone levels [catalyst, stress, inhibitor]
based on observations from the generation process. Uses a shared feature
extractor with separate actor and critic heads.

Architecture:
    Observation (64D) -> Shared MLP (128, 64) -> Actor Head -> Actions (3D)
                                              -> Critic Head -> Value (1D)

The actor outputs a Beta distribution over [0, 1] for each hormone,
which naturally bounds the actions to the valid range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from typing import Tuple, Dict, Optional
import numpy as np


class HormoneRLAgent(nn.Module):
    """
    Actor-Critic agent for hormone control.

    Uses a shared feature extractor followed by separate heads for:
    - Actor: Outputs parameters for Beta distributions over [0, 1]
    - Critic: Outputs state value estimate

    The Beta distribution is used because it naturally supports bounded
    continuous actions in [0, 1], which matches the hormone range.
    """

    def __init__(self, config: dict, device: torch.device = None):
        """
        Initialize the HormoneRLAgent.

        Args:
            config: Configuration dictionary with network parameters
            device: Device for tensor operations
        """
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network dimensions
        self.obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        self.action_dim = config.get("RL_ACTION_DIM", 3)
        self.hidden_dim = config.get("RL_HIDDEN_DIM", 128)

        # Shared feature extractor with LayerNorm for observation normalization
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh()
        )

        # Actor head: outputs alpha and beta for Beta distribution
        # Using softplus to ensure positive parameters
        self.actor_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim * 2)  # alpha and beta for each action
        )

        # Critic head: outputs state value
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._initialize_weights()

        self.to(self.device)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller initialization for output layers
        for layer in [self.actor_head[-1], self.critic_head[-1]]:
            nn.init.orthogonal_(layer.weight, gain=0.01)

    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            observation: Observation tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            action_dist: Beta distribution over actions
            value: State value estimate
        """
        # Handle single observation
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Shared features
        features = self.feature_extractor(observation)

        # Actor: get distribution parameters
        actor_output = self.actor_head(features)

        # Split into alpha and beta, apply softplus for positivity
        alpha = F.softplus(actor_output[:, :self.action_dim]) + 1.0  # +1 for stability
        beta = F.softplus(actor_output[:, self.action_dim:]) + 1.0

        # Create Beta distribution
        action_dist = Beta(alpha, beta)

        # Critic: get value
        value = self.critic_head(features)

        return action_dist, value

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get action from policy.

        Args:
            observation: Observation tensor
            deterministic: If True, return mean action instead of sampling

        Returns:
            Action tensor of shape (action_dim,) or (batch, action_dim)
        """
        with torch.no_grad():
            action_dist, _ = self.forward(observation)

            if deterministic:
                # Use mode of Beta distribution
                alpha = action_dist.concentration1
                beta = action_dist.concentration0
                action = (alpha - 1) / (alpha + beta - 2)
                action = action.clamp(0.01, 0.99)  # Numerical stability
            else:
                action = action_dist.sample()

        # Remove batch dimension if input was single observation
        if observation.dim() == 1:
            action = action.squeeze(0)

        return action

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: Batch of observations (batch, obs_dim)
            actions: Batch of actions (batch, action_dim)

        Returns:
            log_probs: Log probabilities of actions (batch,)
            values: Value estimates (batch,)
            entropy: Entropy of action distribution (batch,)
        """
        action_dist, values = self.forward(observations)

        # Clamp actions to avoid log(0) issues at boundaries
        actions = actions.clamp(0.001, 0.999)

        # Log probability of actions
        log_probs = action_dist.log_prob(actions).sum(dim=-1)

        # Entropy for exploration bonus
        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for observation.

        Args:
            observation: Observation tensor

        Returns:
            Value estimate
        """
        with torch.no_grad():
            _, value = self.forward(observation)
        return value.squeeze(-1)


class GaussianHormoneRLAgent(nn.Module):
    """
    Alternative agent using Gaussian (Normal) distribution with tanh squashing.

    This is an alternative to the Beta distribution approach, using:
    - Gaussian distribution for unconstrained sampling
    - Tanh squashing to bound actions to [0, 1]

    Some RL algorithms work better with Gaussian distributions.
    """

    def __init__(self, config: dict, device: torch.device = None):
        """
        Initialize the Gaussian HormoneRLAgent.

        Args:
            config: Configuration dictionary
            device: Device for tensor operations
        """
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = config.get("RL_OBSERVATION_DIM", 64)
        self.action_dim = config.get("RL_ACTION_DIM", 3)
        self.hidden_dim = config.get("RL_HIDDEN_DIM", 128)

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh()
        )

        # Actor: mean and log_std
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim)
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        # Critic
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass."""
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        features = self.feature_extractor(observation)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)

        action_dist = Normal(mean, std)
        value = self.critic_head(features)

        return action_dist, value

    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Get action with tanh squashing to [0, 1]."""
        with torch.no_grad():
            action_dist, _ = self.forward(observation)

            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.rsample()

            # Squash to [0, 1] using sigmoid instead of tanh for [0,1] range
            action = torch.sigmoid(action)

        if observation.dim() == 1:
            action = action.squeeze(0)

        return action

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_dist, values = self.forward(observations)

        # Inverse sigmoid to get unsquashed actions
        actions_unsquashed = torch.log(actions / (1 - actions + 1e-8) + 1e-8)

        log_probs = action_dist.log_prob(actions_unsquashed).sum(dim=-1)

        # Correction for squashing (Jacobian of sigmoid)
        log_probs -= torch.log(actions * (1 - actions) + 1e-8).sum(dim=-1)

        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        with torch.no_grad():
            _, value = self.forward(observation)
        return value.squeeze(-1)


def create_agent(config: dict, device: torch.device = None) -> nn.Module:
    """
    Factory function to create the appropriate agent type.

    Args:
        config: Configuration dictionary
        device: Device for tensor operations

    Returns:
        Agent instance
    """
    agent_type = config.get("RL_AGENT_TYPE", "beta")

    if agent_type == "beta":
        return HormoneRLAgent(config, device)
    elif agent_type == "gaussian":
        return GaussianHormoneRLAgent(config, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
