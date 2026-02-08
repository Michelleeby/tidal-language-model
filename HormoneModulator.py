"""
HormoneModulator.py

Applies hormone effects to generation parameters in the ConstantLanguageModel.
Since ConstantLanguageModel has no physics simulation, hormones modulate generation
parameters directly:

- catalyst: Sampling temperature (more exploration)
- stress: Attention focus (sharpen attention on recent tokens)
- inhibitor: Repetition penalty (avoid repeating tokens)

This module provides the bridge between RL-controlled hormone levels and
the generation behavior of the language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HormoneEffects:
    """Container for hormone-modulated generation parameters."""
    temperature: float
    repetition_penalty: float
    attention_bias: Optional[torch.Tensor]  # Position-based attention bias


class HormoneModulator(nn.Module):
    """
    Modulates generation parameters based on hormone levels.

    Hormone effects:
    - catalyst: Controls sampling temperature
      - Effect: base_temp * (0.5 + catalyst) -> [0.5x, 1.5x] of base
      - High catalyst = more exploration/creativity

    - stress: Controls attention focus on recent tokens
      - Effect: Adds position-based bias to attention scores
      - High stress = sharper focus on recent context

    - inhibitor: Controls repetition penalty
      - Effect: base_penalty * (1 + inhibitor * 1.5) -> [1x, 2.5x] of base
      - High inhibitor = stronger penalty for repeating tokens
    """

    def __init__(self, config: dict):
        """
        Initialize the HormoneModulator.

        Args:
            config: Configuration dictionary with modulator parameters
        """
        super().__init__()
        self.config = config

        # Base generation parameters
        self.base_temperature = config.get("RL_BASE_TEMPERATURE", 1.0)
        self.base_repetition_penalty = config.get("RL_BASE_REPETITION_PENALTY", 1.2)

        # Hormone effect ranges
        self.catalyst_temp_min = config.get("RL_CATALYST_TEMP_MIN", 0.5)
        self.catalyst_temp_max = config.get("RL_CATALYST_TEMP_MAX", 1.5)

        self.inhibitor_penalty_min = config.get("RL_INHIBITOR_PENALTY_MIN", 1.0)
        self.inhibitor_penalty_max = config.get("RL_INHIBITOR_PENALTY_MAX", 2.5)

        self.stress_attention_strength = config.get("RL_STRESS_ATTENTION_STRENGTH", 2.0)

        # Track current hormone levels for logging
        self.current_hormones = {
            "catalyst": 0.5,
            "stress": 0.5,
            "inhibitor": 0.5
        }

    def compute_temperature(self, catalyst: float) -> float:
        """
        Compute modulated temperature based on catalyst hormone.

        Args:
            catalyst: Catalyst hormone level in [0, 1]

        Returns:
            Modulated temperature value
        """
        # Linear interpolation: catalyst=0 -> min, catalyst=1 -> max
        temp_multiplier = self.catalyst_temp_min + catalyst * (self.catalyst_temp_max - self.catalyst_temp_min)
        return self.base_temperature * temp_multiplier

    def compute_repetition_penalty(self, inhibitor: float) -> float:
        """
        Compute modulated repetition penalty based on inhibitor hormone.

        Args:
            inhibitor: Inhibitor hormone level in [0, 1]

        Returns:
            Modulated repetition penalty value
        """
        # Linear interpolation: inhibitor=0 -> min, inhibitor=1 -> max
        penalty_multiplier = self.inhibitor_penalty_min + inhibitor * (self.inhibitor_penalty_max - self.inhibitor_penalty_min)
        return self.base_repetition_penalty * penalty_multiplier / self.base_repetition_penalty  # Normalize to [1, 2.5]

    def compute_attention_bias(
        self,
        stress: float,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute position-based attention bias based on stress hormone.

        Higher stress = sharper focus on recent tokens.
        Creates a bias that increases attention weights for recent positions.

        Args:
            stress: Stress hormone level in [0, 1]
            seq_len: Current sequence length
            device: Device to create tensor on

        Returns:
            Attention bias tensor of shape (seq_len,)
        """
        # Create position indices (0 = oldest, seq_len-1 = most recent)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Normalize to [0, 1]
        if seq_len > 1:
            positions = positions / (seq_len - 1)

        # Stress increases focus on recent tokens (higher positions)
        # bias = stress_strength * stress * position
        # This adds positive bias to recent tokens in attention
        bias = self.stress_attention_strength * stress * positions

        return bias

    def forward(
        self,
        hormones: torch.Tensor,
        seq_len: int,
        device: torch.device
    ) -> HormoneEffects:
        """
        Compute all hormone effects for generation.

        Args:
            hormones: Tensor of shape (3,) with [catalyst, stress, inhibitor]
            seq_len: Current sequence length for attention bias
            device: Device for tensor creation

        Returns:
            HormoneEffects dataclass with modulated parameters
        """
        # Extract individual hormone levels
        if isinstance(hormones, torch.Tensor):
            hormones = hormones.detach().cpu()
            catalyst = float(hormones[0].clamp(0, 1))
            stress = float(hormones[1].clamp(0, 1))
            inhibitor = float(hormones[2].clamp(0, 1))
        else:
            catalyst, stress, inhibitor = hormones
            catalyst = max(0, min(1, catalyst))
            stress = max(0, min(1, stress))
            inhibitor = max(0, min(1, inhibitor))

        # Update current levels for logging
        self.current_hormones = {
            "catalyst": catalyst,
            "stress": stress,
            "inhibitor": inhibitor
        }

        # Compute modulated parameters
        temperature = self.compute_temperature(catalyst)
        repetition_penalty = self.compute_repetition_penalty(inhibitor)
        attention_bias = self.compute_attention_bias(stress, seq_len, device)

        return HormoneEffects(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention_bias=attention_bias
        )

    def get_current_hormones(self) -> Dict[str, float]:
        """Return the current hormone levels for logging."""
        return self.current_hormones.copy()

    def apply_attention_bias(
        self,
        attention_scores: torch.Tensor,
        attention_bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention bias to attention scores.

        Args:
            attention_scores: Raw attention scores of shape (..., seq_len)
            attention_bias: Position bias of shape (seq_len,)

        Returns:
            Biased attention scores
        """
        return attention_scores + attention_bias

    def apply_repetition_penalty_to_logits(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        repetition_penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits for previously generated tokens.

        Args:
            logits: Output logits of shape (vocab_size,) or (batch, vocab_size)
            generated_tokens: Previously generated token IDs
            repetition_penalty: Penalty factor > 1.0

        Returns:
            Logits with repetition penalty applied
        """
        if len(generated_tokens) == 0 or repetition_penalty == 1.0:
            return logits

        # Get unique tokens to penalize
        unique_tokens = torch.unique(generated_tokens)

        # Apply penalty: divide positive logits, multiply negative logits
        modified_logits = logits.clone()

        if logits.dim() == 1:
            # Single sequence
            for token_id in unique_tokens:
                if modified_logits[token_id] > 0:
                    modified_logits[token_id] /= repetition_penalty
                else:
                    modified_logits[token_id] *= repetition_penalty
        else:
            # Batched
            for token_id in unique_tokens:
                mask = modified_logits[:, token_id] > 0
                modified_logits[mask, token_id] /= repetition_penalty
                modified_logits[~mask, token_id] *= repetition_penalty

        return modified_logits


class RandomHormonePolicy:
    """
    Baseline policy that outputs random hormone levels.
    Used for ablation studies comparing learned vs random policies.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        """Return random hormone levels in [0, 1]."""
        return torch.rand(3, device=self.device)


class FixedHormonePolicy:
    """
    Baseline policy that outputs fixed hormone levels.
    Used for ablation studies comparing learned vs fixed policies.
    """

    def __init__(
        self,
        catalyst: float = 0.5,
        stress: float = 0.5,
        inhibitor: float = 0.5,
        device: torch.device = None
    ):
        self.device = device or torch.device("cpu")
        self.hormones = torch.tensor(
            [catalyst, stress, inhibitor],
            device=self.device
        )

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        """Return fixed hormone levels."""
        return self.hormones.clone()


class NoHormonePolicy:
    """
    Baseline policy for no-hormone ablation (neutral values).
    Temperature = 1.0, repetition_penalty = base, no attention bias.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        # These values result in neutral effects:
        # catalyst=0.5 -> temp_mult=1.0, stress=0 -> no bias, inhibitor=0 -> penalty=1.0
        self.hormones = torch.tensor([0.5, 0.0, 0.0], device=self.device)

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        """Return neutral hormone levels."""
        return self.hormones.clone()
