"""
GatingModulator.py

Applies gate signal effects to generation parameters in the TransformerLM.

Gate signals modulate generation parameters directly:
- creativity: Sampling temperature (more exploration)
- focus: Attention focus (sharpen attention on recent tokens)
- stability: Repetition penalty (avoid repeating tokens)

This module provides the bridge between RL-controlled gate signals and
the generation behavior of the language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GatingEffects:
    """Container for gate-modulated generation parameters."""
    temperature: float
    repetition_penalty: float
    attention_bias: Optional[torch.Tensor]


class GatingModulator(nn.Module):
    """
    Modulates generation parameters based on gate signal levels.

    Gate signal effects:
    - creativity: Controls sampling temperature
      - Effect: base_temp * (0.5 + creativity) -> [0.5x, 1.5x] of base
      - High creativity = more exploration

    - focus: Controls attention focus on recent tokens
      - Effect: Adds position-based bias to attention scores
      - High focus = sharper focus on recent context

    - stability: Controls repetition penalty
      - Effect: base_penalty * (1 + stability * 1.5) -> [1x, 2.5x] of base
      - High stability = stronger penalty for repeating tokens
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.base_temperature = config.get("RL_BASE_TEMPERATURE", 1.0)
        self.base_repetition_penalty = config.get("RL_BASE_REPETITION_PENALTY", 1.2)

        self.creativity_temp_min = config.get("RL_CREATIVITY_TEMP_MIN",
                                              config.get("RL_CATALYST_TEMP_MIN", 0.5))
        self.creativity_temp_max = config.get("RL_CREATIVITY_TEMP_MAX",
                                              config.get("RL_CATALYST_TEMP_MAX", 1.5))

        self.stability_penalty_min = config.get("RL_STABILITY_PENALTY_MIN",
                                                config.get("RL_INHIBITOR_PENALTY_MIN", 1.0))
        self.stability_penalty_max = config.get("RL_STABILITY_PENALTY_MAX",
                                                config.get("RL_INHIBITOR_PENALTY_MAX", 2.5))

        self.focus_attention_strength = config.get("RL_FOCUS_ATTENTION_STRENGTH",
                                                   config.get("RL_STRESS_ATTENTION_STRENGTH", 2.0))

        self.current_gate_activations = {
            "creativity": 0.5,
            "focus": 0.5,
            "stability": 0.5,
        }

    def compute_temperature(self, creativity: float) -> float:
        temp_multiplier = self.creativity_temp_min + creativity * (self.creativity_temp_max - self.creativity_temp_min)
        return self.base_temperature * temp_multiplier

    def compute_repetition_penalty(self, stability: float) -> float:
        penalty_multiplier = self.stability_penalty_min + stability * (self.stability_penalty_max - self.stability_penalty_min)
        return self.base_repetition_penalty * penalty_multiplier / self.base_repetition_penalty

    def compute_attention_bias(
        self, focus: float, seq_len: int, device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        if seq_len > 1:
            positions = positions / (seq_len - 1)
        bias = self.focus_attention_strength * focus * positions
        return bias

    def forward(
        self, gate_signals: torch.Tensor, seq_len: int, device: torch.device,
    ) -> GatingEffects:
        if isinstance(gate_signals, torch.Tensor):
            gate_signals = gate_signals.detach().cpu()
            creativity = float(gate_signals[0].clamp(0, 1))
            focus = float(gate_signals[1].clamp(0, 1))
            stability = float(gate_signals[2].clamp(0, 1))
        else:
            creativity, focus, stability = gate_signals
            creativity = max(0, min(1, creativity))
            focus = max(0, min(1, focus))
            stability = max(0, min(1, stability))

        self.current_gate_activations = {
            "creativity": creativity,
            "focus": focus,
            "stability": stability,
        }

        temperature = self.compute_temperature(creativity)
        repetition_penalty = self.compute_repetition_penalty(stability)
        attention_bias = self.compute_attention_bias(focus, seq_len, device)

        return GatingEffects(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention_bias=attention_bias,
        )

    def get_current_gate_activations(self) -> Dict[str, float]:
        return self.current_gate_activations.copy()

    def apply_attention_bias(
        self, attention_scores: torch.Tensor, attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        return attention_scores + attention_bias

    def apply_repetition_penalty_to_logits(
        self, logits: torch.Tensor, generated_tokens: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if len(generated_tokens) == 0 or repetition_penalty == 1.0:
            return logits

        unique_tokens = torch.unique(generated_tokens)
        modified_logits = logits.clone()

        if logits.dim() == 1:
            for token_id in unique_tokens:
                if modified_logits[token_id] > 0:
                    modified_logits[token_id] /= repetition_penalty
                else:
                    modified_logits[token_id] *= repetition_penalty
        else:
            for token_id in unique_tokens:
                mask = modified_logits[:, token_id] > 0
                modified_logits[mask, token_id] /= repetition_penalty
                modified_logits[~mask, token_id] *= repetition_penalty

        return modified_logits


class RandomGatingPolicy:
    """Baseline policy that outputs random gate signal levels."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        return torch.rand(3, device=self.device)


class FixedGatingPolicy:
    """Baseline policy that outputs fixed gate signal levels."""

    def __init__(
        self,
        creativity: float = 0.5,
        focus: float = 0.5,
        stability: float = 0.5,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.gate_signals = torch.tensor(
            [creativity, focus, stability], device=self.device,
        )

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        return self.gate_signals.clone()


class NeutralGatingPolicy:
    """Baseline policy for neutral gating (no modulation effect)."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.gate_signals = torch.tensor([0.5, 0.0, 0.0], device=self.device)

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        return self.gate_signals.clone()
