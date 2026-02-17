"""
GatingModulator.py

Applies gate signal effects to generation parameters in the TransformerLM.

Gate signals modulate generation parameters directly:
- creativity: Sampling temperature + nucleus (top-p) width
- focus: Top-k filtering sharpness
- stability: Repetition penalty (avoid repeating tokens)

This module provides the bridge between RL-controlled gate signals and
the generation behavior of the language model.
"""

import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass


@dataclass
class GatingEffects:
    """Container for gate-modulated generation parameters."""
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float


class GatingModulator(nn.Module):
    """
    Modulates generation parameters based on gate signal levels.

    Gate signal effects:
    - creativity: Controls sampling temperature and nucleus (top-p) width
      - Temperature: base_temp * (min + creativity * (max - min))
      - Top-p: top_p_min + creativity * (top_p_max - top_p_min)
      - High creativity = higher temperature + wider nucleus

    - focus: Controls top-k filtering (sampling sharpness)
      - Effect: top_k_max - focus * (top_k_max - top_k_min)
      - High focus = low top-k = sharper sampling

    - stability: Controls repetition penalty
      - Effect: penalty in range [min, max] based on stability level
      - High stability = stronger penalty for repeating tokens
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.base_temperature = config.get("RL_BASE_TEMPERATURE", 1.0)
        self.base_repetition_penalty = config.get("RL_BASE_REPETITION_PENALTY", 1.2)

        self.creativity_temp_min = config.get("RL_CREATIVITY_TEMP_MIN",
                                              config.get("RL_CATALYST_TEMP_MIN", 0.3))
        self.creativity_temp_max = config.get("RL_CREATIVITY_TEMP_MAX",
                                              config.get("RL_CATALYST_TEMP_MAX", 2.0))

        self.stability_penalty_min = config.get("RL_STABILITY_PENALTY_MIN",
                                                config.get("RL_INHIBITOR_PENALTY_MIN", 1.0))
        self.stability_penalty_max = config.get("RL_STABILITY_PENALTY_MAX",
                                                config.get("RL_INHIBITOR_PENALTY_MAX", 3.5))

        self.focus_top_k_min = config.get("RL_FOCUS_TOP_K_MIN", 5)
        self.focus_top_k_max = config.get("RL_FOCUS_TOP_K_MAX", 100)

        self.creativity_top_p_min = config.get("RL_CREATIVITY_TOP_P_MIN", 0.7)
        self.creativity_top_p_max = config.get("RL_CREATIVITY_TOP_P_MAX", 1.0)

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
        return self.base_repetition_penalty * penalty_multiplier

    def compute_top_k(self, focus: float) -> int:
        """Map focus signal to top-k. High focus = low top-k (sharper)."""
        top_k = self.focus_top_k_max - focus * (self.focus_top_k_max - self.focus_top_k_min)
        return int(round(top_k))

    def compute_top_p(self, creativity: float) -> float:
        """Map creativity signal to top-p. High creativity = high top-p (wider nucleus)."""
        return self.creativity_top_p_min + creativity * (self.creativity_top_p_max - self.creativity_top_p_min)

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
        top_k = self.compute_top_k(focus)
        top_p = self.compute_top_p(creativity)

        return GatingEffects(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
        )

    def get_current_gate_activations(self) -> Dict[str, float]:
        return self.current_gate_activations.copy()

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
