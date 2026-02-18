"""
GatingModulator.py

Applies gate signal effects to generation parameters in the TransformerLM.

Single modulation gate on a conservative-to-exploratory axis:
  modulation=0.0 (conservative): low temperature, narrow sampling, mild penalty
  modulation=1.0 (exploratory):  high temperature, wide sampling, strong penalty

All parameters move in the same direction as modulation increases.

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
    Modulates generation parameters based on a single modulation gate signal.

    The modulation signal controls all parameters on a conservative-to-exploratory
    axis. All parameters increase with modulation:

    - Temperature: base_temp * (min + modulation * (max - min))
    - Top-K: min + modulation * (max - min)
    - Top-P: min + modulation * (max - min)
    - Repetition penalty: base_penalty * (min + modulation * (max - min))
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.base_temperature = config.get("RL_BASE_TEMPERATURE", 1.0)
        self.base_repetition_penalty = config.get("RL_BASE_REPETITION_PENALTY", 1.2)

        self.temp_min = config.get("RL_MODULATION_TEMP_MIN",
                                   config.get("RL_CREATIVITY_TEMP_MIN", 0.3))
        self.temp_max = config.get("RL_MODULATION_TEMP_MAX",
                                   config.get("RL_CREATIVITY_TEMP_MAX", 2.0))

        self.penalty_min = config.get("RL_MODULATION_PENALTY_MIN",
                                      config.get("RL_STABILITY_PENALTY_MIN", 1.0))
        self.penalty_max = config.get("RL_MODULATION_PENALTY_MAX",
                                      config.get("RL_STABILITY_PENALTY_MAX", 3.5))

        self.top_k_min = config.get("RL_MODULATION_TOP_K_MIN",
                                    config.get("RL_FOCUS_TOP_K_MIN", 5))
        self.top_k_max = config.get("RL_MODULATION_TOP_K_MAX",
                                    config.get("RL_FOCUS_TOP_K_MAX", 100))

        self.top_p_min = config.get("RL_MODULATION_TOP_P_MIN",
                                    config.get("RL_CREATIVITY_TOP_P_MIN", 0.7))
        self.top_p_max = config.get("RL_MODULATION_TOP_P_MAX",
                                    config.get("RL_CREATIVITY_TOP_P_MAX", 1.0))

        self.current_gate_activations = {
            "modulation": 0.5,
        }

    def compute_temperature(self, modulation: float) -> float:
        temp_multiplier = self.temp_min + modulation * (self.temp_max - self.temp_min)
        return self.base_temperature * temp_multiplier

    def compute_repetition_penalty(self, modulation: float) -> float:
        penalty_multiplier = self.penalty_min + modulation * (self.penalty_max - self.penalty_min)
        return self.base_repetition_penalty * penalty_multiplier

    def compute_top_k(self, modulation: float) -> int:
        """Map modulation signal to top-k. All params increase with modulation."""
        top_k = self.top_k_min + modulation * (self.top_k_max - self.top_k_min)
        return int(round(top_k))

    def compute_top_p(self, modulation: float) -> float:
        """Map modulation signal to top-p. High modulation = high top-p (wider nucleus)."""
        return self.top_p_min + modulation * (self.top_p_max - self.top_p_min)

    def forward(
        self, gate_signals: torch.Tensor, seq_len: int, device: torch.device,
    ) -> GatingEffects:
        if isinstance(gate_signals, torch.Tensor):
            gate_signals = gate_signals.detach().cpu()
            modulation = float(gate_signals[0].clamp(0, 1))
        else:
            modulation = max(0, min(1, gate_signals[0]))

        self.current_gate_activations = {
            "modulation": modulation,
        }

        temperature = self.compute_temperature(modulation)
        repetition_penalty = self.compute_repetition_penalty(modulation)
        top_k = self.compute_top_k(modulation)
        top_p = self.compute_top_p(modulation)

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
        return torch.rand(1, device=self.device)


class FixedGatingPolicy:
    """Baseline policy that outputs a fixed modulation level."""

    def __init__(
        self,
        modulation: float = 0.5,
        device: torch.device = None,
        # Legacy kwargs for backward compatibility during transition
        creativity: float = None,
        focus: float = None,
        stability: float = None,
    ):
        self.device = device or torch.device("cpu")
        self.gate_signals = torch.tensor(
            [modulation], device=self.device,
        )

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        return self.gate_signals.clone()


class NeutralGatingPolicy:
    """Baseline policy for neutral gating (no modulation effect)."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.gate_signals = torch.tensor([0.5], device=self.device)

    def get_action(self, observation: torch.Tensor = None) -> torch.Tensor:
        return self.gate_signals.clone()
