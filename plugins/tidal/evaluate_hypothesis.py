"""
Hypothesis evaluation for Direction A: Adaptive Depth via Input-Dependent Learned Gating.

Provides:
- compute_gate_metrics(): Analyze gate activations for sparsity and adaptivity.
- assess_hypothesis(): Evaluate H1 conditions C1/C2/C3 against thresholds.

Decision Criteria (all three must be TRUE for H1):
  C1 (Quality):    PPL_gated / PPL_ungated <= 1.05
  C2 (Sparsity):   fraction of (token, layer) pairs where min_gate < 0.1 >= 0.10
  C3 (Adaptivity): mean CoV of per-token gate activations across all gates > 0.20
"""

import json
import os
from typing import Dict, List, Tuple

import torch

# H1 decision thresholds
C1_QUALITY_THRESHOLD = 1.05
C2_SPARSITY_THRESHOLD = 0.10
C3_ADAPTIVITY_THRESHOLD = 0.20


def compute_gate_metrics(
    gate_activations: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict:
    """
    Compute sparsity and adaptivity metrics from gate activations.

    Args:
        gate_activations: List of (attn_gate, ffn_gate) tuples per layer.
            Each tensor has shape (batch, seq_len, 1) or (tokens, 1).

    Returns:
        Dict with:
          - per_gate_stats: list of {mean, std, cov} dicts (one per gate)
          - sparsity_fraction: float (C2 metric)
          - mean_cov: float (C3 metric)
    """
    per_gate_stats = []
    all_covs = []

    # Flatten each gate's activations across batch and seq_len
    for attn_gate, ffn_gate in gate_activations:
        for gate_vals in [attn_gate, ffn_gate]:
            flat = gate_vals.flatten().float()
            mean = flat.mean().item()
            std = flat.std().item()
            cov = std / mean if mean > 1e-8 else 0.0
            per_gate_stats.append({"mean": mean, "std": std, "cov": cov})
            all_covs.append(cov)

    mean_cov = sum(all_covs) / len(all_covs) if all_covs else 0.0

    # Sparsity: fraction of (token, layer) pairs where min(attn, ffn) < 0.1
    total_pairs = 0
    sparse_pairs = 0
    for attn_gate, ffn_gate in gate_activations:
        # min across attn/ffn for each token at this layer
        min_gate = torch.min(attn_gate, ffn_gate)  # (batch, seq_len, 1)
        flat_min = min_gate.flatten()
        total_pairs += flat_min.numel()
        sparse_pairs += (flat_min < 0.1).sum().item()

    sparsity_fraction = sparse_pairs / total_pairs if total_pairs > 0 else 0.0

    return {
        "per_gate_stats": per_gate_stats,
        "sparsity_fraction": sparsity_fraction,
        "mean_cov": mean_cov,
    }


def assess_hypothesis(
    ppl_gated: float,
    ppl_ungated: float,
    sparsity_fraction: float,
    mean_cov: float,
) -> Dict:
    """
    Evaluate H1 conditions against thresholds.

    Args:
        ppl_gated: Validation perplexity of the gated model.
        ppl_ungated: Validation perplexity of the ungated baseline.
        sparsity_fraction: C2 metric from compute_gate_metrics.
        mean_cov: C3 metric from compute_gate_metrics.

    Returns:
        Dict with c1, c2, c3 booleans, h1 boolean, and numeric values.
    """
    ppl_ratio = ppl_gated / ppl_ungated if ppl_ungated > 0 else float("inf")

    c1 = ppl_ratio <= C1_QUALITY_THRESHOLD
    c2 = sparsity_fraction >= C2_SPARSITY_THRESHOLD
    c3 = mean_cov > C3_ADAPTIVITY_THRESHOLD

    return {
        "h1": c1 and c2 and c3,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "ppl_ratio": ppl_ratio,
        "sparsity_fraction": sparsity_fraction,
        "mean_cov": mean_cov,
    }


def print_verdict(result: Dict) -> None:
    """Print a human-readable verdict."""
    print("=" * 60)
    print("HYPOTHESIS EVALUATION — Direction A: Adaptive Depth")
    print("=" * 60)
    print()

    conditions = [
        ("C1 Quality", result["c1"],
         f"PPL ratio = {result['ppl_ratio']:.4f} (threshold: <= {C1_QUALITY_THRESHOLD})"),
        ("C2 Sparsity", result["c2"],
         f"Sparsity = {result['sparsity_fraction']:.4f} (threshold: >= {C2_SPARSITY_THRESHOLD})"),
        ("C3 Adaptivity", result["c3"],
         f"Mean CoV = {result['mean_cov']:.4f} (threshold: > {C3_ADAPTIVITY_THRESHOLD})"),
    ]

    for name, passed, detail in conditions:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    print()
    verdict = "H1 (Alternative)" if result["h1"] else "H0 (Null)"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)
