# 0008. Input-Dependent Gating for Adaptive Depth

**Date:** 2026-02-21
**Status:** Accepted

## Context

Seven ADRs (0001-0007) document a systematic exploration of RL-controlled gating for the Tidal TransformerLM. The core architecture — a PPO agent controlling a single modulation signal that scales attention and FFN outputs via `DynamicGate` MLPs — was tested across three diagnostic experiments:

| Experiment | Approach | Learned vs Neutral |
|---|---|---|
| Exp 1 | Weighted reward + zero repetition | -0.001 |
| Exp 2 | PPO-Lagrangian diversity constraint | -0.005 |
| Exp 3 | Coupled REINFORCE through unfrozen gates | -0.009 |

In every experiment, the learned policy underperformed neutral gating (modulation = 0.50). The gap *widened* with each iteration. The root cause is structural: a single scalar signal controlling 4 generation parameters in monotonic lockstep traces an inescapable perplexity-diversity trade-off. No RL algorithm can find a better operating point on a Pareto frontier that has no interior.

Meanwhile, the 2025-2026 ML research landscape has converged on input-dependent gating as a first-class architectural primitive:

- **NeurIPS 2025 Best Paper** (Qwen): Query-dependent sigmoid gates on attention output — `sigmoid(W_g * q)` producing a scalar per head. Demonstrated quality improvements with negligible parameter overhead.
- **Mixture-of-Depths** (Raposo et al., 2024): Per-token routing decisions that allow easy tokens to skip layers entirely, achieving the same quality at lower compute.
- **Gated Linear Attention** (Yang et al., 2024): Data-dependent gating in linear attention variants, showing that input-conditional computation allocation is broadly effective.

The Tidal architecture already has the gate infrastructure (`DynamicGate` MLPs at every attention and FFN output). The question is whether replacing the external RL controller with input-dependent gates — trained end-to-end with the LM objective — produces measurable adaptive computation behavior.

## Decision

Replace the external RL gating controller with **input-dependent sigmoid gates** that learn per-token computation allocation end-to-end during Phase 1 (LM pretraining). The external RL pipeline (Phase 2) is preserved for existing workflows via a config switch.

### New gate class: `InputDependentGate`

A single linear projection + sigmoid that maps each token's hidden state to a scalar gate value:

```python
class InputDependentGate(nn.Module):
    def __init__(self, embed_dim):
        self.proj = nn.Linear(embed_dim, 1)  # 257 params per gate
        # bias=2.0, weight=0 → sigmoid(2.0) ≈ 0.88 (near-identity at init)

    def forward(self, x):  # x: (batch, seq_len, embed_dim)
        return torch.sigmoid(self.proj(x))  # → (batch, seq_len, 1)
```

This produces a **scalar per token** — shape `(batch, seq_len, 1)` — answering "should this token use this layer?" (an adaptive depth decision). This is semantically distinct from `DynamicGate`, which produces **per-dimension scaling** — shape `(batch, 1, embed_dim)` — answering "which features should this layer emphasize?" The design follows the NeurIPS 2025 Best Paper pattern.

Parameter overhead: 257 params per gate x 12 gates = 3,084 total (0.01% of the 30.7M model).

### Gate mode switch

`GatedTransformerBlock` accepts a `gate_mode` parameter:
- `"external"` (default): Instantiates `DynamicGate(GATE_DIM, embed_dim)` — existing behavior
- `"input_dependent"`: Instantiates `InputDependentGate(embed_dim)` — new behavior

The mode is set via `GATE_MODE` in the YAML config. Default remains `"external"` so all existing workflows (training, RL, generation) are unaffected.

### Gate regularization

L1 regularization on gate activations pushes values toward 0, encouraging sparsity (layer skipping). Controlled by `GATE_REG_WEIGHT` in config (default: 0.0). When active, the Trainer computes:

```python
gate_loss = mean(all_gate_activations)
total_loss = ce_loss + gate_reg_weight * gate_loss
```

### Gate analysis

`Evaluator.analyze_gate_activations()` runs the validation set through the model with `return_gate_activations=True`, accumulates all gate values, and computes:
- Per-gate statistics (mean, std, coefficient of variation)
- Sparsity fraction: fraction of (token, layer) pairs where `min(attn_gate, ffn_gate) < 0.1`
- Mean CoV across all 12 gates

### Hypothesis evaluation

`evaluate_hypothesis.py` provides `assess_hypothesis()` which evaluates three conditions against pre-registered thresholds:

| Condition | Metric | Threshold |
|---|---|---|
| C1 (Quality) | PPL_gated / PPL_ungated | <= 1.05 |
| C2 (Sparsity) | Fraction of tokens skipping a layer | >= 0.10 |
| C3 (Adaptivity) | Mean gate CoV | > 0.20 |

H1 = C1 AND C2 AND C3. Any single failure yields H0.

### Files affected

| File | Change |
|---|---|
| `plugins/tidal/TransformerLM.py` | Added `InputDependentGate`; modified `GatedTransformerBlock` for dual gate mode; added gate activation collection to `TransformerLM.forward()` |
| `plugins/tidal/Trainer.py` | Added gate regularization loss to `_train_epoch`; reads `GATE_REG_WEIGHT` from config |
| `plugins/tidal/Evaluator.py` | Added `analyze_gate_activations()` method; `run()` calls it when `GATE_MODE: "input_dependent"` |
| `plugins/tidal/evaluate_hypothesis.py` | New: `compute_gate_metrics()`, `assess_hypothesis()`, `print_verdict()` |
| `plugins/tidal/configs/base_config.yaml` | Added `GATE_MODE: "external"`, `GATE_REG_WEIGHT: 0.0` |
| `plugins/tidal/configs/adaptive_depth_config.yaml` | New: experiment overlay config |
| `plugins/tidal/tests/test_InputDependentGate.py` | New: 26 tests (gate, block, model, checkpoint, frozen, reg, analysis, hypothesis) |

## Consequences

### Positive
- **Eliminates the RL bottleneck**: Gate behavior is learned end-to-end with the LM objective, not constrained to a 1D external action space that traces a monotonic trade-off
- **Grounded in 2025-2026 research**: Follows the NeurIPS 2025 Best Paper design pattern (query-dependent sigmoid gates) and Mixture-of-Depths paradigm
- **Minimal parameter overhead**: 3,084 additional params (0.01% of model) — the gate infrastructure already exists
- **Clean experimental design**: 4-model comparison (ungated / fixed-gate / learned-no-reg / learned-regularized) with two-stage lambda selection to avoid multiple comparisons
- **Backward compatible**: Default config stays `"external"` — RL training, generation, and all existing checkpoints work unchanged

### Negative
- **Gates may collapse to constant**: Without regularization pressure, gates could learn a near-uniform value (~0.88 from init), producing no adaptive depth behavior (H0)
- **Quality-sparsity trade-off is unknown**: Aggressive sparsity regularization may push gates toward 0 faster than the model can compensate, degrading perplexity beyond the 5% threshold
- **Cross-mode checkpoints are incompatible**: `InputDependentGate` state_dict keys (`attn_gate.proj.weight`) differ from `DynamicGate` keys (`attn_gate.net.0.weight`). Loading across modes raises `RuntimeError` — this is tested and intentional

### Neutral
- The existing `DynamicGate` class and all RL infrastructure (`RLTrainer`, `GatingPolicyAgent`, `GatingModulator`, `RewardComputer`, `GatingEnvironment`) are preserved intact
- Checkpoint format for each mode is self-consistent — same-mode save/load works correctly
- The hypothesis has a clear H0 exit: if gates don't learn sparsity or adaptivity, the result is still informative (validates that this architecture needs stronger inductive bias for adaptive depth)

## Alternatives Considered

### Continue iterating on RL gating (Experiments 4+)
Further RL experiments (multi-step look-ahead, model-based RL, learned reward functions). Rejected because three experiments spanning weighted rewards, Lagrangian constraints, and coupled REINFORCE all produced the same result: learned policy <= neutral. The evidence points to a structural limitation (1D action space on a Pareto frontier), not an algorithm problem.

### Multi-gate RL with per-layer actions
Expand the RL agent's action space to 12 dimensions (one per gate), giving it independent control over each layer's contribution. Rejected per ADR 0001: the original 3-gate system demonstrated that redundant action dimensions collapse to constants. Going from 1D to 12D would dramatically increase exploration complexity without addressing the fundamental issue that external scalar signals can't capture token-level computation needs.

### Mixture-of-Experts (MoE) routing
Replace gates with a top-k router that selects a subset of layers per token, following the Mixture-of-Depths paper more literally. Rejected because the Tidal model is small (30.7M params, 6 layers) — discrete routing with only 6 options lacks granularity. Continuous sigmoid gates allow graduated computation allocation (e.g., 0.3 scaling vs. full skip), which is more appropriate at this model scale.

### Gated Linear Attention replacement
Replace the standard multi-head attention with gated linear attention variants (e.g., GLA, HGRN2). Rejected because this would be a much larger architectural change (replacing the attention mechanism entirely) and would confound the question of whether input-dependent gating produces adaptive depth. The current approach isolates the gating question by keeping the attention mechanism identical.

## References

- Supersedes (as research direction): ADRs [0003](0003-recalibrate-entropy-homeostasis-and-reward-weights.md), [0005](0005-diversity-homeostasis.md), [0006](0006-ppo-lagrangian-diversity-constraint.md), [0007](0007-coupled-dynamicgate-training.md) (RL gating experiments)
- Builds on: [0001 — Single Modulation Gate](0001-single-modulation-gate.md) (gate infrastructure)
- Plan: `PLAN.md` (Direction A: Adaptive Depth via Input-Dependent Learned Gating)
- Code: `plugins/tidal/TransformerLM.py` (InputDependentGate, GatedTransformerBlock)
- Code: `plugins/tidal/Trainer.py` (gate regularization)
- Code: `plugins/tidal/Evaluator.py` (gate analysis)
- Code: `plugins/tidal/evaluate_hypothesis.py` (hypothesis evaluation)
- Tests: `plugins/tidal/tests/test_InputDependentGate.py` (26 tests)
- Config: `plugins/tidal/configs/adaptive_depth_config.yaml`
- NeurIPS 2025 Best Paper: Qwen gated attention (query-dependent sigmoid gate on attention output)
- Raposo et al. (2024): Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
