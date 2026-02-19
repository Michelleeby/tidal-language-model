# 0003. Recalibrate Entropy Homeostasis & Reward Weights for 1D Gate

**Date:** 2026-02-19
**Status:** Accepted

## Context

After collapsing the 3-gate system to a single modulation gate
([ADR 0001](0001-single-modulation-gate.md)), the RL agent's learned policy
underperformed both fixed (0.5) and neutral gating strategies in ablation:

| Strategy | Mean Reward | Diversity |
|---|---|---|
| Fixed (0.5) | **0.197** | **0.734** |
| Neutral | **0.197** | 0.719 |
| Learned | 0.182 | 0.677 |

Two root causes were identified via `get_rl_metrics` and manual inspection:

### 1. Entropy homeostasis never activated

The homeostatic controller (`EntropyHomeostasis` in `RLTrainer.py`) boosts
`entropy_coef` when observed policy entropy drops below
`RL_POLICY_ENTROPY_TARGET`. The target was `-1.0`, calibrated for the sum
entropy of a 3D Beta distribution (roughly `-0.33` per dimension). After the
collapse to 1D, the Beta distribution's entropy only reached `-0.678` at its
lowest — well above `-1.0` — so the release mechanism never triggered.

By contrast, the previous 3-gate experiment saw entropy drop to `-1.024`,
triggering homeostasis and boosting `entropy_coef` from 0.01 to 0.044.
The 1D agent trained with a flat `entropy_coef = 0.01` for the entire run.

### 2. Entropy-based rewards dominated quality signals

The reward weights allocated 40% to entropy-related components (diversity 0.25
+ sampling 0.15) and only 40% to quality components (perplexity 0.30 +
coherence 0.10). With repetition at 20%, the agent learned to hover near
entropy targets rather than optimising generation quality. The learned policy's
lower diversity score (0.677 vs 0.734) suggests it over-indexed on the
perplexity/repetition trade-off while under-exploring the diversity axis.

## Decision

### 1. Set entropy target to `-0.35`

The old 3D target of `-1.0` was effectively `-0.33` per dimension. Setting
`-0.35` for the 1D case is slightly conservative — it triggers before full
entropy collapse, ensuring homeostasis activates within the observed 1D Beta
entropy range of -0.5 to -0.7.

### 2. Rebalance reward weights toward quality

| Component | Old Weight | New Weight | Category |
|---|---|---|---|
| Perplexity | 0.30 | **0.35** | Quality |
| Diversity | 0.25 | **0.15** | Entropy |
| Sampling | 0.15 | 0.15 | Entropy |
| Repetition | 0.20 | 0.20 | Penalty |
| Coherence | 0.10 | **0.15** | Quality |

Entropy-related: 40% down to 30%. Quality-related: 40% up to 50%.

### Implementation details

**`plugins/tidal/configs/rl_config.yaml`:**
- `RL_POLICY_ENTROPY_TARGET`: `-1.0` to `-0.35`
- Comment on line 48 updated to reference 1D Beta entropy range
- Reward weights updated to `0.35 / 0.15 / 0.15 / 0.20 / 0.15`

**`plugins/tidal/RLTrainer.py`:**
- `EntropyHomeostasis.__init__` default fallback: `-1.0` to `-0.35`

**`plugins/tidal/RewardComputer.py`:**
- Default fallbacks updated: perplexity `0.30` to `0.35`, diversity `0.25` to
  `0.15`, coherence `0.10` to `0.15`

**`plugins/tidal/tests/test_GatingRL.py`:**
- New test: `TestEntropyHomeostasis.test_triggers_for_1d_beta_entropy_range` —
  verifies homeostasis triggers at entropy values -0.5 and -0.7
- New class: `TestRewardWeightBalance` with 3 tests — verifies defaults sum to
  1.0, match the rebalanced values, and that quality > entropy weight
- 6 existing test classes updated with recalibrated entropy values and new
  reward weights

## Consequences

### Positive
- Homeostasis will activate during 1D training runs, adaptively boosting
  `entropy_coef` when the Beta distribution collapses — preventing the agent
  from locking into a single modulation value
- Quality signals (perplexity + coherence = 50%) now outweigh entropy signals
  (diversity + sampling = 30%), giving the agent a stronger gradient toward
  generating coherent, low-perplexity text
- The learned policy should close the reward gap with fixed/neutral gating,
  since it can now explore more (via homeostasis) and is rewarded more for
  quality (via rebalancing)

### Negative
- Lowering diversity weight from 0.25 to 0.15 may reduce lexical variety in
  generated text if the agent over-exploits the quality signal. Monitorable
  via the diversity component in `get_rl_metrics`
- The `-0.35` target is calibrated empirically from one training run. Different
  learning rates or concentration caps may shift the entropy range, requiring
  recalibration

### Neutral
- Existing model checkpoints (TransformerLM) are unaffected — only RL training
  config changes
- The 8 test classes using the older 4-component weight pattern
  (`0.4/0.3/0.2/0.1`) were not updated, as they test general reward behaviour
  with self-contained configs that sum to 1.0

## Alternatives Considered

### Scale target proportionally: -1.0 / 3 = -0.33
Use exact per-dimension scaling from the 3D case. Rejected because `-0.33` is
very close to the healthy entropy region for 1D Beta (around `-0.2` to `-0.3`
for broad distributions), which would cause homeostasis to trigger too
aggressively during normal exploration. `-0.35` adds a small conservative
margin.

### Keep weights, only fix homeostasis
Fix the entropy target but leave reward weights at `0.30/0.25/0.15/0.20/0.10`.
Rejected because the weight imbalance is an independent problem: even with
correct homeostasis, allocating 40% of the reward to entropy metrics biases the
agent toward entropy-satisficing rather than quality-maximising behaviour.

### Adaptive reward weights based on training phase
Start with higher entropy weights for exploration, then shift toward quality
weights for exploitation. Rejected as premature complexity — the static
rebalancing addresses the observed failure mode, and the homeostatic controller
already provides adaptive exploration pressure via `entropy_coef`. Adding a
second adaptive mechanism risks interaction effects that are hard to debug.

## References

- Related ADR: [0001 — Single Modulation Gate](0001-single-modulation-gate.md)
- Code: `plugins/tidal/RLTrainer.py` (EntropyHomeostasis)
- Code: `plugins/tidal/RewardComputer.py`
- Config: `plugins/tidal/configs/rl_config.yaml`
- Tests: `plugins/tidal/tests/test_GatingRL.py`
