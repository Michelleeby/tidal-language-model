# 0005. Diversity Homeostasis for RL Gating Agent

**Date:** 2026-02-19
**Status:** Superseded by [0006](0006-ppo-lagrangian-diversity-constraint.md) for diversity enforcement. Retained for `"weighted"` constraint mode.

## Context

After PR #8 ([ADR 0003](0003-recalibrate-entropy-homeostasis-and-reward-weights.md))
reduced `RL_REWARD_DIVERSITY_WEIGHT` from 0.25 to 0.15, ablation shows the
learned RL agent suffers severe diversity collapse (0.438 vs 0.725 for fixed
gating) while achieving nearly equal total reward. The agent exploits the
reduced diversity pressure to trade diversity for perplexity gains.

| Strategy | Mean Reward | Diversity |
|---|---|---|
| Fixed (0.5) | 0.197 | **0.725** |
| Random | 0.146 | 0.551 |
| Learned | **0.198** | 0.438 |

Simply restoring the old weight (0.25) would re-introduce the entropy > quality
imbalance that ADR 0003 fixed. The fix must be **adaptive** — boost diversity
pressure only when diversity is actually collapsing.

## Decision

Add a `DiversityHomeostasis` controller that mirrors the existing
`EntropyHomeostasis` pattern (release + decay + clamp) but operates on the
diversity reward weight rather than the entropy coefficient.

### Mechanism

1. **Release**: if `mean_diversity < target` → `weight += release_rate * (target - mean_diversity)`
2. **Decay**: `weight = decay_rate * weight + (1 - decay_rate) * baseline`
3. **Clamp**: `weight = clamp(weight, min, max)`

### Config parameters

| Parameter | Value | Rationale |
|---|---|---|
| `RL_DIVERSITY_HOMEOSTASIS_TARGET` | 0.55 | Between random baseline (0.55) and fixed-0.5 (0.72) |
| `RL_DIVERSITY_HOMEOSTASIS_RELEASE_RATE` | 0.03 | Gentler than entropy (0.05) — reward weights affect gradient magnitude more directly |
| `RL_DIVERSITY_HOMEOSTASIS_DECAY_RATE` | 0.95 | Matches entropy homeostasis |
| `RL_DIVERSITY_WEIGHT_MIN` | 0.15 | Current baseline (ADR 0003 value) |
| `RL_DIVERSITY_WEIGHT_MAX` | 0.35 | Up to old weight + headroom, not exceeding perplexity weight |

### Integration point

The controller operates in `PPOTrainer.train()` **after `collect_rollouts()`**,
reading `rollout_stats["mean_reward_diversity"]` and mutating
`env.reward_computer.diversity_weight` for the next rollout. This is the correct
loop point because the diversity reward is computed during rollout collection.

### Activation

Activated by the **presence** of `RL_DIVERSITY_HOMEOSTASIS_TARGET` in config.
When absent, `diversity_homeostasis` is `None` (backward compatible).

### Implementation details

**`plugins/tidal/RLTrainer.py`:**
- New class `DiversityHomeostasis` (after `EntropyHomeostasis`)
- `PPOTrainer.__init__`: create controller when config key present
- `PPOTrainer.train()`: step controller after rollout, update reward computer weight
- `PPOTrainer.save_checkpoint`: persist `diversity_homeostasis_weight`
- `PPOTrainer.load_checkpoint`: restore weight and sync to reward computer

**`plugins/tidal/configs/rl_config.yaml`:**
- New config block after entropy homeostasis section

**`plugins/tidal/tests/test_GatingRL.py`:**
- `TestDiversityHomeostasis` (9 tests) mirroring `TestEntropyHomeostasis`
- `TestPPOTrainerDiversityHomeostasis` (4 tests) mirroring `TestPPOTrainerHomeostaticSchedule`

**No changes to `RewardComputer.py`** — `diversity_weight` is already a public
attribute used directly in `compute_step_reward`.

## Consequences

### Positive
- Diversity collapse is corrected adaptively: the weight boosts only when the
  agent is actually collapsing, preserving the ADR 0003 quality-first weighting
  during healthy training
- The mechanism is familiar — same release/decay/clamp pattern as entropy
  homeostasis, easy to understand and debug
- Backward compatible — existing configs without the target key behave identically

### Negative
- Two independent homeostatic controllers operating on different reward
  components add training dynamics complexity. Interaction effects are possible
  but mitigated by the gentler release rate (0.03 vs 0.05)
- The 0.55 target is calibrated from one ablation run. Different model sizes or
  datasets may need recalibration

### Neutral
- Existing TransformerLM checkpoints are unaffected — only RL training config
  changes
- The diversity weight is logged in training history and TensorBoard for monitoring

## Alternatives Considered

### Restore old diversity weight (0.25)
Rejected — would undo ADR 0003's quality-first rebalancing. The static 0.25
weight causes the agent to satisfice on entropy metrics rather than optimize
generation quality even when diversity is healthy.

### Generalize EntropyHomeostasis into a generic HomeostasisController
Rejected — Rule of Three. The two controllers monitor different quantities
(policy entropy vs reward diversity), operate at different loop points (after
`update_policy` vs after `collect_rollouts`), and have independent config
namespaces. Premature generalization would complicate both without clear benefit.
If a third homeostatic controller is needed, generalize then.

### Adaptive reward weights based on training phase
Rejected (same as ADR 0003) — premature complexity. The homeostatic approach
is reactive and self-correcting, requiring no phase schedule tuning.

### Minimum diversity constraint (hard floor)
Rejected — a hard floor would clip the reward signal, creating a discontinuous
gradient landscape. The homeostatic approach smoothly adjusts pressure, giving
the agent a continuous gradient to improve diversity.

## References

- Related ADR: [0003 — Recalibrate Entropy Homeostasis & Reward Weights](0003-recalibrate-entropy-homeostasis-and-reward-weights.md)
- Related ADR: [0001 — Single Modulation Gate](0001-single-modulation-gate.md)
- Code: `plugins/tidal/RLTrainer.py` (EntropyHomeostasis, DiversityHomeostasis)
- Code: `plugins/tidal/RewardComputer.py` (diversity_weight attribute)
- Config: `plugins/tidal/configs/rl_config.yaml`
- Tests: `plugins/tidal/tests/test_GatingRL.py`
