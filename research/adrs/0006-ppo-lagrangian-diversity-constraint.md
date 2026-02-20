# 0006. PPO-Lagrangian Diversity Constraint

**Date:** 2026-02-20
**Status:** Accepted

## Context

ADR 0005's `DiversityHomeostasis` controller improved mean diversity from 0.438
to 0.470 during RL gating training — still far below the 0.55 target. The root
cause is structural: **weighted rewards allow the agent to trade diversity for
perplexity gains**, regardless of how aggressively the diversity weight is
boosted. The agent maximizes the total weighted sum, and the perplexity +
coherence components (combined weight 0.50) provide a steeper gradient than the
Gaussian-shaped diversity + sampling rewards (combined weight 0.30).

| Controller | Mean Diversity | Target |
|---|---|---|
| No homeostasis | 0.438 | 0.55 |
| DiversityHomeostasis (ADR 0005) | 0.470 | 0.55 |
| Fixed gating (0.5) | 0.725 | — |
| Random gating | 0.551 | — |

The homeostatic approach adjusts the *weight* of diversity in the reward, but
the agent can always find a trade-off point where reducing diversity by epsilon
gains more from perplexity than it loses from the boosted diversity weight.
The problem requires a fundamentally different formulation: **constrained
optimization** where diversity is not a reward component to be traded off, but
a hard constraint that must be satisfied.

## Decision

Replace weighted diversity reward with **PPO-Lagrangian** constrained
optimization:

- **Primary reward** (maximize): perplexity + coherence + repetition (diversity
  and sampling weights zeroed, remaining weights renormalized to sum to 1.0)
- **Hard constraint** (enforce): `diversity_reward >= 0.55` (configurable)
- **Learned Lagrange multiplier**: automatically scales penalty for constraint
  violations via dual gradient ascent

### Activation

Opt-in via `RL_CONSTRAINT_MODE: "lagrangian"` in config. Default `"weighted"`
preserves all existing behavior. When Lagrangian mode is active, it supersedes
`DiversityHomeostasis` (forced to `None`).

### LagrangeMultiplier class

New class in `RLTrainer.py` (after `DiversityHomeostasis`):

- **Parameterization**: Raw parameter transformed via softplus for
  non-negativity. Adam optimizer with `weight_decay=0.01` so the multiplier
  naturally decays toward zero when the constraint is satisfied.
- **Cost function**: `cost = max(0, threshold - diversity_reward)`. Positive
  cost means diversity is below threshold (constraint violated).
- **Dual update**: Gradient ascent on the Lagrangian dual, executed once per
  iteration (after rollouts, not per mini-batch) for stability on a slow
  timescale.
- **State persistence**: `state_dict()`/`load_state_dict()` for checkpoint
  round-trips.

### Cost critic head

New `cost_critic_head` on `GatingPolicyAgent` (same architecture as reward
critic: `Linear(64,32) -> Tanh -> Linear(32,1)`). Created only when
`RL_CONSTRAINT_MODE == "lagrangian"`. Parameters are automatically included in
the agent's optimizer.

**Separate methods, not conditional returns**: `forward()` and
`evaluate_actions()` signatures are unchanged. New methods
`forward_with_cost()` and `evaluate_actions_with_cost()` added for Lagrangian
mode, avoiding fragile variable-length tuple unpacking.

### Reward split

In Lagrangian mode, `diversity_weight` and `sampling_weight` are set to 0 on
the reward computer. Remaining weights are renormalized:

| Component | Weighted Mode | Lagrangian Mode |
|---|---|---|
| Perplexity | 0.35 | 0.50 (0.35/0.70) |
| Repetition | 0.20 | 0.29 (0.20/0.70) |
| Coherence | 0.15 | 0.21 (0.15/0.70) |
| Diversity | 0.15 | 0 (via constraint) |
| Sampling | 0.15 | 0 |

### Combined advantage formula

During policy updates, reward and cost advantages are combined as:

```
A_combined = (A_reward - lambda * A_cost) / (1 + lambda)
```

This scales the cost penalty by the current multiplier value and normalizes by
`(1 + lambda)` to prevent the effective learning rate from exploding as lambda
grows.

### Implementation

| File | Change |
|---|---|
| `plugins/tidal/RLTrainer.py` | `LagrangeMultiplier` class, `RolloutBuffer` extended with `store_costs`, `PPOTrainer` modified (`__init__`, `collect_rollouts`, new `compute_cost_advantages`, `update_policy`, `train`, `save/load_checkpoint`) |
| `plugins/tidal/GatingPolicyAgent.py` | `cost_critic_head`, `forward_with_cost()`, `evaluate_actions_with_cost()`, `get_cost_value()` |
| `plugins/tidal/configs/rl_config.yaml` | New Lagrangian config section (commented out by default) |
| `plugins/tidal/tests/test_GatingRL.py` | 5 new test classes (28 tests) |
| `plugins/tidal/RewardComputer.py` | No changes |
| `plugins/tidal/GatingEnvironment.py` | No changes |

## Consequences

### Positive
- Diversity becomes a **hard floor** rather than a tradeable reward component:
  the agent cannot sacrifice diversity for perplexity gains as long as the
  multiplier correctly scales the constraint penalty
- The multiplier is **self-tuning**: it increases when the constraint is
  violated and decays when satisfied, requiring no manual weight scheduling
- **Fully backward compatible**: existing configs without `RL_CONSTRAINT_MODE`
  behave identically (weighted mode with optional DiversityHomeostasis)
- Cost critic shares the feature extractor, adding minimal parameters (~2K)
  and no additional forward passes through the language model

### Negative
- Lagrangian optimization adds training complexity: a second critic head, a
  second GAE pass for cost advantages, and a dual variable update loop.
  Debugging reward vs. constraint dynamics requires monitoring both the
  primary reward and the multiplier trajectory
- The dual variable introduces a second timescale. If the multiplier learning
  rate is too high, it can oscillate; too low, the constraint is enforced too
  slowly. The `weight_decay=0.01` on the Adam optimizer mitigates this but
  adds a hyperparameter
- The cost function `max(0, threshold - diversity)` creates a discontinuous
  gradient at exactly `diversity = threshold`. In practice this is smoothed by
  averaging over rollout steps, but it could cause instability near the
  boundary

### Neutral
- ADR 0005's `DiversityHomeostasis` class remains in the codebase for
  `"weighted"` mode users. The Lagrangian mode simply forces it to `None`
- Three new history keys (`lagrange_multiplier`, `mean_cost`,
  `cost_value_loss`) are logged to TensorBoard and training metrics JSON
- Existing TransformerLM checkpoints are unaffected; only RL training
  configuration changes

## Alternatives Considered

### Increase DiversityHomeostasis aggressiveness
Raise `release_rate` from 0.03 to 0.10+ and `weight_max` from 0.35 to 0.50+.
Rejected because the fundamental problem is structural: boosting the weight in
a sum-of-weighted-rewards formulation just shifts the trade-off point. The
agent can always find a local optimum that sacrifices diversity for quality
as long as both are in the same weighted sum. Empirically, ADR 0005 already
demonstrated this — even with adaptive boosting, diversity only reached 0.470.

### Reward clipping / minimum diversity reward floor
Set `diversity_reward = max(floor, diversity_reward)` to create a guaranteed
minimum contribution. Rejected because this clips the gradient when diversity
is below the floor, removing the signal the agent needs to *improve* diversity.
The Lagrangian approach preserves the gradient through the cost critic.

### Constrained Policy Optimization (CPO)
CPO uses a trust region approach for constrained MDPs with theoretical
guarantees. Rejected as over-complex for a single constraint. PPO-Lagrangian
is simpler to implement (adds ~100 lines vs CPO's second-order optimization),
integrates naturally with the existing PPO infrastructure, and is well-tested
in safety-constrained RL literature.

### Multi-objective optimization (Pareto front)
Treat quality and diversity as separate objectives and find the Pareto front.
Rejected because the user wants a single policy, not a family of trade-offs.
The Lagrangian approach effectively picks a point on the Pareto front by
specifying the diversity threshold.

## References

- Supersedes (for diversity enforcement): [0005 — Diversity Homeostasis](0005-diversity-homeostasis.md)
- Related: [0003 — Recalibrate Entropy Homeostasis & Reward Weights](0003-recalibrate-entropy-homeostasis-and-reward-weights.md)
- Related: [0001 — Single Modulation Gate](0001-single-modulation-gate.md)
- Code: `plugins/tidal/RLTrainer.py` (LagrangeMultiplier, PPOTrainer)
- Code: `plugins/tidal/GatingPolicyAgent.py` (cost_critic_head)
- Config: `plugins/tidal/configs/rl_config.yaml`
- Tests: `plugins/tidal/tests/test_GatingRL.py`
- Literature: Stooke, Edwards, Ray (2020) — "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
