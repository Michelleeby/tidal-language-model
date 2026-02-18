# 0001. Single Modulation Gate

**Date:** 2026-02-18
**Status:** Accepted

## Context

Gate Trajectory Analysis of experiment `20260217-143303` revealed that the
original 3-gate design (creativity, focus, stability) suffered from two
structural problems:

1. **Gate signal collapse.** The RL agent converged to near-constant values for
   creativity (~0.52) and focus (~0.48) across all prompts and generation steps.
   Only the stability gate showed meaningful variance (0.35-0.75). Cross-prompt
   standard deviations for creativity and focus were effectively zero, meaning
   these gates were "dead" — the agent learned to ignore them.

2. **Redundant parameter mapping.** The `GatingModulator` mapped three
   independent gate signals to four generation parameters, but the mappings
   overlapped heavily:
   - Creativity controlled temperature (0.3-2.0x) and top-p (0.7-1.0)
   - Focus controlled top-k (100-5, inverse)
   - Stability controlled repetition penalty (1.0-3.5x)

   Temperature and top-k/top-p all modulate the same underlying quantity:
   how peaked or flat the sampling distribution is. There was no
   reason for two separate knobs (creativity, focus) to control what is
   effectively one axis of variation. Meanwhile, stability's repetition
   penalty operated on a 3.5x multiplier range, dwarfing the effect of
   the other gates at the logit level.

The `DynamicGate` modules inside each `GatedTransformerBlock` (which scale
attention and FFN outputs per-dimension) also received all 3 gate signals
but showed near-zero learned weight norms for the creativity and focus
dimensions — confirming the network itself found no use for them.

These findings are scale-independent: the root causes are architectural
(parameter mapping ranges and redundancy), not a consequence of model size
or dataset. Fixing them at TinyStories scale before scaling up is the
correct sequencing.

## Decision

Replace the 3-gate system (creativity, focus, stability) with a single
**modulation** gate on a conservative-to-exploratory axis.

The modulation signal is a scalar in [0, 1] where:
- **0.0 = conservative**: low temperature, narrow top-k, tight top-p, mild
  repetition penalty
- **1.0 = exploratory**: high temperature, wide top-k, full top-p, strong
  repetition penalty

All four generation parameters now move monotonically in the same direction
as modulation increases, eliminating the redundancy between creativity/focus
and removing the arbitrary separation of correlated controls.

### Implementation details

**Model layer** (`TransformerLM.py`):
- `GatedTransformerBlock.GATE_DIM` changed from 3 to 1.
- `DynamicGate` input dimension is now 1. Each block's `attn_gate` and
  `ffn_gate` MLPs map a 1D signal to per-dimension scaling factors
  (1 -> 32 -> embed_dim -> sigmoid).
- `gate_signals` tensor shape changed from `(batch, 3)` to `(batch, 1)`.
- Observation builder `_build_rl_observation` context features: 1 previous
  modulation value + 2 padding zeros (was 3 gate values).

**RL agent** (`GatingPolicyAgent.py`):
- `RL_ACTION_DIM` changed from 3 to 1.
- Actor head outputs 2 parameters (alpha, beta for a single Beta distribution)
  instead of 6.
- `get_action()` returns a 1D tensor instead of 3D.

**Modulator** (`GatingModulator.py`):
- Single `modulation` signal drives all four parameters:
  temperature, top-k, top-p, repetition penalty.
- Config keys renamed from `RL_CREATIVITY_*` / `RL_FOCUS_*` / `RL_STABILITY_*`
  to `RL_MODULATION_*` (old keys retained as fallbacks for migration).
- `FixedGatingPolicy`, `RandomGatingPolicy`, `NeutralGatingPolicy` all
  operate on 1D tensors.

**Environment** (`GatingEnvironment.py`):
- `action_dim` = 1 throughout.
- `prev_action` is a 1-element tensor.
- Info dict reports `gate_signals.modulation` instead of three separate keys.

**Reward** (`RewardComputer.py`):
- `compute_sampling_reward()` replaces the old focus reward, measuring entropy
  of the post-filtered distribution. This creates a direct gradient path:
  modulation -> top-k/top-p -> filtered distribution -> sampling entropy.
- Weight key renamed from `RL_REWARD_FOCUS_WEIGHT` to `RL_REWARD_SAMPLING_WEIGHT`.

**Checkpoint migration** (`migrate_checkpoint.py`):
- Converts old 3-gate `DynamicGate` weights (3x32 input layer) to 1-gate
  (1x32) by extracting the stability column (index 2), which was the only
  gate with learned non-zero weights.
- Converts old RL agent checkpoints (action_dim=3) to action_dim=1.

**Dashboard and MCP tools**:
- Charts (`GateTrajectoryChart`, `RLGateSignalsChart`) render a single
  modulation series instead of three.
- Generation tools accept a single `modulation` parameter instead of
  `creativity` / `focus` / `stability`.
- Block patterns and API types updated for 1-gate schema.

**Config files**:
- `rl_config.yaml`: `RL_ACTION_DIM: 1`, all modulator ranges under
  `RL_MODULATION_*` namespace.
- `base_config.yaml`: `GATE_DIM: 1`.
- `manifest.yaml`: generation config specifies single `modulation` parameter.

## Consequences

### Positive
- Eliminates two dead parameters from the RL action space, reducing
  exploration burden from [0,1]^3 to [0,1]^1.
- All generation parameters are now controlled by a single semantically
  meaningful axis (conservative vs. exploratory).
- Simpler reward signal: the agent no longer needs to learn that
  creativity and focus are interchangeable.
- Faster RL convergence expected (1D Beta distribution vs. 3D).
- Smaller `DynamicGate` MLPs (1x32 input vs. 3x32) — marginal but real
  parameter savings across 12 gate modules (2 per block x 6 blocks).
- Dashboard and trajectory analysis are more interpretable with a single
  time series.

### Negative
- Loses the ability to independently control temperature vs. top-k if a
  future model/dataset combination benefits from decoupled axes. This can
  be re-introduced later if empirical evidence warrants it.
- Existing RL checkpoints require migration via `migrate_checkpoint.py`.
  Old 3-gate model checkpoints also require migration (DynamicGate weight
  shape change).
- The stability-column extraction heuristic for migration assumes the
  stability gate was the only informative one — true for experiment
  `20260217-143303` but may not generalize to all future training runs
  (not a practical concern since no other runs exist).

### Neutral
- The `DynamicGate` mechanism itself is unchanged — it still maps a
  low-dimensional signal to per-dimension scaling via a small MLP with
  sigmoid output initialized near 1.0. Only its input dimensionality
  changed.
- Reward component weights are unchanged; only the focus component was
  renamed to "sampling" to reflect its new gradient pathway.

## Alternatives Considered

### Keep 3 gates, fix the parameter mapping ranges
Equalize the effect magnitudes so creativity and focus have comparable
impact to stability. Rejected because the fundamental issue is redundancy:
temperature, top-k, and top-p all control distribution peakedness.
Giving them separate knobs adds exploration burden without adding
expressiveness. Empirical evidence (near-zero learned weights in
DynamicGate for creativity/focus dimensions) confirms the network
agrees these are redundant.

### Reduce to 2 gates (sampling shape + repetition penalty)
Split the control into "how peaked is sampling" and "how much to penalize
repeats." Rejected because repetition penalty scales with temperature in
practice — aggressive penalty with low temperature produces degenerate
greedy outputs. A single axis that moves both together is more natural
and avoids the agent needing to learn their correlation.

### Per-layer gate signals (1 gate per block)
Give each of the 6 transformer blocks its own modulation signal (6D action
space). Rejected as premature: the current DynamicGate MLPs already learn
per-dimension scaling from a shared signal, so per-layer variation is
implicitly supported. Adding 6 independent signals would increase
exploration burden 6x without evidence that layers need different
modulation levels.

## References

- Session: `research/sessions/20260218_185115_...explain_the_results_of_the_gate_trajectory_analysis_for_expe.md`
- Session: `research/sessions/20260218_190207_...implement_the_following_plan_plan_single_modulation_gate_red.md`
- Code: `plugins/tidal/TransformerLM.py` (DynamicGate, GatedTransformerBlock)
- Code: `plugins/tidal/GatingModulator.py`
- Code: `plugins/tidal/GatingPolicyAgent.py`
- Code: `plugins/tidal/GatingEnvironment.py`
- Code: `plugins/tidal/RewardComputer.py`
- Code: `plugins/tidal/migrate_checkpoint.py`
- Config: `plugins/tidal/configs/rl_config.yaml`
- Experiment: `20260217-143303-commit_1939ffe-config_3b95bac57d`
