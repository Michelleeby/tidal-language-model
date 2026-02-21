# 0007. Coupled DynamicGate Training

**Date:** 2026-02-21
**Status:** Accepted

## Context

Experiment 3 tested the hypothesis that unfreezing the 12 DynamicGate MLPs
(~24K params) during RL training would break the action-space bottleneck
identified in Experiments 1 and 2 — where no modulation value beats neutral
gating (0.50) on composite reward because the `GatingModulator` maps a single
[0,1] signal to 4 generation parameters in lockstep, tracing a monotonic
perplexity-diversity trade-off.

The implementation used a **decoupled auxiliary loss**: a separate Adam optimizer
trained the gate MLPs to maximize logit entropy (diversity) via
`-gate_entropy_weight * mean(logit_entropy)`, while the PPO agent independently
maximized composite reward. The two optimizers shared no gradient signal.

Experiment 3 results (run `20260221-011746-commit_nogit-rl_0803bea545`):

| Policy | Mean Reward | Diversity | Perplexity |
|---|---|---|---|
| random | 0.095 | 0.552 | -0.471 |
| fixed | 0.152 | 0.723 | -0.484 |
| neutral | **0.157** | 0.718 | -0.472 |
| learned | 0.148 | **0.488** | **-0.301** |

The learned policy underperforms neutral by -0.009 — *worse* than Experiment 1's
delta of -0.001. Comparison across all diagnostic experiments:

| Experiment | Learned | Neutral | Delta |
|---|---|---|---|
| Exp 1 (Weighted + zero rep) | 0.150 | 0.151 | -0.001 |
| Exp 2 (Lagrangian 0.40) | 0.146 | 0.151 | -0.005 |
| **Exp 3 (Unfrozen gates)** | **0.148** | **0.157** | **-0.009** |

The failure mode is clear: the gate MLPs and PPO agent were fighting each other.
The auxiliary entropy loss pushed gates toward higher diversity while PPO pushed
modulation toward conservative generation (~0.32). The result was worse diversity
(0.488, below even random's 0.552) and a larger gap to neutral than before.

The root cause is **objective decoupling**: the gate MLPs optimized for logit
entropy without any connection to the reward the PPO agent was actually chasing.
There was no mechanism for the agent to learn *how* the gate adjustments affected
its reward, and no mechanism for the gates to learn what the agent needed.

## Decision

Replace the decoupled auxiliary entropy loss with **coupled training**: include
the DynamicGate MLP parameters in the PPO policy gradient computation so that
reward signal flows through the gates.

### Gradient path

Currently, `update_policy` computes `loss.backward()` on observations stored in
the rollout buffer. These observations are detached from the model's forward
pass (necessary to avoid double-backward errors). The gate MLPs therefore
receive no gradient from the PPO loss.

Coupled training changes the gradient path during `collect_rollouts`:

1. The environment's `step()` runs `model.forward_with_hidden()` with
   `gate_signals=action` and `torch.enable_grad()` (already implemented for
   `gate_training=True`).
2. The logits from this forward pass flow through the DynamicGate MLPs:
   `input → GatedTransformerBlock → attn_gate(signal) * attn_output → ... →
   ffn_gate(signal) * ffn_output → logits`.
3. Instead of computing a separate entropy loss on these logits, compute the
   **PPO-compatible reward signal** and backpropagate it through the gate
   parameters.

The key design question is *how* to compute a differentiable reward from
logits. The PPO policy loss itself is not differentiable through the
environment (rewards are scalars). Two options were considered:

**Option A — Differentiable surrogate reward**: Compute a differentiable
approximation of the composite reward directly from logits (e.g., cross-entropy
as perplexity proxy, logit entropy as diversity proxy) and backpropagate
through gates. This creates a direct gradient path but uses a *different*
objective than what the PPO agent optimizes, risking a subtler form of the
same decoupling problem.

**Option B — REINFORCE-style gate gradient**: Use the scalar PPO reward as a
reinforcement signal for the gate parameters. At each step, compute
`gate_loss = -reward * log_prob_of_action` where the log_prob is computed
with gradients flowing through the gate MLPs (since they affect the logits
which affect the action distribution). This aligns gate updates with the
exact same objective the PPO agent maximizes.

We choose **Option B** because it guarantees objective alignment: the gates
learn to reshape hidden states in whatever direction actually improves the
reward the agent receives, without introducing a surrogate objective.

### Implementation

**`collect_rollouts` changes** (`RLTrainer.py`):

During each rollout step when `gate_training=True`:
1. Run `model.forward_with_hidden()` with `enable_grad()` (already done)
2. Compute the log-probability of the sampled action under the current
   policy, but with the forward pass that flows through gate MLPs
3. After receiving the scalar reward from the environment, compute:
   `gate_loss = -reward * gate_log_prob / num_steps`
4. Call `gate_loss.backward()` immediately (per-step, as in Experiment 3)
5. After the rollout completes, call `gate_optimizer.step()`

The per-step backward pattern from Experiment 3 is retained to avoid
accumulating a computation graph across all rollout steps.

**Key difference from Experiment 3**: The loss is `−reward × log_prob`
(REINFORCE) instead of `−entropy_weight × logit_entropy` (auxiliary). The
reward is the same scalar reward that the PPO agent receives, creating
objective alignment.

**Gate optimizer** (`RLTrainer.py`):
- Retain separate Adam optimizer for gate params (same as Experiment 3)
- Remove `RL_GATE_ENTROPY_WEIGHT` config key (no longer needed)
- Add `RL_GATE_REWARD_BASELINE` (EMA of recent rewards) to reduce
  REINFORCE variance: `gate_loss = -(reward - baseline) * gate_log_prob`
- Retain `RL_GATE_LR: 1e-4` (may need tuning; gates should move slowly
  relative to the PPO agent to maintain stability)

**Config changes** (`rl_config.yaml`):
```yaml
RL_UNFREEZE_DYNAMIC_GATES: true
RL_GATE_LR: 1.0e-4
RL_GATE_TRAINING_MODE: "coupled"  # "coupled" (REINFORCE) or "auxiliary" (entropy)
RL_GATE_REWARD_BASELINE_ALPHA: 0.05  # EMA smoothing for REINFORCE baseline
```

The `RL_GATE_TRAINING_MODE` key allows switching between coupled and auxiliary
modes for comparison. The `RL_GATE_ENTROPY_WEIGHT` key is retained for
`"auxiliary"` mode backward compatibility.

**Checkpoint changes**: Same as Experiment 3 — `gate_state_dict` and
`gate_optimizer_state_dict` are saved and restored. Additionally persist the
reward baseline EMA value.

**Files affected**:

| File | Changes |
|---|---|
| `plugins/tidal/RLTrainer.py` | Replace entropy loss with REINFORCE gate loss in `collect_rollouts`, add reward baseline EMA, retain gate optimizer step/checkpoint logic |
| `plugins/tidal/GatingEnvironment.py` | No changes (gate_training + last_logits already implemented) |
| `plugins/tidal/configs/rl_config.yaml` | Replace `RL_GATE_ENTROPY_WEIGHT` with `RL_GATE_TRAINING_MODE` and `RL_GATE_REWARD_BASELINE_ALPHA` |
| `plugins/tidal/tests/test_GatingRL.py` | Update gate loss tests for REINFORCE semantics, add baseline tests |
| `plugins/tidal/train_rl.py` | No changes (unfreeze_dynamic_gates already implemented) |

### Log-probability computation

The gate_log_prob must be computed in a way that gradients flow through the
gate MLPs but NOT through the frozen LM weights. The action is sampled by
the PPO agent (which uses detached observations), but the logits that
determine the action's probability are produced by a forward pass through
the gated model. Specifically:

1. The agent samples `action` from its Beta distribution (no gate gradient)
2. The environment runs `model.forward_with_hidden(context, gate_signals=action)`
   with `enable_grad()` — gradients flow through the unfrozen gate MLPs
3. We compute `log_prob = Beta(alpha, beta).log_prob(action)` where alpha/beta
   come from the agent's actor head applied to the observation built from the
   gated forward pass
4. Since the observation includes hidden state summaries that flowed through
   the gates, `log_prob` has a gradient path to the gate parameters
5. `(-reward * log_prob).backward()` sends the reward signal through this path

This means the gates learn: "when I scale hidden states this way, the agent
tends to take actions that get higher/lower reward." The gates adapt to make
the agent's job easier.

## Consequences

### Positive
- **Objective alignment**: Gate MLPs and PPO agent optimize the exact same
  reward signal, eliminating the tug-of-war observed in Experiment 3
- **Reward-directed adaptation**: Gates learn per-layer, per-dimension scaling
  that actually improves composite reward (perplexity + diversity + coherence),
  not just diversity in isolation
- **Preserves ADR 0001**: Action space remains 1D — gates receive richer
  training signal without expanding the agent's action dimensionality
- **Minimal code change**: Replaces ~5 lines of entropy loss computation with
  ~10 lines of REINFORCE loss computation; all infrastructure (gate optimizer,
  checkpointing, env flags) already exists from Experiment 3

### Negative
- **REINFORCE variance**: Scalar reward × log_prob has high variance. The EMA
  baseline reduces this but may not eliminate it. Gate learning could be noisy,
  requiring lower learning rates or more rollout steps for stable updates
- **Slower gate convergence**: The entropy auxiliary loss provided a strong,
  low-variance gradient (logit entropy is differentiable and smooth). REINFORCE
  is inherently noisier, so gates may take longer to move away from their
  near-identity initialization
- **Risk of gate collapse**: If early rewards are dominated by perplexity
  (which favors conservative modulation), gates could learn to suppress
  diversity-helpful scaling before the PPO agent has learned to exploit
  diversity. The reward baseline mitigates this but doesn't guarantee it

### Neutral
- The `"auxiliary"` mode is retained as a config option for A/B comparison
- Gate parameter count (~24K), unfreezing logic, and checkpoint format are
  unchanged from Experiment 3
- The frozen LM weights receive no gradients in either mode — only the 12
  DynamicGate MLPs are trainable

## Alternatives Considered

### Multi-dimensional action space
Give the agent independent control over temperature, top-k, top-p, and
repetition penalty (4D action space). Rejected per ADR 0001: the original
3-gate system demonstrated that redundant action dimensions collapse to
constants, adding exploration burden without expressiveness. Temperature,
top-k, and top-p all control distribution peakedness — the agent proved it
doesn't need separate knobs for them. Returning to a multi-dimensional space
would re-introduce the dead-parameter problem ADR 0001 solved.

### Conditional gate signals (observation-dependent gates)
Pass the 64D observation vector (instead of the 1D scalar action) to the
DynamicGate MLPs so gates produce context-dependent scaling. This is
orthogonal to coupled training and could be combined with it, but adds
complexity (64→32→embed_dim MLP, 4x parameter increase per gate) without
first testing whether the simpler fix (objective alignment) resolves the
bottleneck. If coupled training with scalar gate signals still fails, this
becomes the natural next step.

### Include gate params in the PPO optimizer directly
Add gate parameters to the same Adam optimizer as the PPO agent, so
`loss.backward()` in `update_policy` trains both agent and gates. Rejected
because the buffer observations are detached (required to prevent
double-backward errors), breaking the gradient path to gates. Re-computing
forward passes during `update_policy` would be expensive (multiple epochs ×
mini-batches × full LM forward pass per observation) and architecturally
invasive.

### Differentiable surrogate reward (Option A)
Compute a differentiable composite reward from logits (cross-entropy for
perplexity, entropy for diversity, cosine similarity for coherence) and
backpropagate through gates. Rejected because it introduces a second
objective that approximates but doesn't match the PPO reward. This risks a
subtler form of the same decoupling problem — the gates optimize a proxy
while the agent optimizes the real thing. REINFORCE with the actual reward
avoids this entirely.

## References

- Supersedes (for gate training method): Experiment 3 decoupled auxiliary entropy loss
- Related: [0001 — Single Modulation Gate](0001-single-modulation-gate.md)
- Related: [0006 — PPO-Lagrangian Diversity Constraint](0006-ppo-lagrangian-diversity-constraint.md)
- Code: `plugins/tidal/RLTrainer.py` (PPOTrainer, gate optimizer, collect_rollouts)
- Code: `plugins/tidal/TransformerLM.py` (DynamicGate, GatedTransformerBlock)
- Code: `plugins/tidal/GatingEnvironment.py` (gate_training flag)
- Config: `plugins/tidal/configs/rl_config.yaml`
- Experiment: `20260221-011746-commit_nogit-rl_0803bea545` (Experiment 3)
- Diagnostic report: `Diagnostic Experiments: Action-Space Bottleneck Validation`
