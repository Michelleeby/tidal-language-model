# Direction A: Adaptive Depth via Input-Dependent Learned Gating

## Hypothesis

**H1 (Alternative):** Input-dependent sigmoid gates at attention and FFN outputs in a 6-layer TransformerLM, trained end-to-end with cross-entropy loss and L1 gate sparsity regularization, will learn token-level computation allocation that preserves language modeling quality while exhibiting measurable sparsity and input dependence.

**H0 (Null):** Input-dependent gates either degrade model quality, collapse to near-constant activation (no learned sparsity), or fail to exhibit token-dependent behavior — reducing to a fixed architecture with extra parameters.

### Decision Criteria (all three must be TRUE for H1)

| Condition | Metric | TRUE threshold | FALSE threshold |
|-----------|--------|----------------|-----------------|
| **C1 — Quality** | `PPL_gated / PPL_ungated` on TinyStories validation | ≤ 1.05 | > 1.05 |
| **C2 — Sparsity** | Fraction of (token, layer) pairs where `min_gate < 0.1` | ≥ 0.10 | < 0.10 |
| **C3 — Adaptivity** | Mean coefficient of variation (σ/μ) of per-token gate activations across all 12 gates | > 0.20 | ≤ 0.20 |

**H1 = C1 ∧ C2 ∧ C3** — all three must pass. Any single failure → H0.

### Operational Definitions

- **"Input-dependent gate"**: A learned linear projection `sigmoid(W_g · x + b_g)` where `x` is the pre-norm hidden state at each sub-layer. The gate produces a **scalar per token** per sub-layer — shape `(batch, seq_len, 1)`. This is an intentional design difference from the existing `DynamicGate`, which produces **per-dimension scaling** — shape `(batch, 1, embed_dim)`. The scalar gate answers "should this token use this layer?" (a depth/skip decision), whereas per-dimension gating answers "which features should this layer emphasize?" (a feature selection decision). Adaptive depth requires the former. This follows the NeurIPS 2025 Best Paper design where `sigmoid(W_g · q)` produces a scalar per head.
- **"Sparsity"**: For each token at each layer, compute `min(attn_gate, ffn_gate)`. If this value < 0.1, that token is "effectively skipping" at least one sub-layer computation at that layer. C2 measures the fraction of (token, layer) pairs where this occurs.
- **"Adaptivity"**: For each gate (12 total: 6 layers × 2 sub-layers), compute the coefficient of variation (std/mean) of gate activations across all tokens in the validation set. C3 is the mean of these 12 CoV values. A CoV > 0.20 means gates produce meaningfully different values for different tokens (not a constant scalar).
- **"Ungated baseline"**: The same `TransformerLM` architecture trained with `gate_signals=None` (all gates return 1.0). This is the current Phase 1 training — no architectural change needed for the baseline.
- **"Fixed-gate baseline"**: The same architecture with `InputDependentGate` modules, but gate weights frozen at initialization (sigmoid(2.0) ≈ 0.88 constant for all tokens). Isolates whether input-dependent gating matters vs. simply having a learned constant scale factor.

---

## Experimental Design

### Models to train (4 runs + λ sweep)

| Model | Gate Mode | Gate Weights | Regularization | Purpose |
|-------|-----------|-------------|---------------|---------|
| **A (Ungated)** | None (`gate_signals=None`) | N/A | None | Establishes ungated perplexity floor |
| **B (Fixed-gate)** | Input-dependent | **Frozen** at init (constant ≈0.88) | None | Controls for "having gates at all" vs. input-dependent learning |
| **C (Learned, no reg)** | Input-dependent | Trained | λ = 0 | Tests whether gates learn anything without sparsity pressure |
| **D (Learned, regularized)** | Input-dependent | Trained | λ pre-registered | Tests whether sparsity pressure produces adaptive depth |

Model B is critical: it isolates whether improvements come from input-dependent gating (H1) or merely from having a constant scaling factor (H0). If C or D beats A but not B, the gates haven't learned meaningful input dependence.

### Controlled variables (identical across all models)
- Architecture: 6 layers, 256D, 8 heads, 1024 FFN, 50257 vocab, 256 context
- Data: TinyStories train split, GPT-2 BPE tokenization
- Training: 3 epochs, batch 2048 (64 × 32 accumulation), Adam, cosine LR annealing
- Random seed: **42** (fixed across all runs for reproducibility)
- Evaluation: TinyStories validation split, full-set perplexity

### λ selection (two-stage, avoids multiple comparisons)

**Problem**: Sweeping λ ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2} and picking the best on the same validation set inflates the probability of a spurious H1.

**Solution**: Two-stage protocol with data splitting.

1. **Stage 1 — λ tuning** (on tuning split):
   - Split TinyStories validation set 50/50 into `val_tune` and `val_eval`
   - Train 5 models with λ ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
   - Evaluate perplexity + gate sparsity on `val_tune` only
   - Select λ* = the λ that maximizes C2 (sparsity) while keeping `PPL_gated / PPL_ungated ≤ 1.05` on `val_tune`
   - If no λ satisfies the quality constraint on `val_tune`, report the Pareto frontier and conclude H0

2. **Stage 2 — Hypothesis evaluation** (on held-out eval split):
   - Pre-register λ* (the single selected value)
   - Train Model D with λ* from scratch (fresh random seed 42)
   - Evaluate C1, C2, C3 **exclusively on `val_eval`** (never seen during λ selection)
   - This is the single hypothesis test. No further tuning.

### Evaluation protocol
1. Train Models A, B, C, D to completion (3 epochs each)
2. Compute validation perplexity on `val_eval` for each model
3. Run gate activation analysis on `val_eval` (Models B, C, D):
   - Record all 12 gate activations for every token
   - Compute per-gate mean, std, CoV
   - Compute sparsity fraction (min_gate < 0.1)
   - Compute per-layer gate histograms
4. Evaluate C1, C2, C3 on Model D against thresholds using `val_eval`
5. Report H1 or H0
6. Report Model B vs. C comparison as supplementary analysis (does learning help vs. fixed?)

---

## Implementation Plan

### Step 1: New gate class — `InputDependentGate`

**File**: `plugins/tidal/TransformerLM.py`

Create `InputDependentGate` alongside existing `DynamicGate` (do not remove `DynamicGate`).

```python
class InputDependentGate(nn.Module):
    """
    Input-dependent gate that produces a per-token scalar from the hidden state.

    Maps hidden_state (batch, seq_len, embed_dim) → gate (batch, seq_len, 1).
    Initialized so sigmoid output ≈ 1.0 at start (neutral / no skip).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 1)
        # Initialize to near-1.0 output (sigmoid(2.0) ≈ 0.88)
        with torch.no_grad():
            self.proj.bias.fill_(2.0)
            self.proj.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) — the pre-norm hidden state
        Returns:
            gate: (batch, seq_len, 1) — per-token scalar gate ∈ [0, 1]
        """
        return torch.sigmoid(self.proj(x))
```

**Design rationale**: Single linear layer + sigmoid matches the NeurIPS 2025 Best Paper design (`sigmoid(W_g · q)`). Minimal parameter overhead (256 + 1 = 257 params per gate × 12 gates = 3,084 total). Initialized to near-identity so the model starts equivalent to ungated baseline.

### Step 2: Modify `GatedTransformerBlock` to support both gate modes

**File**: `plugins/tidal/TransformerLM.py`

Add a `gate_mode` parameter to `GatedTransformerBlock.__init__`:
- `"external"` (default): Current behavior — instantiates `DynamicGate(GATE_DIM, embed_dim)` with external signal
- `"input_dependent"`: Instantiates `InputDependentGate(embed_dim)` with hidden state

```python
def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1,
             max_seq_len=512, gate_mode="external"):
    ...
    self.gate_mode = gate_mode
    if gate_mode == "input_dependent":
        self.attn_gate = InputDependentGate(embed_dim)
        self.ffn_gate = InputDependentGate(embed_dim)
    else:
        self.attn_gate = DynamicGate(self.GATE_DIM, embed_dim)
        self.ffn_gate = DynamicGate(self.GATE_DIM, embed_dim)
```

The forward pass changes:
- **External mode**: `attn_output = attn_output * self.attn_gate(gate_signals)` (current, broadcasts `(batch, 1, embed_dim)`)
- **Input-dependent mode**: `attn_output = attn_output * self.attn_gate(normalized)` (broadcasts `(batch, seq_len, 1)` — scalar per token, uniform across dimensions)

Key: In input-dependent mode, `gate_signals` parameter is ignored. The gate reads the pre-norm hidden state directly.

Also add `return_gate_activations` parameter to `forward()`. When True, return gate values alongside the output as a tuple `(output, (attn_gate_vals, ffn_gate_vals))`.

**Checkpoint compatibility**: The two gate modes produce different state_dict keys (`attn_gate.proj.weight` for input-dependent vs. `attn_gate.net.0.weight` for external). Loading a checkpoint saved under one mode into a model configured for the other will raise a `RuntimeError` with a key mismatch. This is correct behavior — cross-mode checkpoint loading is not supported. Add a test to verify this fails clearly.

### Step 3: Modify `TransformerLM` to propagate gate mode and collect activations

**File**: `plugins/tidal/TransformerLM.py`

- Read `GATE_MODE` from config (default: `"external"` for backward compatibility)
- Pass `gate_mode` to each `GatedTransformerBlock`
- Add `return_gate_activations` parameter to `forward()`
- When enabled, collect gate activations from each block into `viz_data["gate_activations"]` — a list of 6 tuples, each `(attn_gate, ffn_gate)` of shape `(batch, seq_len, 1)`

### Step 4: Add gate regularization to `Trainer`

**File**: `plugins/tidal/Trainer.py`

In `_train_epoch`, after computing the cross-entropy loss, add gate regularization:

```python
if self.gate_reg_weight > 0:
    gate_activations = viz_data.get("gate_activations", [])
    gate_loss = sum(g.mean() for pair in gate_activations for g in pair)
    gate_loss = gate_loss / (2 * len(gate_activations))  # normalize by num gates
    loss = loss + self.gate_reg_weight * gate_loss
```

Read `GATE_REG_WEIGHT` from config (default: 0.0).

The forward call in `_train_epoch` needs to pass `return_gate_activations=True` when `gate_reg_weight > 0`.

### Step 5: Add gate analysis to `Evaluator`

**File**: `plugins/tidal/Evaluator.py`

Add method `analyze_gate_activations(max_batches=None)` that:
1. Runs the validation set through the model with `return_gate_activations=True`
2. Accumulates gate activations across all batches
3. Computes per-gate statistics: mean, std, CoV (σ/μ)
4. Computes sparsity fraction: for each (token, layer), `min(attn_gate, ffn_gate) < 0.1`
5. Saves results to `evaluation_results/gate_analysis.json`
6. Returns a dict with:
   - `per_gate_stats`: list of 12 dicts with {mean, std, cov}
   - `sparsity_fraction`: float (C2 metric)
   - `mean_cov`: float (C3 metric)
   - `per_layer_histograms`: gate value distributions

Update `run()` to call `analyze_gate_activations()` when gate mode is `"input_dependent"`.

### Step 6: Config changes

**File**: `plugins/tidal/configs/base_config.yaml` — keep default as `"external"` for backward compatibility:
```yaml
# Gate configuration
GATE_MODE: "external"          # "external" (RL-controlled) or "input_dependent" (learned)
GATE_REG_WEIGHT: 0.0           # L1 regularization weight for gate sparsity (0 = no reg)
```

**New file**: `plugins/tidal/configs/adaptive_depth_config.yaml` — experiment-specific override:
```yaml
# Adaptive Depth experiment (Direction A)
# Inherits all values from base_config.yaml, overrides gate settings only.
GATE_MODE: "input_dependent"
GATE_REG_WEIGHT: 0.0           # Set per-experiment: 0 for Model C, λ* for Model D
RANDOM_SEED: 42
```

This keeps existing RL workflows (`train_rl.py`, `Generator.py`) unaffected. The experiment config is loaded as an overlay on top of base config.

### Step 7: Hypothesis evaluation script

**File**: `plugins/tidal/evaluate_hypothesis.py`

Script that:
1. Takes paths to baseline and gated model evaluation results
2. Reads perplexity values and gate analysis
3. Evaluates C1, C2, C3 against thresholds
4. Prints a clear PASS/FAIL for each condition and overall H1/H0 verdict

---

## Test Plan (TDD — tests written first)

### New test file: `plugins/tidal/tests/test_InputDependentGate.py`

**Tests for `InputDependentGate`**:
1. `test_output_shape` — output is `(batch, seq_len, 1)` for input `(batch, seq_len, embed_dim)`
2. `test_neutral_initialization` — output mean ∈ [0.85, 0.92] at initialization (tight bound around sigmoid(2.0) ≈ 0.88)
3. `test_output_range` — all outputs ∈ [0, 1] (sigmoid guarantees this)
4. `test_gradient_flow` — gradients flow back through the gate to the input

**Tests for `GatedTransformerBlock` with input-dependent mode**:
5. `test_input_dependent_mode_forward` — forward produces correct output shape
6. `test_input_dependent_ignores_gate_signals` — passing `gate_signals` has no effect in input-dependent mode
7. `test_external_mode_backward_compat` — existing external mode still works identically
8. `test_gate_activations_returned` — `return_gate_activations=True` returns correct structure
9. `test_gate_activations_shape` — each gate activation is `(batch, seq_len, 1)` in input-dependent mode

**Tests for `TransformerLM` with input-dependent gates**:
10. `test_model_with_input_dependent_gates` — full model forward pass works
11. `test_gate_activations_collected` — `viz_data["gate_activations"]` has 6 entries (one per layer)
12. `test_gradient_flow_through_gates` — gate parameters receive gradients during training
13. `test_backward_compat_external_mode` — model with `GATE_MODE: "external"` works as before

**Tests for checkpoint compatibility**:
14. `test_cross_mode_checkpoint_fails_clearly` — loading an `"external"` mode checkpoint into an `"input_dependent"` model raises `RuntimeError` with key mismatch (and vice versa)
15. `test_same_mode_checkpoint_roundtrip` — saving and loading an `"input_dependent"` checkpoint works correctly, all keys match

**Tests for fixed-gate baseline (Model B)**:
16. `test_frozen_gates_produce_constant_output` — with gate weights frozen at init, all tokens get the same gate value ≈ 0.88 regardless of input
17. `test_frozen_gates_no_gradient` — gate parameters have `requires_grad=False` and receive no gradients

**Tests for gate regularization in `Trainer`**:
18. `test_gate_reg_loss_computed` — regularization loss is non-zero when gates are active
19. `test_gate_reg_zero_when_disabled` — no regularization when `GATE_REG_WEIGHT: 0.0`

**Tests for gate analysis in `Evaluator`**:
20. `test_gate_analysis_output_structure` — returns dict with expected keys
21. `test_sparsity_fraction_range` — sparsity fraction ∈ [0, 1]
22. `test_cov_computation` — CoV computed correctly for known inputs

**Tests for hypothesis evaluation**:
23. `test_h1_all_pass` — returns H1 when all conditions met
24. `test_h0_quality_fail` — returns H0 when C1 fails
25. `test_h0_sparsity_fail` — returns H0 when C2 fails
26. `test_h0_adaptivity_fail` — returns H0 when C3 fails

---

## Files Modified

| File | Change |
|------|--------|
| `plugins/tidal/TransformerLM.py` | Add `InputDependentGate` class; modify `GatedTransformerBlock` for dual gate mode (`gate_mode` param controls which gate class is instantiated); add gate activation collection to `TransformerLM.forward()` |
| `plugins/tidal/Trainer.py` | Add gate regularization loss to `_train_epoch`; add random seed support |
| `plugins/tidal/Evaluator.py` | Add `analyze_gate_activations()` method |
| `plugins/tidal/configs/base_config.yaml` | Add `GATE_MODE` (default: `"external"`) and `GATE_REG_WEIGHT` (default: 0.0) |
| `plugins/tidal/configs/adaptive_depth_config.yaml` | New: experiment-specific config overlay |
| `plugins/tidal/evaluate_hypothesis.py` | New: hypothesis evaluation script |
| `plugins/tidal/tests/test_InputDependentGate.py` | New: 26 tests covering gates, blocks, model, checkpoints, trainer, evaluator, hypothesis |

## Files NOT Modified

| File | Reason |
|------|--------|
| `DynamicGate` class | Kept intact — `GATE_MODE: "external"` still uses it |
| `GatingModulator.py` | Not needed — input-dependent gates don't use external signals |
| `GatingPolicyAgent.py` | Not needed — no RL controller in this experiment |
| `RLTrainer.py` | Not needed — Phase 2 RL is not part of this hypothesis |
| `RewardComputer.py` | Not needed |
| `GatingEnvironment.py` | Not needed |
| `Generator.py` | Gate mode is inference-only concern; generator reads model config |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Gates collapse to constant ~1.0 (no learning) | Medium | L1 regularization sweep; monitor gate stats during training; Model B (fixed-gate) serves as detection baseline |
| Gates collapse to constant ~0.0 (kills model) | Low | Initialization bias=2.0 starts at 0.88; L1 reg pushes toward 0 gradually, not catastrophically |
| Quality degrades beyond 5% threshold | Medium | Two-stage λ sweep with Pareto analysis on `val_tune`; report frontier even if H0 |
| Training instability from gate gradients | Low | Gates are a single linear layer (257 params each) — minimal gradient pathology risk |
| Checkpoint incompatibility | Low | Cross-mode loading fails with clear `RuntimeError` (tested). Same-mode roundtrip verified. Config default stays `"external"` |
| Multiple comparisons inflate H1 probability | Addressed | Two-stage protocol: λ selected on `val_tune`, H1 evaluated on held-out `val_eval` |
| AMP numerical instability in sigmoid gates | Low | Sigmoid outputs are well-behaved in float16 range; monitor for NaN during training |
| Confounding: gates help via constant scaling, not input dependence | Addressed | Model B (frozen-gate) controls for this explicitly |

---

## Execution Order

### Implementation (TDD)
1. Write all 26 tests — expect all to fail
2. Implement `InputDependentGate` (Step 1) — gate tests pass
3. Modify `GatedTransformerBlock` (Step 2) — block + checkpoint tests pass
4. Modify `TransformerLM` (Step 3) — model + frozen-gate tests pass
5. Add gate regularization to `Trainer` (Step 4) — trainer tests pass
6. Add gate analysis to `Evaluator` (Step 5) — evaluator tests pass
7. Implement hypothesis evaluation script (Step 7) — hypothesis tests pass
8. Create configs (Step 6) — `base_config.yaml` default unchanged, new `adaptive_depth_config.yaml`

### Experimentation
9. Train Model A (ungated baseline, seed=42)
10. Train Model B (fixed-gate, frozen at init, seed=42)
11. Train Model C (learned gates, λ=0, seed=42)
12. **Stage 1**: λ sweep on `val_tune` — train 5 models with λ ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
13. Select and pre-register λ*
14. Train Model D (learned gates, λ=λ*, seed=42, fresh)
15. **Stage 2**: Evaluate C1, C2, C3 on `val_eval` for Model D
16. Report H1 or H0
17. Write ADR documenting the outcome regardless of H1/H0
