---
name: analyze-run
description: Analyze a Tidal training run (foundational LM or RL gating). Use when the user provides CSVs, screenshots, metrics, or mentions analyzing an experiment, training run, reward shaping, gating signals, or PPO results.
---

# Analyze Training Run

When the user provides data from a training run, follow this structured analysis methodology:

## 1. Identify the Training Phase
- **Foundational (Phase 1):** Cross-entropy loss on TinyStories, TransformerLM
- **RL Gating (Phase 2):** PPO agent controlling a single **modulation** gate on a conservative-to-exploratory axis (0.0 = conservative, 1.0 = exploratory). See [ADR 0001](research/adrs/0001-single-modulation-gate.md).

## 2. For Foundational Runs, Check:
- **Loss curve shape:** Is it monotonically decreasing? Any plateaus?
- **Learning rate schedule:** Did warmup and cosine annealing behave correctly?
- **Gradient norms:** Any spikes indicating instability? (max_grad_norm=1.0 clipping)
- **Validation vs training loss:** Gap indicates overfitting
- **Perplexity trajectory:** Should decrease, target varies by corpus size
- **Data pipeline:** Tokenized cache should be uint16 (not int64). If data loading is slow on a remote GPU, check whether the pre-built cache was transferred. See [ADR 0002](research/adrs/0002-uint16-data-cache.md).

## 3. For RL Gating Runs, Check:

### 3a. Determine Constraint Mode
The RL trainer operates in one of two modes controlled by `RL_CONSTRAINT_MODE`:

- **`"weighted"` (default):** All reward components in a single weighted sum. Optional `DiversityHomeostasis` controller adaptively boosts diversity weight when diversity drops below target. See [ADR 0005](research/adrs/0005-diversity-homeostasis.md).
- **`"lagrangian"`:** Diversity is enforced as a **hard constraint** via a learned Lagrange multiplier, not as a reward component. Primary reward = perplexity + sampling + coherence. See [ADR 0006](research/adrs/0006-ppo-lagrangian-diversity-constraint.md).

### 3b. Common Metrics (Both Modes)
- **Episode rewards:** Trend and variance over episodes
- **Modulation gate signal:** Is it being modulated across steps and prompts, or stuck at a constant?
  - Beta distribution output should show variance, not collapse to 0 or 1
  - If modulation hovers near 0.5 with near-zero variance, the agent hasn't learned a meaningful policy
  - A single time series (not three) — all four generation parameters (temperature, top-k, top-p, repetition penalty) move together along the conservative-to-exploratory axis
- **Explained variance:** Should increase toward 1.0 over training
- **Policy loss vs value loss:** Both should decrease; if policy loss diverges, check entropy coefficient
- **KL divergence:** Should stay bounded (PPO clipping working)

### 3c. Entropy Homeostasis (Both Modes)
- **Target:** `-0.35` for the 1D Beta distribution (recalibrated from `-1.0` for the old 3D system). See [ADR 0003](research/adrs/0003-recalibrate-entropy-homeostasis-and-reward-weights.md).
- **Check:** Did homeostasis activate? Look for `entropy_coef` increasing beyond its initial value (default 0.01)
- **If it never activated:** The agent may have locked into a single modulation value without adaptive correction — check if observed policy entropy stayed above -0.35

### 3d. Weighted Mode Specifics
- **Reward components breakdown (weights sum to 1.0):**
  - Perplexity: 0.35 (quality — lower perplexity = better)
  - Coherence: 0.15 (quality — semantic consistency)
  - Diversity: 0.15 (entropy — vocabulary richness)
  - Sampling: 0.15 (entropy — post-filtered distribution entropy)
  - Repetition: 0.20 (penalty — penalizes repeated n-grams)
- **Quality (50%) should outweigh entropy (30%)** — if the agent satisfices on entropy metrics rather than optimizing quality, the balance may need further tuning
- **DiversityHomeostasis** (if enabled via `RL_DIVERSITY_HOMEOSTASIS_TARGET`):
  - Check if diversity weight was boosted above its 0.15 baseline (max 0.35)
  - Target is 0.55 — if mean diversity stays well below this despite weight boosting, the weighted formulation may be insufficient (consider switching to Lagrangian mode)

### 3e. Lagrangian Mode Specifics
- **Primary reward components (renormalized weights):**
  - Perplexity: 0.54 (quality)
  - Sampling: 0.23 (entropy)
  - Coherence: 0.23 (quality)
  - Diversity: 0 (enforced via constraint, not reward)
  - Repetition: 0 (already handled by GatingModulator's logit-level penalty)
- **Diversity constraint:** `diversity_reward >= threshold` (default 0.55). Check `mean_cost` — should trend toward 0 as constraint is satisfied
- **Lagrange multiplier trajectory:** Should rise when diversity is below threshold and decay (via weight_decay=0.01) when satisfied. Oscillation indicates the dual learning rate is too high
- **Cost critic:** `cost_value_loss` should decrease, indicating the cost critic is learning to predict constraint violations
- **Combined advantage:** `A_combined = (A_reward - lambda * A_cost) / (1 + lambda)`. If lambda grows very large, the effective learning rate for the primary reward shrinks — check whether the primary reward stalls as the multiplier climbs

## 4. Red Flags to Call Out
- Reward plateau before convergence → learning rate too low or reward shaping issue
- Gate collapse (modulation stuck near 0 or 1) → Beta concentration params too high, or entropy homeostasis not activating
- Explained variance < 0 → value function worse than mean predictor
- **Weighted mode:** Diversity collapse (diversity << 0.55) despite DiversityHomeostasis → structural limitation of weighted formulation, consider Lagrangian mode
- **Lagrangian mode:** Lagrange multiplier growing without bound → constraint is persistently violated, check if diversity threshold is realistic for the model/dataset
- **Lagrangian mode:** Primary reward stalling while multiplier is high → the `(1 + lambda)` normalization is shrinking the effective reward gradient; the agent is spending all its capacity satisfying the constraint
- Entropy homeostasis never activating (entropy_coef stays flat) → check if target (-0.35) is appropriate for the observed entropy range
- Gradient norm consistently at clip value → model struggling, consider architecture changes

## 5. Recommendations Format
After analysis, provide:
1. **Verdict:** Effective / Partially Effective / Ineffective
2. **Key Findings:** Bullet the 2-3 most important observations
3. **Constraint Mode Assessment:** Is the current mode (weighted/lagrangian) appropriate given the observed diversity and reward dynamics?
4. **Next Experiment Suggestion:** Concrete config change to try next
5. **Data to Archive:** Note which files should be saved for the research log
