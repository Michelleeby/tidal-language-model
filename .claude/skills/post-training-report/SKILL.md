---
name: post-training-report
description: Generate a comprehensive post-training analysis report for a completed experiment. Use when the user mentions "/post-training-report", asks to create a training report, or wants automated analysis of a completed LM or RL experiment in the dashboard.
---

# Post-Training Report

Generate a rich analysis report for a completed Tidal experiment, combining interactive charts with AI-generated analysis paragraphs, and persist it in the dashboard Reports tab.

## 1. Identify the Experiment

- If the user provides an experiment ID, use that.
- Otherwise, call `list_experiments()` and pick the most recently completed experiment.
- Confirm the experiment ID with the user before proceeding.

## 2. Determine Experiment Type

Read the `experimentType` field from the experiment listing:
- `"lm"` → Language Model pretraining (Phase 1)
- `"rl"` → RL Gating Controller (Phase 2)
- `"unknown"` → Check `hasRLMetrics`: if true treat as RL, otherwise treat as LM.

## 3. Fetch Data

Call MCP tools **in parallel** to gather all available data:

**For LM experiments:**
- `get_metrics(expId, mode="historical")` — full loss/LR/perplexity time series
- `get_status(expId)` — training status, timing, steps
- `get_evaluation(expId)` — perplexity score + generated samples
- `get_checkpoints(expId)` — checkpoint list

**For RL experiments:**
- `get_rl_metrics(expId)` — full RL training history
- `get_status(expId)` — training status, timing, steps
- `get_evaluation(expId)` — evaluation results
- `get_ablation(expId)` — ablation study comparisons
- `get_checkpoints(expId)` — checkpoint list

## 4. Analyze the Data

Apply the analysis framework from the `/analyze-run` skill methodology:

### For LM Experiments
- **Loss curve shape:** Monotonically decreasing? Plateaus?
- **Learning rate schedule:** Did warmup and cosine annealing behave correctly?
- **Gradient norms:** Spikes indicating instability? (max_grad_norm=1.0 clipping)
- **Perplexity trajectory:** Decreasing trend?
- **Final evaluation perplexity** and quality of generated samples

### For RL Experiments

#### Determine Constraint Mode
Check if the RL metrics contain `lagrange_multiplier` data:
- If present → **Lagrangian mode** (`RL_CONSTRAINT_MODE="lagrangian"`)
- If absent → **Weighted mode** (`RL_CONSTRAINT_MODE="weighted"`)

#### Common Metrics (Both Modes)
- **Episode rewards:** Trend and variance
- **Modulation gate signal:** Is it being modulated or stuck? Beta distribution should show variance
- **Explained variance:** Should trend toward 1.0
- **Policy loss vs value loss:** Both should decrease
- **Entropy homeostasis:** Target is -0.35. Did `entropy_coef` increase beyond initial value?

#### Weighted Mode Specifics
- Reward component breakdown (perplexity: 0.35, coherence: 0.15, diversity: 0.15, sampling: 0.15, repetition: 0.20)
- DiversityHomeostasis activation (diversity_weight boosted above 0.15 baseline?)

#### Lagrangian Mode Specifics
- Primary reward components (perplexity: 0.54, sampling: 0.23, coherence: 0.23)
- Lagrange multiplier trajectory: rises when diversity below threshold, decays when satisfied
- `mean_cost` trend: should approach 0
- `cost_value_loss`: should decrease
- Combined advantage normalization effects

### Red Flags to Call Out
- Reward plateau before convergence
- Gate collapse (modulation stuck near 0 or 1)
- Explained variance < 0
- Weighted: diversity collapse despite homeostasis
- Lagrangian: multiplier growing without bound, primary reward stalling
- Entropy homeostasis never activating

## 5. Build the Blocks Array

Construct a BlockNote JSON blocks array. Every block MUST have `id` (use `crypto.randomUUID()` format, e.g. a random hex string), `type`, and `children: []`.

### Block Format Reference

**Heading block:**
```json
{
  "id": "<uuid>",
  "type": "heading",
  "props": { "level": 1 },
  "content": [{ "type": "text", "text": "Title Here" }],
  "children": []
}
```

**Paragraph block:**
```json
{
  "id": "<uuid>",
  "type": "paragraph",
  "content": [{ "type": "text", "text": "Paragraph text here." }],
  "children": []
}
```

**Bold text in paragraph:**
```json
{
  "id": "<uuid>",
  "type": "paragraph",
  "content": [
    { "type": "text", "text": "Bold text", "styles": { "bold": true } },
    { "type": "text", "text": " — Normal text" }
  ],
  "children": []
}
```

**LM chart block:**
```json
{
  "id": "<uuid>",
  "type": "experimentChart",
  "props": { "experimentId": "<expId>", "metricKey": "Losses/Total", "chartMode": "lm" },
  "children": []
}
```

**RL chart block:**
```json
{
  "id": "<uuid>",
  "type": "experimentChart",
  "props": { "experimentId": "<expId>", "chartMode": "rl", "rlMetricKey": "episode_rewards" },
  "children": []
}
```

**Ablation chart block:**
```json
{
  "id": "<uuid>",
  "type": "experimentChart",
  "props": { "experimentId": "<expId>", "chartMode": "ablation", "ablationMetricKey": "mean_reward" },
  "children": []
}
```

### Report Structure

Build the blocks array with this structure:

#### Header
- **h1:** `{expId} Post-Training Analysis`

#### Figures and Data (h2)

**For LM experiments — 4 chart sections:**
1. **Losses/Total** — `metricKey: "Losses/Total"`, `chartMode: "lm"` + analysis paragraph
2. **Perplexity** — `metricKey: "Perplexity"`, `chartMode: "lm"` + analysis paragraph
3. **Learning Rate** — `metricKey: "Learning Rate"`, `chartMode: "lm"` + analysis paragraph
4. **Iterations/Second** — `metricKey: "Iterations/Second"`, `chartMode: "lm"` + analysis paragraph

**For RL experiments — core charts + mode-specific:**

Core (always include):
1. `episode_rewards` + analysis
2. `policy_loss` + analysis
3. `value_loss` + analysis
4. `entropy` + analysis
5. `gate_modulation` + analysis
6. `explained_variance` + analysis

Reward components (include if data is present):
7. `reward_perplexity` + analysis
8. `reward_diversity` + analysis
9. `reward_coherence` + analysis
10. `reward_repetition` + analysis
11. `reward_sampling` + analysis

Lagrangian-specific (include only if `lagrange_multiplier` data is present):
12. `lagrange_multiplier` + analysis
13. `mean_cost` + analysis
14. `cost_value_loss` + analysis
15. `entropy_coef` + analysis
16. `diversity_weight` + analysis

Weighted-specific (include only if in weighted mode and data is present):
- `entropy_coef` + analysis
- `diversity_weight` + analysis

Ablation (include if experiment `hasAblation`):
- Ablation chart: `chartMode: "ablation"`, `ablationMetricKey: "mean_reward"` + analysis

Each chart section follows this pattern:
1. Bold figure title paragraph (e.g., "**Figure 1: Training Loss**")
2. experimentChart block
3. Analysis paragraph describing what the chart shows

#### Analysis (h2)
Full analysis applying the checklist from Step 4. Write 2-4 paragraphs covering:
- Overall training trajectory assessment
- Key metric interactions and dynamics
- Constraint mode assessment (for RL)
- Comparison to expected behavior

#### Outcome and Next Steps (h2)
Following the `/analyze-run` recommendations format:
1. **Verdict:** Effective / Partially Effective / Ineffective
2. **Key Findings:** 2-3 bullet points
3. **Constraint Mode Assessment** (RL only): Is the current mode appropriate?
4. **Next Experiment Suggestion:** Concrete config change to try
5. **Evaluation Summary:** If evaluation data is available, mention final perplexity and sample quality

## 6. Create the Report

Execute these two MCP calls **sequentially**:

1. **Scaffold:** `generate_report(pattern="experiment-overview", experimentId=<expId>, title="<expId> Post-Training Analysis")`
   - Extract the report ID from the response: `response.report.id`

2. **Enrich:** `update_report(reportId=<id>, title="<expId> Post-Training Analysis", blocks=<your blocks array>)`
   - This replaces the scaffolded blocks with the full enriched content

## 7. Present to User

Output:
- The dashboard URL: `http://localhost:5173/reports/<reportId>`
- A brief summary of the report contents (experiment type, number of charts, verdict)
- Any red flags or notable findings worth highlighting immediately
