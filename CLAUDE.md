# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtualenv
source tidal-env/bin/activate

# Run all tests (no pytest — use unittest)
python -m unittest discover -s tests -v                      # infrastructure tests
python -m unittest discover -s plugins/tidal/tests -v        # model tests

# Run a single test file
python -m unittest plugins.tidal.tests.test_TransformerLM -v

# Run a single test method
python -m unittest plugins.tidal.tests.test_TransformerLM.TestTransformerLM.test_forward_pass -v

# Train the language model
python3 plugins/tidal/Main.py --config plugins/tidal/configs/base_config.yaml

# Train the RL gating agent (requires a trained TransformerLM checkpoint)
python3 plugins/tidal/train_rl.py \
    --config plugins/tidal/configs/base_config.yaml \
    --rl-config plugins/tidal/configs/rl_config.yaml \
    --checkpoint experiments/<experiment_id>/transformer-lm_v1.0.0.pth

# Generate text
python3 plugins/tidal/Generator.py \
    --config plugins/tidal/configs/base_config.yaml \
    --checkpoint experiments/<experiment_id>/checkpoint_foundational_epoch_1.pth \
    --prompt "Once upon a time" --max_tokens 50 --temperature 0.8 --top_k 50

# Generate text with RL-controlled gating
python3 plugins/tidal/Generator.py \
    --config plugins/tidal/configs/base_config.yaml \
    --checkpoint <model_checkpoint> \
    --rl-agent --rl-checkpoint <rl_checkpoint> \
    --prompt "Once upon a time"

# Start dashboard (Redis via Docker Compose + Fastify + Vite)
cd dashboard && docker compose up -d && npm run dev

# Start dashboard server only (port 4400)
npm run server:dev -w dashboard

# Start dashboard client only (port 5173, proxies /api to 4400)
npm run client:dev -w dashboard

# Build dashboard for production
cd dashboard && npm run build
```

## Architecture

The system has two training phases:

**Phase 1 — Language Model Pretraining** (`plugins/tidal/Main.py` → `plugins/tidal/Trainer.py`):
Standard cross-entropy training of `TransformerLM` on TinyStories (HuggingFace `roneneldan/TinyStories`) with GPT-2 BPE tokenization (50257 vocab). Uses gradient accumulation, mixed-precision (AMP), cosine LR annealing with warmup, and `torch.compile`. Checkpoints are raw `state_dict` files (not wrapped in metadata dicts). Experiments get unique IDs from git commit hash + config file hash.

**Phase 2 — RL Gating Controller** (`plugins/tidal/train_rl.py` → `plugins/tidal/RLTrainer.py`):
A PPO agent (`GatingPolicyAgent`) learns to control 1 gate signal — [modulation] on a conservative-to-exploratory axis — that modulates generation behavior of a **frozen** TransformerLM. The agent observes a 64D vector (token statistics, hidden state summaries, context) and outputs continuous actions in [0, 1] via a Beta distribution. The `GatingModulator` maps actions to generation parameters (temperature, repetition penalty, attention bias). The `RewardComputer` provides dense per-step rewards from perplexity, diversity, repetition, and coherence components.

**Gating mechanism in the model**: Each of the 6 `GatedTransformerBlock` layers contains two `DynamicGate` modules (one for attention output, one for FFN output). These are small MLPs (1→32→embed_dim→sigmoid) that convert the 1D gate signal into per-dimension scaling factors. Initialized with bias=2.0 so sigmoid output starts near 1.0 (neutral). When `gate_signals=None`, gates produce unit scaling.

**Key data flow**: `plugins/tidal/DataPipeline.py` loads TinyStories, tokenizes with GPT-2 BPE, flattens all tokens, and chunks into fixed-length sequences. Each sample is `(input_ids, target_ids)` with standard causal LM shift (chunk[:-1], chunk[1:]).

**Plugin system**: Model-specific code lives in `plugins/tidal/` with a `manifest.yaml` describing training phases, checkpoint patterns, generation config, metrics config, and infrastructure requirements. The dashboard reads the manifest via `PluginRegistry` instead of hardcoding model knowledge. Shared infrastructure (`worker_agent.py`, `MetricsLogger.py`, `experiment_utils.py`) stays at the project root.

## Important Notes

- `Trainer.py` saves checkpoints as raw `model.state_dict()` (handles `torch.compile` via `_orig_mod`). `Evaluator` and `Generator` load them directly with `model.load_state_dict()`.
- RL checkpoints (from `RLTrainer`) are dicts with keys: `agent_state_dict`, `optimizer_state_dict`, `global_step`, `config`.
- `Evaluator.__init__` creates its own `TransformerLM` using `config.get("VOCAB_SIZE")` — this must match the checkpoint.
- `MetricsLogger` writes to both Redis (real-time SSE) and JSONL on disk (archival). Redis is optional — gracefully degrades to disk-only if unavailable.
- `research/legacy/dashboard.py` is the deprecated Streamlit dashboard. The replacement is in `dashboard/` (Fastify + React).
- Legacy physics-based architecture lives in `research/legacy/` — do not import from there.

## MUST USE INSTRUCTIONS

- You ALWAYS use Test Driven Development (TDD),
    - You write tests first.
    - You make sure all tests fail.
    - You develop code.
    - You make sure all tests pass.
- You ALWAYS plan before you code. When you plan,
    - You consider design patterns (and the antipatterns they address).
    - You make tradeoffs clear up front.
