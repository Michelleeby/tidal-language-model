# Tidal Language Model

A **Gated Transformer** language model (~30.7M parameters) with an **RL-learned gating controller** that modulates generation behavior at inference time. The model is pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) using GPT-2 BPE tokenization (50,257 vocab), then a PPO agent learns to control three gate signals — **creativity**, **focus**, and **stability** — that dynamically scale attention and FFN outputs across all transformer layers.

## Architecture

### Phase 1 — Language Model (`TransformerLM`)

```
Input tokens
  → Token Embeddings (256D) + Positional Embeddings
  → 6 × GatedTransformerBlock
  → LayerNorm → Output Projection → Logits (50,257)
```

Each `GatedTransformerBlock` contains two `DynamicGate` modules (one for attention, one for FFN). These are small MLPs (`3 → 32 → 256 → sigmoid`) that convert 3D gate signals into per-dimension scaling factors applied to the residual stream. Initialized with bias = 2.0 so `sigmoid(output) ≈ 1.0` at the start (neutral — no effect). When no gate signals are provided, gates produce unit scaling.

Training uses cross-entropy loss with gradient accumulation, mixed-precision (AMP), cosine LR annealing with warmup, and `torch.compile`.

### Phase 2 — RL Gating Controller (`GatingPolicyAgent`)

A PPO actor-critic agent observes a **64D** vector (token statistics, chunked hidden-state summaries, generation context) and outputs continuous actions in `[0, 1]` via a Beta distribution:

| Signal | Effect |
|---|---|
| **creativity** | Scales sampling temperature (`0.5×` – `1.5×` base) |
| **focus** | Adds position-based attention bias toward recent context |
| **stability** | Scales repetition penalty (`1×` – `2.5×` base) |

The `GatingModulator` maps actions to generation parameters, and the `RewardComputer` provides dense per-step rewards from perplexity, diversity, repetition, and coherence components.

### Real-Time Dashboard

A monitoring dashboard built with **Fastify 5 + React 19 + Redis 7** (replacing the deprecated Streamlit dashboard). Training metrics flow from `MetricsLogger` → Redis (real-time SSE) + JSONL (archival) → Fastify API → React frontend.

## Project Structure

```
tidal-language-model/
├── TransformerLM.py          # Gated Transformer language model
├── GatingPolicyAgent.py      # PPO actor-critic for gate control
├── GatingEnvironment.py      # RL environment wrapping the LM
├── GatingModulator.py        # Maps gate signals → generation params
├── RewardComputer.py         # Multi-component reward function
├── Main.py                   # Phase 1 training entry point
├── train_rl.py               # Phase 2 RL training entry point
├── Trainer.py                # LM training loop
├── RLTrainer.py              # PPO training loop
├── Generator.py              # Text generation (with optional RL gating)
├── Evaluator.py              # Model evaluation
├── DataPipeline.py           # TinyStories loading + GPT-2 BPE tokenization
├── MetricsLogger.py          # Redis + JSONL metrics logging
├── DynamicLRScheduler.py     # Cosine annealing with warmup
├── Utils.py                  # Shared utilities
├── configs/                  # YAML config files (gitignored)
├── experiments/              # Checkpoints and run artifacts
├── tests/                    # Unit tests (unittest)
├── dashboard/                # Fastify + React monitoring dashboard
│   ├── docker-compose.yml    # Redis 7 service
│   └── packages/
│       ├── server/           # Fastify 5 API (port 4400)
│       ├── client/           # React 19 / Vite 6 (port 5173)
│       └── shared/           # Shared types
├── legacy_research/          # Original physics-based architecture
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU
- Node.js 20+ and Docker (for the dashboard)

### Environment

```bash
python3 -m venv tidal-env
source tidal-env/bin/activate
pip3 install -r requirements.txt
```

Install PyTorch with CUDA support (set the version for your GPU):

```bash
CUDA_VERSION="cu129"
pip3 install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
```

## Usage

### Train the Language Model (Phase 1)

```bash
python3 Main.py --config configs/base_config.yaml
```

Checkpoints are saved as raw `state_dict` files to `experiments/<experiment_id>/`.

### Train the RL Gating Agent (Phase 2)

Requires a trained TransformerLM checkpoint:

```bash
python3 train_rl.py \
    --config configs/base_config.yaml \
    --rl-config configs/rl_config.yaml \
    --checkpoint experiments/<experiment_id>/transformer-lm_v1.0.0.pth
```

### Generate Text

Standard generation:

```bash
python3 Generator.py \
    --config configs/base_config.yaml \
    --checkpoint experiments/<experiment_id>/checkpoint_foundational_epoch_1.pth \
    --prompt "Once upon a time" --max_tokens 50 --temperature 0.8 --top_k 50
```

With RL-controlled gating:

```bash
python3 Generator.py \
    --config configs/base_config.yaml \
    --checkpoint <model_checkpoint> \
    --rl-agent --rl-checkpoint <rl_checkpoint> \
    --prompt "Once upon a time"
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

### Start the Dashboard

```bash
cd dashboard && docker compose up -d && npm run dev
```

## Legacy

The original architecture used an N-body physics simulation in 2D space with a "Semantic Endocrine System" for modulating embeddings. That work has been archived to `legacy_research/` along with its [README](legacy_research/README.md).

## License

[GPL-3.0](LICENSE)
