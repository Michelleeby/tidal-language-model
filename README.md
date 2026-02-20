# Tidal Language Model

A **Gated Transformer** language model (~30.7M parameters) with an **RL-learned gating controller** that modulates generation behavior at inference time. The model is pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) using GPT-2 BPE tokenization (50,257 vocab), then a PPO agent learns to control a single **modulation** gate on a conservative-to-exploratory axis that dynamically scales attention and FFN outputs across all transformer layers.

## Architecture

### Phase 1 — Language Model (`TransformerLM`)

```
Input tokens
  → Token Embeddings (256D) + Positional Embeddings
  → 6 × GatedTransformerBlock
  → LayerNorm → Output Projection → Logits (50,257)
```

Each `GatedTransformerBlock` contains two `DynamicGate` modules (one for attention, one for FFN). These are small MLPs (`1 → 32 → 256 → sigmoid`) that convert a 1D modulation signal into per-dimension scaling factors applied to the residual stream. Initialized with bias = 2.0 so `sigmoid(output) ≈ 1.0` at the start (neutral — no effect). When no gate signals are provided, gates produce unit scaling.

Training uses cross-entropy loss with gradient accumulation, mixed-precision (AMP), cosine LR annealing with warmup, and `torch.compile`.

### Phase 2 — RL Gating Controller (`GatingPolicyAgent`)

A PPO actor-critic agent observes a **64D** vector (token statistics, chunked hidden-state summaries, generation context) and outputs a single continuous action in `[0, 1]` via a Beta distribution — the **modulation** signal:

- **0.0 = conservative**: low temperature, narrow top-k, tight top-p, mild repetition penalty
- **1.0 = exploratory**: high temperature, wide top-k, full top-p, strong repetition penalty

All four generation parameters move monotonically with modulation, eliminating the redundancy of the [previous 3-gate design](research/adrs/0001-single-modulation-gate.md). The `GatingModulator` maps the action to generation parameters, and the `RewardComputer` provides dense per-step rewards from perplexity, diversity, sampling entropy, repetition, and coherence components (weighted 50% quality, 30% entropy, 20% penalty).

### Real-Time Dashboard

A monitoring dashboard built with **Fastify 5 + React 19 + Redis 7** (replacing the deprecated Streamlit dashboard). Training metrics flow from `MetricsLogger` → Redis (real-time SSE) + JSONL (archival) → Fastify API → React frontend. An **MCP server** (`dashboard/packages/mcp/`) exposes experiment data to Claude Code with a [lazy disk cache](research/adrs/0004-lazy-disk-cache-for-mcp-http-client.md) that eliminates redundant network calls for completed experiments.

![Training metrics dashboard](docs/images/dashboard-training.png)

![RL gating dashboard with reward curves and ablation comparison](docs/images/dashboard-rl-gating.png)

### Job Orchestration

The dashboard includes a job orchestration system that manages training runs as background processes. A worker agent (`worker_agent.py`) spawns and monitors training subprocesses, sending heartbeats every 10 seconds and polling for control signals. Jobs progress through a lifecycle: `pending` → `provisioning` → `starting` → `running` → `completed` / `failed` / `cancelled`. Only one LM training job can run at a time. Health checks detect stale jobs after 60 seconds without a heartbeat.

**Dashboard controls** let you create, monitor, and stop jobs via the API:

```bash
# Create an LM training job
curl -X POST http://localhost:4400/api/jobs \
  -H "Authorization: Bearer $TIDAL_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"type": "lm-training", "configPath": "configs/base_config.yaml"}'

# Create an RL training job
curl -X POST http://localhost:4400/api/jobs \
  -H "Authorization: Bearer $TIDAL_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"type": "rl-training", "configPath": "configs/base_config.yaml", "rlConfigPath": "configs/rl_config.yaml", "checkpoint": "experiments/<id>/transformer-lm_v1.0.0.pth"}'

# Gracefully stop a running job
curl -X POST http://localhost:4400/api/jobs/<jobId>/signal \
  -H "Authorization: Bearer $TIDAL_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"signal": "complete"}'
```

### Authentication & Rate Limiting

**Bearer token authentication** protects all mutating dashboard endpoints (creating jobs, sending signals, cancelling). The token is set via the `TIDAL_AUTH_TOKEN` environment variable and compared using timing-safe equality to prevent timing attacks. Read-only endpoints (listing jobs, SSE streams) are unauthenticated.

```bash
# Set the auth token before starting the server
export TIDAL_AUTH_TOKEN="your-secret-token-here"
```

Requests to protected endpoints must include the header:

```
Authorization: Bearer your-secret-token-here
```

| Endpoint | Auth Required |
|---|---|
| `POST /api/jobs` | Yes |
| `POST /api/jobs/:jobId/signal` | Yes |
| `POST /api/jobs/:jobId/cancel` | Yes |
| `GET /api/jobs`, `GET /api/jobs/active`, `GET /api/jobs/:jobId` | No |
| `GET /api/jobs/stream`, `GET /api/experiments/:expId/stream` | No |
| `POST /api/generate` | No (rate limited) |

**Rate limiting** is applied to the `POST /api/generate` endpoint using a Redis-backed token bucket (5 requests, refilling 1 token per 6 seconds). Rate limit status is returned in `X-RateLimit-Limit` and `X-RateLimit-Remaining` response headers. If Redis is unavailable, rate limiting fails open.

### Generation with RL-Controlled Gating

![Text generation with RL-controlled gating](docs/images/generation-output.png)

## Project Structure

```
tidal-language-model/
├── plugins/tidal/               # Model plugin (all model-specific code)
│   ├── TransformerLM.py         # Gated Transformer language model
│   ├── GatingPolicyAgent.py     # PPO actor-critic for gate control
│   ├── GatingEnvironment.py     # RL environment wrapping the LM
│   ├── GatingModulator.py       # Maps modulation signal → generation params
│   ├── RewardComputer.py        # Multi-component reward function
│   ├── Main.py                  # Phase 1 training entry point
│   ├── train_rl.py              # Phase 2 RL training entry point
│   ├── Trainer.py               # LM training loop
│   ├── RLTrainer.py             # PPO training loop (with entropy homeostasis)
│   ├── Generator.py             # Text generation (with optional RL gating)
│   ├── Evaluator.py             # Model evaluation
│   ├── DataPipeline.py          # TinyStories loading + GPT-2 BPE (uint16 cache)
│   ├── TrajectoryAnalyzer.py    # Gate signal trajectory analysis
│   ├── DynamicLRScheduler.py    # Cosine annealing with warmup
│   ├── migrate_checkpoint.py    # 3-gate → 1-gate checkpoint migration
│   ├── inference_server.py      # Inference server for dashboard
│   ├── Utils.py                 # Shared utilities
│   ├── manifest.yaml            # Plugin descriptor (phases, checkpoints, metrics)
│   ├── configs/                 # YAML config files (gitignored)
│   └── tests/                   # Model unit tests (unittest)
├── MetricsLogger.py             # Redis + JSONL metrics logging
├── worker_agent.py              # Job worker (spawns training subprocesses)
├── experiment_utils.py          # Experiment ID generation and utilities
├── tests/                       # Infrastructure unit tests (unittest)
├── dashboard/                   # Fastify + React monitoring dashboard
│   ├── docker-compose.yml       # Redis 7 service
│   └── packages/
│       ├── server/              # Fastify 5 API (port 4400)
│       │   └── src/
│       │       ├── plugins/     # Auth & rate limiting
│       │       ├── services/    # Job orchestrator, store, worker spawner
│       │       └── routes/      # REST + SSE endpoints
│       ├── client/              # React 19 / Vite 6 (port 5173)
│       ├── mcp/                 # MCP server (with lazy disk cache)
│       └── shared/              # Shared types
├── research/                    # ADRs, session logs, experiment data
├── experiments/                 # Checkpoints and run artifacts
├── scripts/                     # Utility scripts
├── legacy_research/             # Original physics-based architecture
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
python3 plugins/tidal/Main.py --config plugins/tidal/configs/base_config.yaml
```

Checkpoints are saved as raw `state_dict` files to `experiments/<experiment_id>/`.

### Train the RL Gating Agent (Phase 2)

Requires a trained TransformerLM checkpoint:

```bash
python3 plugins/tidal/train_rl.py \
    --config plugins/tidal/configs/base_config.yaml \
    --rl-config plugins/tidal/configs/rl_config.yaml \
    --checkpoint experiments/<experiment_id>/transformer-lm_v1.0.0.pth
```

### Generate Text

Standard generation:

```bash
python3 plugins/tidal/Generator.py \
    --config plugins/tidal/configs/base_config.yaml \
    --checkpoint experiments/<experiment_id>/checkpoint_foundational_epoch_1.pth \
    --prompt "Once upon a time" --max_tokens 50 --temperature 0.8 --top_k 50
```

With RL-controlled gating:

```bash
python3 plugins/tidal/Generator.py \
    --config plugins/tidal/configs/base_config.yaml \
    --checkpoint <model_checkpoint> \
    --rl-agent --rl-checkpoint <rl_checkpoint> \
    --prompt "Once upon a time"
```

### Run Tests

```bash
# Infrastructure tests
python -m unittest discover -s tests -v

# Model tests
python -m unittest discover -s plugins/tidal/tests -v
```

### Start the Dashboard

```bash
export TIDAL_AUTH_TOKEN="your-secret-token-here"
cd dashboard && docker compose up -d && npm run dev
```

### Dashboard Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TIDAL_AUTH_TOKEN` | *(required)* | Bearer token for authenticating mutating API requests |
| `PORT` | `4400` | Fastify server port |
| `HOST` | `0.0.0.0` | Server bind address |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `EXPERIMENTS_DIR` | `./experiments` | Directory for checkpoints and run artifacts |
| `PYTHON_BIN` | `./tidal-env/bin/python` | Python executable used by the worker spawner |
| `DEFAULT_COMPUTE_PROVIDER` | `local` | Compute provider for job orchestration |

## Configuration

Two YAML files in [`plugins/tidal/configs/`](plugins/tidal/configs/) control training:

- [**`base_config.yaml`**](plugins/tidal/configs/base_config.yaml) — Model architecture, data pipeline, and LM training hyperparameters
- [**`rl_config.yaml`**](plugins/tidal/configs/rl_config.yaml) — PPO agent, gating modulator, reward function, and RL training loop

## Architecture Decision Records

Design decisions are documented in [`research/adrs/`](research/adrs/):

| ADR | Decision |
|---|---|
| [0001](research/adrs/0001-single-modulation-gate.md) | Collapse 3-gate system to single modulation gate |
| [0002](research/adrs/0002-uint16-data-cache.md) | Use uint16 for tokenized data cache (4x size reduction) |
| [0003](research/adrs/0003-recalibrate-entropy-homeostasis-and-reward-weights.md) | Recalibrate entropy homeostasis and reward weights for 1D gate |
| [0004](research/adrs/0004-lazy-disk-cache-for-mcp-http-client.md) | Lazy disk cache for MCP HTTP client |
| [0005](research/adrs/0005-diversity-homeostasis.md) | Diversity homeostasis for RL gating agent |
| [0006](research/adrs/0006-ppo-lagrangian-diversity-constraint.md) | PPO-Lagrangian diversity constraint |

## Legacy

The original architecture used an N-body physics simulation in 2D space with a "Semantic Endocrine System" for modulating embeddings. That work has been archived to `research/legacy` along with its [README](research/legacy/README.md).

## License

[GPL-3.0](LICENSE)
