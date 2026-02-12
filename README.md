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
├── worker_agent.py           # Job worker (spawns training subprocesses)
├── DynamicLRScheduler.py     # Cosine annealing with warmup
├── Utils.py                  # Shared utilities
├── configs/                  # YAML config files (gitignored)
├── experiments/              # Checkpoints and run artifacts
├── tests/                    # Unit tests (unittest)
├── dashboard/                # Fastify + React monitoring dashboard
│   ├── docker-compose.yml    # Redis 7 service
│   └── packages/
│       ├── server/           # Fastify 5 API (port 4400)
│       │   └── src/
│       │       ├── plugins/  # Auth & rate limiting
│       │       ├── services/ # Job orchestrator, store, worker spawner
│       │       └── routes/   # REST + SSE endpoints
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

Config files live in `configs/` (gitignored). Two YAML files control training:

- **`base_config.yaml`** — Model architecture, data pipeline, and LM training hyperparameters
- **`rl_config.yaml`** — PPO agent, gating modulator, reward function, and RL training loop

### Example: `configs/base_config.yaml`

```yaml
# Model Architecture
MODEL_VERSION: "1.0.0"
MODEL_NAME: "transformer-lm"
EMBED_DIM: 256
NUM_TRANSFORMER_BLOCKS: 6
NUM_ATTENTION_HEADS: 8
FFN_HIDDEN_DIM: 1024
DROPOUT: 0.1
MAX_CONTEXT_LENGTH: 256

# Data Pipeline
DATASET: "roneneldan/TinyStories"
TOKENIZER: "gpt2"
VOCAB_SIZE: 50257

# Training Hyperparameters
BATCH_SIZE: 64
DESIRED_BATCH_SIZE: 2048
NUM_EPOCHS: 12
PATIENCE: 5
MIN_DELTA: 0.0001

# Learning Rate Schedule (cosine annealing with warmup)
LEARNING_RATE_SCHEDULER:
  WARMUP_RATIO: 0.1
  BASE_LR: 0.001
  MIN_LR: 1.0e-6

# Evaluation & Logging
EVAL_BATCH_SIZE: 64
VALIDATION_SPLIT_RATIO: 0.1
NUM_CPU_CORE_WORKERS: 4
LOG_DIRECTORY: "logs"
ENABLE_CONSOLE_LOGGING: false
TRAINING_MODEL_ARTIFACT_CACHE_FREQUENCY: 1

# Device & Optimization
DEVICE: "auto"
MAX_GRAD_NORM: 1.0

# TensorBoard
TENSORBOARD_TAGS:
  LOSSES: "Loss"
  LEARNING_RATE: "Learning"
```

### Example: `configs/rl_config.yaml`

```yaml
# RL Agent
RL_AGENT_TYPE: "beta"
RL_OBSERVATION_DIM: 64
RL_ACTION_DIM: 3
RL_HIDDEN_DIM: 128

# PPO Hyperparameters
RL_LEARNING_RATE: 3.0e-4
RL_GAMMA: 0.99
RL_GAE_LAMBDA: 0.95
RL_CLIP_EPSILON: 0.2
RL_ENTROPY_COEF: 0.01
RL_VALUE_COEF: 0.5
RL_MAX_GRAD_NORM: 0.5

# Training Loop
RL_ROLLOUT_STEPS: 128
RL_NUM_EPOCHS: 4
RL_BATCH_SIZE: 32
RL_TOTAL_TIMESTEPS: 100000

# Environment
RL_MAX_EPISODE_LENGTH: 50
RL_PROMPT_MIN_LENGTH: 3
RL_PROMPT_MAX_LENGTH: 10
RL_TOP_K: 40

# Gating Modulator (how signals affect generation)
RL_BASE_TEMPERATURE: 1.0
RL_BASE_REPETITION_PENALTY: 1.2
RL_CREATIVITY_TEMP_MIN: 0.5
RL_CREATIVITY_TEMP_MAX: 1.5
RL_STABILITY_PENALTY_MIN: 1.0
RL_STABILITY_PENALTY_MAX: 2.5
RL_FOCUS_ATTENTION_STRENGTH: 2.0

# Reward Function Weights
RL_REWARD_PERPLEXITY_WEIGHT: 0.4
RL_REWARD_DIVERSITY_WEIGHT: 0.3
RL_REWARD_REPETITION_WEIGHT: 0.2
RL_REWARD_COHERENCE_WEIGHT: 0.1
RL_PERPLEXITY_CLIP: 100.0
RL_ENTROPY_TARGET: 5.0

# Evaluation
RL_EVAL_EPISODES: 50
RL_EVAL_INTERVAL: 1000

# Logging
RL_LOG_INTERVAL: 10
RL_METRICS_INTERVAL: 50
RL_CHECKPOINT_INTERVAL: 100

# TensorBoard
TENSORBOARD_TAGS:
  LOSSES: "Loss"
  LEARNING_RATE: "Learning"
  RL_REWARDS: "RL/Rewards"
  RL_POLICY: "RL/Policy"
  RL_GATING: "RL/Gating"
```

## Legacy

The original architecture used an N-body physics simulation in 2D space with a "Semantic Endocrine System" for modulating embeddings. That work has been archived to `legacy_research/` along with its [README](legacy_research/README.md).

## License

[GPL-3.0](LICENSE)
