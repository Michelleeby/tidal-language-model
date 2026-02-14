# Tidal Dashboard

Scientific notebook dashboard for monitoring, analyzing, and exporting training experiments. Built with Fastify 5 + React 19 + Redis 7.

## Quick Start (Local)

```bash
# Prerequisites: Node >= 20, Docker

cd dashboard
npm install                     # install all workspace dependencies
docker compose up -d            # start Redis (127.0.0.1:6379)
npm run dev                     # start Fastify (4400) + Vite (5173)
```

Open http://localhost:5173. The Vite dev server proxies `/api` requests to the Fastify server on port 4400.

## Architecture

```
dashboard/
├── docker-compose.yml              # Redis 7 (alpine)
├── redis/redis.conf                # 2GB maxmemory, allkeys-lfu
└── packages/
    ├── shared/                     # TypeScript types shared between server + client
    ├── server/                     # Fastify 5 API server (port 4400)
    │   └── src/
    │       ├── routes/             # REST endpoints + SSE streams
    │       ├── services/           # PluginRegistry, JobOrchestrator, etc.
    │       └── config.ts           # Environment variable config
    └── client/                     # React 19 + Vite 6 + Tailwind CSS 4
        └── src/
            ├── components/
            │   ├── notebook/       # CollapsibleSection, ExperimentSidebar
            │   ├── charts/         # uPlot charts + ChartExportButton
            │   ├── config/         # ConfigViewer (YAML syntax highlighting)
            │   └── generation/     # GenerationPanel, GenerationComparison
            ├── hooks/              # React Query hooks (metrics, configs, reports)
            ├── stores/             # Zustand (experiment selection, sidebar state)
            └── utils/              # Report generators (HTML, Markdown)
```

**Data flow**: Python training scripts → `MetricsLogger` → Redis (real-time) + JSONL (archival) → Fastify API → SSE → React charts

## UI Overview

The dashboard is organized as a scientific notebook. An experiment sidebar on the left lists all experiments with status indicators. The main area shows collapsible sections:

- **Overview** — Training status card, metric cards, epoch progress
- **Monitor** — Loss curves (with EMA smoothing), learning rate, perplexity, throughput, RL reward/loss/episode charts
- **Configuration** — Read-only YAML config viewer with syntax highlighting
- **Checkpoints** — Browse and manage model checkpoints
- **Generation** — Embedded playground with single and side-by-side comparison modes
- **Logs** — Live training log viewer
- **Samples** — Evaluation sample outputs

Every chart has a CSV/JSON export button. Full experiment reports can be exported as self-contained HTML or Markdown.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start everything (shared build + server + client in parallel) |
| `npm run build` | Production build (all workspaces) |
| `npm test` | Run all tests (server uses `node:test`) |
| `npm run server:dev` | Start Fastify server only (port 4400) |
| `npm run client:dev` | Start Vite dev server only (port 5173) |
| `npm run redis:up` | Start Redis via Docker Compose |
| `npm run redis:down` | Stop Redis |

## Testing Locally

### 1. Start Redis + Dashboard

```bash
cd dashboard
docker compose up -d
npm run dev
```

### 2. Run a Local Training Job

In a separate terminal, start a training run. The dashboard will pick up metrics in real-time via Redis:

```bash
source tidal-env/bin/activate
python3 plugins/tidal/Main.py --config plugins/tidal/configs/base_config.yaml
```

### 3. Run Server Tests

```bash
cd dashboard
npm test -w packages/server      # 133 tests across 9 test files
```

## Remote Training with Vast.ai

The dashboard can dispatch training jobs to remote GPUs on Vast.ai. To test this locally (your laptop runs the dashboard, Vast.ai runs the training):

### 1. Get a Vast.ai API Key

Sign up at https://vast.ai, go to Account → API Keys, and copy your key.

### 2. Start the Dashboard with Remote Config

```bash
cd dashboard
docker compose up -d

VASTAI_API_KEY=your-vastai-api-key-here \
DEFAULT_COMPUTE_PROVIDER=vastai \
TIDAL_AUTH_TOKEN=pick-a-secret-token \
TIDAL_REPO_URL=https://github.com/your-username/tidal-language-model.git \
TIDAL_DASHBOARD_URL=http://your-public-ip:4400 \
npm run dev
```

The worker on the remote GPU will call back to `TIDAL_DASHBOARD_URL` to report status, so this must be reachable from the internet. If you're behind NAT, use a tunnel (e.g. `ngrok http 4400`) and set `TIDAL_DASHBOARD_URL` to the tunnel URL.

### 3. Start a Job from the UI

Open http://localhost:5173, select an experiment, and click the training trigger in the Monitor section. The dashboard will provision a Vast.ai GPU instance, clone the repo, and start training. Metrics stream back in real-time via SSE.

### 4. Verify the Remote Worker

The remote instance reports heartbeats back to the dashboard. You can check job status in the Overview section's status card. Logs from the remote worker appear in the Logs section.

## Remote Training with DigitalOcean

Same pattern as Vast.ai, but with DigitalOcean GPU droplets:

```bash
DO_API_KEY=your-digitalocean-api-key \
DO_REGION=tor1 \
DO_SSH_KEY=your-ssh-key-fingerprint \
DEFAULT_COMPUTE_PROVIDER=digitalocean \
TIDAL_AUTH_TOKEN=pick-a-secret-token \
TIDAL_REPO_URL=https://github.com/your-username/tidal-language-model.git \
TIDAL_DASHBOARD_URL=http://your-public-ip:4400 \
npm run dev
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `4400` | Fastify server port |
| `HOST` | `0.0.0.0` | Fastify bind address |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `EXPERIMENTS_DIR` | `../experiments` | Path to experiments directory |
| `PYTHON_BIN` | `../tidal-env/bin/python` | Python binary for local jobs |
| `DEFAULT_COMPUTE_PROVIDER` | `local` | `local`, `vastai`, `digitalocean`, or `aws` |
| `TIDAL_AUTH_TOKEN` | — | Shared secret for worker ↔ dashboard auth |
| `VASTAI_API_KEY` | — | Vast.ai API key (required when provider is `vastai`) |
| `DO_API_KEY` | — | DigitalOcean API key (required when provider is `digitalocean`) |
| `DO_REGION` | `tor1` | DigitalOcean region |
| `DO_SSH_KEY` | — | SSH key fingerprint for DigitalOcean droplets |
| `TIDAL_REPO_URL` | — | Git repo URL for remote workers to clone |
| `TIDAL_DASHBOARD_URL` | — | Public URL of the dashboard (for remote worker callbacks) |
| `INFERENCE_URL` | — | External inference server URL (optional) |

## Remote Server (Vast.ai Instance)

When running the dashboard directly on a Vast.ai training instance:

```bash
cd /workspace/tidal-language-model/dashboard

# 1. Clean build artifacts and rebuild
find packages -name '*.tsbuildinfo' -delete
npm run build -w packages/shared
npm run build -w packages/server
npm run build -w packages/client

# 2. Kill the old server and start a new one
kill $(pgrep -f 'index.js')
TIDAL_AUTH_TOKEN=your-secret-token-here \
EXPERIMENTS_DIR=/workspace/tidal-language-model/experiments \
REDIS_URL=redis://localhost:6379 \
PYTHON_BIN=/opt/conda/bin/python \
nohup node packages/server/dist/index.js > /var/log/dashboard.log 2>&1 &
```

Logs: `tail -f /var/log/dashboard.log`

## Production Build

```bash
cd dashboard
npm run build
# Server: packages/server/dist/
# Client: packages/client/dist/ (static files served by Fastify)
```
