#!/usr/bin/env bash
# vast-onstart.sh — Paste this into the Vast.ai "on-start" script field.
# Sets up infrastructure so everything is ready when you SSH in.
set -euo pipefail

REPO_DIR="/workspace/tidal-language-model"

# 1. Clone repo (skip if already there, e.g. from a restart)
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/Michelleeby/tidal-language-model.git "$REPO_DIR"
fi

# 2. Start Redis
redis-server \
    --daemonize yes \
    --maxmemory 2048mb \
    --maxmemory-policy allkeys-lfu

# 3. Build dashboard
cd "$REPO_DIR/dashboard"
npm ci --prefer-offline 2>/dev/null || npm install
npm run build

# 4. Start dashboard server (Fastify on :4400)
EXPERIMENTS_DIR="$REPO_DIR/experiments" \
REDIS_URL="redis://localhost:6379" \
    nohup node packages/server/dist/index.js > /var/log/dashboard.log 2>&1 &

# 5. Start TensorBoard on :6006
nohup tensorboard --logdir "$REPO_DIR/experiments" --host 0.0.0.0 --port 6006 \
    > /var/log/tensorboard.log 2>&1 &

echo "Infrastructure ready. SSH in and run training."
