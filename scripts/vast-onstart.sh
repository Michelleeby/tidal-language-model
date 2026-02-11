#!/usr/bin/env bash
# vast-onstart.sh — Paste this into the Vast.ai "on-start" script field.
# Sets up infrastructure so everything is ready when you SSH in.
set -euo pipefail

REPO_DIR="/workspace/tidal-language-model"

# 1. Clone repo (skip if already there, e.g. from a restart)
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/Michelleeby/tidal-language-model.git "$REPO_DIR"
fi

# 2. torch.compile deps: gcc + libcuda.so linker symlink
apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
ensure_libcuda_symlink() {
    local real
    real=$(find / -name 'libcuda.so.*' -not -name 'libcuda.so' -type f 2>/dev/null | head -1)
    if [ -z "$real" ]; then
        echo "WARNING: libcuda.so not found — torch.compile may fail"
        return
    fi
    local dir
    dir=$(dirname "$real")
    if [ ! -e "$dir/libcuda.so" ]; then
        ln -s "$real" "$dir/libcuda.so"
        echo "Created linker symlink: $dir/libcuda.so -> $real"
    fi
}
ensure_libcuda_symlink

# 3. Start Redis
redis-server \
    --daemonize yes \
    --maxmemory 2048mb \
    --maxmemory-policy allkeys-lfu

# 4. Build dashboard (shared must be built before client)
cd "$REPO_DIR/dashboard"
find packages -name '*.tsbuildinfo' -delete
npm ci --prefer-offline 2>/dev/null || npm install
npm run build -w packages/shared
npm run build -w packages/server
npm run build -w packages/client

# 5. Start dashboard server (Fastify on :4400)
EXPERIMENTS_DIR="$REPO_DIR/experiments" \
REDIS_URL="redis://localhost:6379" \
PYTHON_BIN="$(which python3)" \
    nohup node packages/server/dist/index.js > /var/log/dashboard.log 2>&1 &

# 6. Start TensorBoard on :6006
nohup tensorboard --logdir "$REPO_DIR/experiments" --host 0.0.0.0 --port 6006 \
    > /var/log/tensorboard.log 2>&1 &

echo "Infrastructure ready. SSH in and run training."
