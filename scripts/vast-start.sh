#!/usr/bin/env bash
# vast-start.sh — Run after SSH-ing into the Vast.ai instance.
# The on-start script already set up Redis, dashboard, and TensorBoard.
#
# Usage:
#   cd /workspace/tidal-language-model
#   bash scripts/vast-start.sh --config configs/base_config.yaml
set -euo pipefail

REPO_DIR="/workspace/tidal-language-model"
cd "$REPO_DIR"

# Verify infrastructure is running
echo "Checking services..."
redis-cli ping || { echo "ERROR: Redis not running"; exit 1; }
curl -sf http://localhost:4400/api/status > /dev/null && echo "Dashboard: OK" \
    || echo "WARNING: Dashboard not responding on :4400 (check /var/log/dashboard.log)"
curl -sf http://localhost:6006 > /dev/null && echo "TensorBoard: OK" \
    || echo "WARNING: TensorBoard not responding on :6006 (check /var/log/tensorboard.log)"

# Check config exists
if [ ! -f "${2:-configs/base_config.yaml}" ]; then
    echo "ERROR: Config file not found. Copy it over first:"
    echo "  scp -P <port> configs/base_config.yaml root@<host>:$REPO_DIR/configs/"
    exit 1
fi

echo "Starting training..."
python3 Main.py "$@"
