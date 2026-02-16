#!/usr/bin/env bash
# rollout.sh run to git pull and then docker build/deploy updated dashboard image.
#
# Usage:
#
# cd /path/to/project/root && scripts/rollout.sh /path/to/working/dir
set -euo pipefail

cd $1
git pull
docker compose -f docker-compose.prod.yml up -d --build
echo "Done!"
