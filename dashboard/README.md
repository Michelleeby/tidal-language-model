# Dashboard

Fastify + React dashboard for monitoring training experiments.

## Local Development

```bash
cd dashboard
docker compose up -d        # start Redis
npm run dev                 # start Fastify (4400) + Vite (5173)
```

## Remote Server (Vast.ai)

Restart the dashboard on the training instance:

```bash
cd /workspace/tidal-language-model/dashboard

# 1. Clean build artifacts and rebuild
find packages -name '*.tsbuildinfo' -delete
npm run build -w packages/shared
npm run build -w packages/server
npm run build -w packages/client

# 2. Kill the old server and start a new one
kill $(pgrep -f 'index.js')
EXPERIMENTS_DIR=/workspace/tidal-language-model/experiments \
REDIS_URL=redis://localhost:6379 \
PYTHON_BIN=/opt/conda/bin/python \
nohup node packages/server/dist/index.js > /var/log/dashboard.log 2>&1 &
```

Logs: `tail -f /var/log/dashboard.log`
