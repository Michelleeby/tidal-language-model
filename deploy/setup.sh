#!/usr/bin/env bash
set -euo pipefail

# ── Tidal Dashboard — DigitalOcean Droplet Setup ──
# Run this after SSH-ing into a fresh Ubuntu 24.04 droplet.
# Usage: bash setup.sh [git-repo-url]

REPO="${1:-https://github.com/your-user/tidal-language-model.git}"
INSTALL_DIR="/opt/tidal"

echo "==> Installing Docker"
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
fi

echo "==> Configuring firewall"
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> Cloning repository"
git clone "$REPO" "$INSTALL_DIR"

echo "==> Creating .env"
AUTH_TOKEN=$(openssl rand -hex 32)
cat > "$INSTALL_DIR/dashboard/.env" <<EOF
TIDAL_AUTH_TOKEN=$AUTH_TOKEN
EOF

echo "==> Building and starting services"
cd "$INSTALL_DIR/dashboard"
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "════════════════════════════════════════════"
echo "  Dashboard is running."
echo ""
echo "  Auth token: $AUTH_TOKEN"
echo ""
echo "  Save the auth token — it won't be shown again."
echo "  (It's also in $INSTALL_DIR/dashboard/.env)"
echo ""
echo "  Useful commands:"
echo "    cd $INSTALL_DIR/dashboard"
echo "    docker compose -f docker-compose.prod.yml logs -f"
echo "    docker compose -f docker-compose.prod.yml restart"
echo "    docker compose -f docker-compose.prod.yml up -d --build  # rebuild after git pull"
echo "════════════════════════════════════════════"
