#!/usr/bin/env bash
set -euo pipefail

echo "=== SELD Digital Twin — Dev Environment Setup ==="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "docker is required but not installed."; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "pnpm is required. Install: npm install -g pnpm"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 1. Install TypeScript dependencies
echo "--- Installing TypeScript dependencies ---"
pnpm install

# 2. Install Python dependencies for each service
echo "--- Installing Python dependencies ---"
for service in libs/seld-common edge training cloud/api-gateway cloud/ingestion-service cloud/device-service spatial semantic vision; do
    if [ -f "$service/pyproject.toml" ]; then
        echo "  -> $service"
        (cd "$service" && uv sync --dev)
    fi
done

# 3. Start infrastructure (Docker Compose)
echo "--- Starting infrastructure services ---"
docker compose up -d postgres redis emqx redpanda

# 4. Wait for PostgreSQL
echo "--- Waiting for PostgreSQL ---"
until docker compose exec postgres pg_isready -U seld >/dev/null 2>&1; do
    sleep 1
done

# 5. Run database migrations
echo "--- Running database migrations ---"
(cd migrations && uv run alembic upgrade head)

# 6. Generate protobuf stubs
echo "--- Generating protobuf stubs ---"
bash scripts/generate-protos.sh

echo ""
echo "=== Dev environment ready! ==="
echo "  PostgreSQL: localhost:5432 (seld/seld_dev_password)"
echo "  Redis:      localhost:6379"
echo "  EMQX:       localhost:1883 (MQTT) / localhost:18083 (Dashboard)"
echo "  Redpanda:   localhost:19092 (Kafka API)"
echo ""
echo "Run 'docker compose up' to start all services."
