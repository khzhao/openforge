#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-/tmp/openclaw-demo-$(date +%Y%m%d-%H%M%S)}"
OPENFORGE_GATEWAY_PORT="${OPENFORGE_GATEWAY_PORT:-8042}"
OPENCLAW_MIDDLEWARE_PORT="${OPENCLAW_MIDDLEWARE_PORT:-8012}"
CLEANUP_TIMEOUT_SECONDS="${CLEANUP_TIMEOUT_SECONDS:-10}"
MIDDLEWARE_HEALTH_TIMEOUT_SECONDS="${MIDDLEWARE_HEALTH_TIMEOUT_SECONDS:-30}"
OPENCLAW_STATE_DB="${OPENCLAW_STATE_DB:-$EXPERIMENT_DIR/openclaw-state.sqlite3}"
OPENCLAW_LOG_DIR="${OPENCLAW_LOG_DIR:-$EXPERIMENT_DIR}"
OPENFORGE_GATEWAY_LOG="${OPENFORGE_GATEWAY_LOG:-$OPENCLAW_LOG_DIR/openforge-gateway.log}"
OPENFORGE_SESSION_LOG="${OPENFORGE_SESSION_LOG:-$OPENCLAW_LOG_DIR/openforge-session.log}"
OPENCLAW_MIDDLEWARE_LOG="${OPENCLAW_MIDDLEWARE_LOG:-$OPENCLAW_LOG_DIR/openclaw-middleware.log}"
GATEWAY_DB_PATH="${GATEWAY_DB_PATH:-$EXPERIMENT_DIR/openforge-gateway.sqlite3}"
GATEWAY_CONFIG_PATH="${GATEWAY_CONFIG_PATH:-$EXPERIMENT_DIR/gateway.yaml}"
OPENFORGE_RUNTIME_CONFIG="${OPENFORGE_RUNTIME_CONFIG:-examples/openclaw/runtime.yaml}"

cleanup_files() {
  rm -f "$OPENCLAW_STATE_DB" "${OPENCLAW_STATE_DB}-shm" "${OPENCLAW_STATE_DB}-wal"
  rm -f "$GATEWAY_DB_PATH" "${GATEWAY_DB_PATH}-shm" "${GATEWAY_DB_PATH}-wal"
  rm -f "$GATEWAY_CONFIG_PATH"
}

cleanup_processes() {
  timeout "$CLEANUP_TIMEOUT_SECONDS" openforge session stop >/dev/null 2>&1 || true
  timeout "$CLEANUP_TIMEOUT_SECONDS" openforge gateway stop >/dev/null 2>&1 || true
  pkill -f "python -m examples.openclaw.app" >/dev/null 2>&1 || true
}

wait_for_middleware() {
  local deadline=$((SECONDS + MIDDLEWARE_HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${OPENCLAW_MIDDLEWARE_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

trap cleanup_processes INT TERM

echo "Cleaning previous state..."
cleanup_processes
cleanup_files
mkdir -p "$EXPERIMENT_DIR" "$OPENCLAW_LOG_DIR"

python - <<'PY' "$ROOT/examples/openclaw/gateway.yaml" "$GATEWAY_CONFIG_PATH" "$GATEWAY_DB_PATH" "$OPENFORGE_GATEWAY_PORT"
from pathlib import Path
import sys
import yaml

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
db_path = Path(sys.argv[3])
gateway_port = int(sys.argv[4])

with source_path.open(encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
payload.setdefault("data", {})["path"] = str(db_path)
payload.setdefault("gateway", {})["port"] = gateway_port

with target_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY

echo "Experiment:"
echo "  Directory:            $EXPERIMENT_DIR"
echo "  Middleware state DB:  $OPENCLAW_STATE_DB"
echo "  OpenForge gateway DB: $GATEWAY_DB_PATH"
echo "  Gateway config:       $GATEWAY_CONFIG_PATH"
echo "  Runtime config:       $OPENFORGE_RUNTIME_CONFIG"
echo "Logs:"
echo "  OpenForge gateway:   $OPENFORGE_GATEWAY_LOG"
echo "  OpenForge session:   $OPENFORGE_SESSION_LOG"
echo "  Middleware:          $OPENCLAW_MIDDLEWARE_LOG"

echo "Starting OpenForge gateway..."
setsid -f openforge gateway start --config "$GATEWAY_CONFIG_PATH" \
  >"$OPENFORGE_GATEWAY_LOG" 2>&1 < /dev/null

echo "Starting OpenForge session..."
openforge session start --runtime-config "$OPENFORGE_RUNTIME_CONFIG" \
  >"$OPENFORGE_SESSION_LOG" 2>&1

echo "Starting OpenClaw middleware on port ${OPENCLAW_MIDDLEWARE_PORT}..."
setsid -f env OPENCLAW_MIDDLEWARE_PORT="${OPENCLAW_MIDDLEWARE_PORT}" \
OPENFORGE_GATEWAY_BASE_URL="http://127.0.0.1:${OPENFORGE_GATEWAY_PORT}" \
OPENCLAW_STATE_DB="${OPENCLAW_STATE_DB}" \
OPENFORGE_GATEWAY_DB_PATH="${GATEWAY_DB_PATH}" \
python -u -m examples.openclaw.app >"$OPENCLAW_MIDDLEWARE_LOG" 2>&1 < /dev/null

if ! wait_for_middleware; then
  echo "Middleware failed to become healthy within ${MIDDLEWARE_HEALTH_TIMEOUT_SECONDS}s." >&2
  echo "Last middleware log lines:" >&2
  tail -n 80 "$OPENCLAW_MIDDLEWARE_LOG" >&2 || true
  exit 1
fi

echo "Services started."
echo "OpenClaw TUI is not launched automatically."
echo "If needed, run:"
echo "  python -m examples.openclaw.tui --state-db \"$OPENCLAW_STATE_DB\" --gateway-db \"$GATEWAY_DB_PATH\""
