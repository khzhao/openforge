#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GATEWAY_LOG="/tmp/openforge-search-r1-gateway.log"
GATEWAY_DB="/tmp/search-r1-gateway.sqlite3"
: >"$GATEWAY_LOG"
rm -f "$GATEWAY_DB"

cleanup() {
  python -m openforge.cli.main session stop >/dev/null 2>&1 || true
  python -m openforge.cli.main gateway stop >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "Starting gateway"
python -m openforge.cli.main gateway start \
  --config examples/search_r1_gateway.yaml \
  >"$GATEWAY_LOG" 2>&1 &

echo "Starting session"
python -m openforge.cli.main session start \
  --runtime-config examples/search_r1_runtime.yaml

echo "Starting training loop"
PYTHONUNBUFFERED=1 python examples/train_search_r1_ninja.py "$@"
