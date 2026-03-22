#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GATEWAY_LOG="/tmp/openforge-gateway.log"
: >"$GATEWAY_LOG"

cleanup() {
  python -m openforge.cli.main session stop >/dev/null 2>&1 || true
  python -m openforge.cli.main gateway stop >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

python -m openforge.cli.main gateway start \
  --config examples/gsm8k_gateway.yaml \
  >"$GATEWAY_LOG" 2>&1 &

python -m openforge.cli.main session start \
  --runtime-config examples/gsm8k_runtime.yaml

PYTHONUNBUFFERED=1 python examples/train_gsm8k_ninja.py \
  --max-updates 250 \
  "$@"
