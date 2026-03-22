#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cleanup() {
  uv run openforge session stop >/dev/null 2>&1 || true
  uv run openforge gateway stop >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

uv run openforge gateway start \
  --config examples/gsm8k_gateway.yaml &

uv run openforge session start \
  --runtime-config examples/gsm8k_runtime.yaml

PYTHONUNBUFFERED=1 uv run python examples/train_gsm8k_ninja.py \
  --max-updates 250 \
  "$@"
