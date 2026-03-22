#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/openforge-gsm8k-$(date +%s)}"
GATEWAY_CONFIG="${GATEWAY_CONFIG:-$ROOT/examples/gsm8k_gateway.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,6}"
RAY_ADDRESS="${RAY_ADDRESS:-local}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT/examples/gsm8k_runtime.yaml}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
MAX_UPDATES="${MAX_UPDATES:-}"

export CUDA_VISIBLE_DEVICES
export RAY_ADDRESS

mkdir -p "$ARTIFACT_DIR"
GATEWAY_LOG="$ARTIFACT_DIR/gateway.log"

cleanup() {
  uv run openforge session stop >/dev/null 2>&1 || true
  uv run openforge gateway stop >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

uv run openforge gateway start \
  --config "$GATEWAY_CONFIG" > "$GATEWAY_LOG" 2>&1 &

uv run openforge session start \
  --runtime-config "$RUNTIME_CONFIG"

if [[ -n "$MAX_UPDATES" ]]; then
  PYTHONUNBUFFERED=1 uv run python examples/train_gsm8k_ninja.py \
    --artifact-dir "$ARTIFACT_DIR/train" \
    --total-epochs "$TOTAL_EPOCHS" \
    --max-updates "$MAX_UPDATES" \
    "$@"
else
  PYTHONUNBUFFERED=1 uv run python examples/train_gsm8k_ninja.py \
    --artifact-dir "$ARTIFACT_DIR/train" \
    --total-epochs "$TOTAL_EPOCHS" \
    "$@"
fi
