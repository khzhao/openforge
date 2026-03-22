#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/openforge-gsm8k-full-256-$(date +%s)}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_NODE="${CPUS_PER_NODE:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,6}"
RAY_ADDRESS="${RAY_ADDRESS:-local}"
BASE_RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT/examples/gsm8k_runtime.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
MAX_UPDATES="${MAX_UPDATES:-}"

mkdir -p "$ARTIFACT_DIR"

GATEWAY_CONFIG="$ARTIFACT_DIR/gateway_config.yaml"
GATEWAY_DB="$ARTIFACT_DIR/gateway.sqlite3"
GATEWAY_LOG="$ARTIFACT_DIR/gateway.log"
RUNTIME_CONFIG="$ARTIFACT_DIR/runtime_config.yaml"
TRAIN_ARTIFACT_DIR="$ARTIFACT_DIR/train"

cat > "$GATEWAY_CONFIG" <<EOF
data:
  path: $GATEWAY_DB

gateway:
  host: $GATEWAY_HOST
  port: $GATEWAY_PORT

cluster:
  num_nodes: 1
  gpus_per_node: $GPUS_PER_NODE
  cpus_per_node: $CPUS_PER_NODE
EOF

gateway_pid=""

cleanup() {
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill -TERM "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "artifact_dir=$ARTIFACT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "base_runtime_config=$BASE_RUNTIME_CONFIG"
echo "runtime_config=$RUNTIME_CONFIG"
echo "gateway_config=$GATEWAY_CONFIG"
echo "model_path=${MODEL_PATH:-<default>}"
echo "max_new_tokens=${MAX_NEW_TOKENS:-<default>}"
echo "ppo_epochs=${PPO_EPOCHS:-<default>}"
echo "total_epochs=$TOTAL_EPOCHS"
echo "max_updates=${MAX_UPDATES:-<none>}"

env \
  BASE_RUNTIME_CONFIG="$BASE_RUNTIME_CONFIG" \
  EFFECTIVE_RUNTIME_CONFIG="$RUNTIME_CONFIG" \
  MODEL_PATH="$MODEL_PATH" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  PPO_EPOCHS="$PPO_EPOCHS" \
  python - <<'PY'
from pathlib import Path
import os

import yaml

base_path = Path(os.environ["BASE_RUNTIME_CONFIG"])
effective_path = Path(os.environ["EFFECTIVE_RUNTIME_CONFIG"])
runtime = yaml.safe_load(base_path.read_text())

model_path = os.environ.get("MODEL_PATH")
if model_path:
    runtime["model"]["model_name_or_path"] = model_path
    runtime["model"]["tokenizer_name_or_path"] = model_path
    runtime["model"]["reference_model_name_or_path"] = model_path

max_new_tokens = os.environ.get("MAX_NEW_TOKENS")
if max_new_tokens:
    runtime["rollout"]["request"]["max_new_tokens"] = int(max_new_tokens)

ppo_epochs = os.environ.get("PPO_EPOCHS")
if ppo_epochs:
    runtime["train"]["ppo_epochs"] = int(ppo_epochs)

effective_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
PY

env \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  RAY_ADDRESS="$RAY_ADDRESS" \
  python -m openforge.gateway.main \
    --config "$GATEWAY_CONFIG" \
    --port "$GATEWAY_PORT" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --cpus-per-node "$CPUS_PER_NODE" \
    > "$GATEWAY_LOG" 2>&1 &
gateway_pid="$!"

python - <<PY
import sys
import time

import httpx

gateway_pid = int("$gateway_pid")
url = "http://$GATEWAY_HOST:$GATEWAY_PORT/health"
deadline = time.monotonic() + 180.0
last_error = None

while time.monotonic() < deadline:
    try:
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            raise SystemExit(0)
        last_error = f"status={response.status_code}"
    except Exception as exc:  # noqa: BLE001
        last_error = repr(exc)
    try:
        import os
        os.kill(gateway_pid, 0)
    except OSError:
        raise SystemExit(
            f"gateway exited before health check passed; see $GATEWAY_LOG"
        )
    time.sleep(1.0)

raise SystemExit(f"timed out waiting for {url}: {last_error}")
PY

python - <<PY
from pathlib import Path

import httpx
import yaml

runtime = yaml.safe_load(Path("$RUNTIME_CONFIG").read_text())
response = httpx.post(
    "http://$GATEWAY_HOST:$GATEWAY_PORT/start_session",
    json={"runtime": runtime},
    timeout=1800.0,
)
response.raise_for_status()
print(response.json())
PY

train_args=(
  examples/train_gsm8k_ninja.py
  --artifact-dir "$TRAIN_ARTIFACT_DIR"
  --gateway-config "$GATEWAY_CONFIG"
  --runtime-config "$RUNTIME_CONFIG"
  --total-epochs "$TOTAL_EPOCHS"
)

if [[ -n "$MAX_UPDATES" ]]; then
  train_args+=(--max-updates "$MAX_UPDATES")
fi

if [[ "$#" -gt 0 ]]; then
  train_args+=("$@")
fi

env PYTHONPATH=src PYTHONUNBUFFERED=1 python "${train_args[@]}"
