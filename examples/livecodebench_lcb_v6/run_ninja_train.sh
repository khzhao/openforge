#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

JOB_DIR="${OPENFORGE_JOB_DIR:-/tmp/openforge-livecodebench-lcb-v6-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_DIR="$JOB_DIR/artifacts"
GATEWAY_LOG="$JOB_DIR/gateway.log"
SESSION_LOG="$JOB_DIR/session.log"
GATEWAY_DB="$JOB_DIR/gateway.sqlite3"
GATEWAY_CONFIG="$JOB_DIR/gateway.yaml"
TRAIN_GROUP_PARALLELISM="${OPENFORGE_TRAIN_GROUP_PARALLELISM:-64}"
VALIDATION_EVERY_UPDATES="${OPENFORGE_VALIDATION_EVERY_UPDATES:-5}"
MAX_VALIDATION_EXAMPLES="${OPENFORGE_MAX_VALIDATION_EXAMPLES:-}"

mkdir -p "$JOB_DIR"

cleanup() {
  openforge session stop >/dev/null 2>&1 || true
  openforge gateway stop >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "JOB_DIR=$JOB_DIR"
echo "GATEWAY_CONFIG=$GATEWAY_CONFIG"
echo "GATEWAY_DB=$GATEWAY_DB"
echo "GATEWAY_LOG=$GATEWAY_LOG"
echo "SESSION_LOG=$SESSION_LOG"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "TRAIN_GROUP_PARALLELISM=$TRAIN_GROUP_PARALLELISM"
echo "VALIDATION_EVERY_UPDATES=$VALIDATION_EVERY_UPDATES"
echo "MAX_VALIDATION_EXAMPLES=$MAX_VALIDATION_EXAMPLES"

python - "$ROOT/examples/livecodebench_lcb_v6/gateway.yaml" "$GATEWAY_CONFIG" "$GATEWAY_DB" <<'PY'
from pathlib import Path
import sys

import yaml

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
db_path = Path(sys.argv[3])

with source_path.open(encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
payload.setdefault("data", {})["path"] = str(db_path)

with target_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY

echo "Starting gateway..."
openforge gateway start \
  --config "$GATEWAY_CONFIG" \
  >"$GATEWAY_LOG" 2>&1 &

echo $! >"$JOB_DIR/gateway.pid"
echo "Gateway process started; logs: $GATEWAY_LOG"

echo "Starting session..."
openforge session start \
  --runtime-config examples/livecodebench_lcb_v6/runtime.yaml \
  >"$SESSION_LOG" 2>&1
echo "Session started; logs: $SESSION_LOG"

echo "Starting training..."
train_cmd=(
  python -m examples.livecodebench_lcb_v6.train_ninja
  --artifact-dir "$ARTIFACT_DIR" \
  --train-group-parallelism "$TRAIN_GROUP_PARALLELISM" \
  --validation-every-updates "$VALIDATION_EVERY_UPDATES" \
)

if [[ -n "$MAX_VALIDATION_EXAMPLES" ]]; then
  train_cmd+=(--max-validation-examples "$MAX_VALIDATION_EXAMPLES")
fi

PYTHONUNBUFFERED=1 "${train_cmd[@]}" "$@"
