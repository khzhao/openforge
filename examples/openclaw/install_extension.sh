#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/extensions/rl-training-headers"
target_dir="${HOME}/.openclaw/extensions/rl-training-headers"

mkdir -p "$(dirname "${target_dir}")"
ln -sfn "${source_dir}" "${target_dir}"

printf 'Linked %s -> %s\n' "${target_dir}" "${source_dir}"
printf 'Next: add the plugin entry from %s to ~/.openclaw/openclaw.json and restart OpenClaw Gateway.\n' \
  "${script_dir}/openclaw.json.example"
