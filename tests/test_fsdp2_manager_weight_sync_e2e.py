# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

from _manager_weight_sync_common import DEFAULT_MODEL, SYNC_MODES, run_weight_sync_e2e


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the manager weight-sync e2e script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=2)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    parser.add_argument("--base-policy-version", type=int, default=500)
    parser.add_argument(
        "--sync-modes",
        default=",".join(SYNC_MODES),
        help="Comma-separated subset of distributed",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested manager weight-sync modes against a real model."""
    args = parse_args()
    sync_modes = tuple(
        mode.strip() for mode in args.sync_modes.split(",") if mode.strip()
    )
    return run_weight_sync_e2e(
        model_path=args.model_path,
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
        base_policy_version=args.base_policy_version,
        sync_modes=sync_modes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
