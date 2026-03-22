# Copyright 2026 openforge

from __future__ import annotations

import argparse

import ray

from _sglang_engine_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    configure_nccl_env,
    require_visible_gpus,
    resolve_local_model_path,
    shutdown_single_engine,
    start_single_engine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_nccl_env()
    require_visible_gpus(2)
    model_path = resolve_local_model_path(args.model_path)
    worker, _cfg, placement_groups = start_single_engine(model_path=model_path)
    try:
        assert_success(
            ray.get(worker.pause_generation.remote(mode="abort")),
            context="pause_generation",
        )
        assert_success(
            ray.get(
                worker.update_weights_from_disk.remote(
                    model_path=model_path,
                    policy_version=args.policy_version,
                )
            ),
            context="update_weights_from_disk",
        )
        assert_weight_version(worker, expected_version=args.policy_version)
        assert_success(
            ray.get(worker.continue_generation.remote()),
            context="continue_generation",
        )
        print("[disk] SUCCESS")
        return 0
    finally:
        shutdown_single_engine(worker, placement_groups)


if __name__ == "__main__":
    raise SystemExit(main())
