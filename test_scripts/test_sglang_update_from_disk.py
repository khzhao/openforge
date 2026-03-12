# Copyright 2026 openforge

from __future__ import annotations

import argparse

from _sglang_weight_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    require_visible_gpus,
    resolve_local_model_path,
    start_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_visible_gpus(1)
    model_path = resolve_local_model_path(args.model_path)

    runtime, _cfg = start_runtime(
        model_path=model_path,
        name="sglang-disk-update-test",
        colocated=True,
        train_total_gpus=1,
    )
    try:
        assert_success(
            runtime.pause_generation(mode="abort"),
            context="pause_generation",
        )
        assert_success(
            runtime.update_weights_from_disk(
                model_path=model_path,
                weight_version=str(args.policy_version),
                keep_pause=True,
            ),
            context="update_weights_from_disk",
        )
        assert_weight_version(runtime, expected_version=args.policy_version)
        assert_success(
            runtime.continue_generation(),
            context="continue_generation",
        )
        print("[disk] SUCCESS")
        return 0
    finally:
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())
