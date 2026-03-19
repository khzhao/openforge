# Copyright 2026 openforge

from __future__ import annotations

import argparse

import ray
import torch

from _sglang_engine_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    assert_weights_changed,
    configure_nccl_env,
    load_source_model,
    require_visible_gpus,
    resolve_local_model_path,
    select_update_tensor,
    serialize_flattened_bucket,
    shutdown_single_engine,
    snapshot_weights,
    start_single_engine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=201)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_nccl_env()
    require_visible_gpus(2)
    model_path = resolve_local_model_path(args.model_path)
    source_model = load_source_model(model_path, device=torch.device("cuda", 0))
    worker, _cfg, placement_groups = start_single_engine(model_path=model_path)
    try:
        name, updated_tensor = select_update_tensor(source_model)
        serialized = serialize_flattened_bucket([(name, updated_tensor)])
        snapshot_weights(worker)
        assert_success(
            ray.get(worker.pause_generation.remote(mode="abort")),
            context="pause_generation",
        )
        assert_success(
            ray.get(
                worker.update_weights_from_tensor.remote(
                    serialized_named_tensors=[serialized],
                    policy_version=args.policy_version,
                    load_format="flattened_bucket",
                    flush_cache=True,
                )
            ),
            context="update_weights_from_tensor",
        )
        assert_weight_version(worker, expected_version=args.policy_version)
        assert_weights_changed(worker, context="tensor/compare")
        assert_success(
            ray.get(worker.continue_generation.remote()),
            context="continue_generation",
        )
        print(f"[tensor] SUCCESS updated {name}")
        return 0
    finally:
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutdown_single_engine(worker, placement_groups)


if __name__ == "__main__":
    raise SystemExit(main())
