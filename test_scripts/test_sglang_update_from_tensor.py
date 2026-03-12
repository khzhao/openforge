# Copyright 2026 openforge

from __future__ import annotations

import argparse
import torch

from _sglang_weight_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    load_source_model,
    require_visible_gpus,
    resolve_local_model_path,
    select_update_tensor,
    serialize_flattened_bucket,
    start_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=201)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_visible_gpus(2)
    model_path = resolve_local_model_path(args.model_path)
    source_device = torch.device("cuda", 0)

    runtime, _cfg = start_runtime(
        model_path=model_path,
        name="sglang-tensor-update-test",
        train_total_gpus=1,
    )
    source_model = load_source_model(model_path, device=source_device)
    try:
        name, updated_tensor = select_update_tensor(source_model)
        serialized = serialize_flattened_bucket([(name, updated_tensor)])

        assert_success(
            runtime.pause_generation(mode="abort"),
            context="pause_generation",
        )
        assert_success(
            runtime.update_weights_from_tensor(
                serialized_named_tensors=[serialized],
                load_format="flattened_bucket",
                weight_version=str(args.policy_version),
                flush_cache=True,
            ),
            context="update_weights_from_tensor",
        )
        assert_weight_version(runtime, expected_version=args.policy_version)
        assert_success(
            runtime.continue_generation(),
            context="continue_generation",
        )
        print(f"[tensor] SUCCESS updated {name}")
        return 0
    finally:
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())
