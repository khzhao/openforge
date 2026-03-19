# Copyright 2026 openforge

from __future__ import annotations

import argparse

import ray
import torch
import torch.distributed as dist

from _sglang_engine_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    assert_weights_changed,
    configure_nccl_env,
    find_free_port,
    init_trainer_update_group,
    load_source_model,
    require_visible_gpus,
    resolve_local_model_path,
    select_update_tensor,
    shutdown_single_engine,
    snapshot_weights,
    start_single_engine,
    torch_dtype_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=301)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_nccl_env()
    require_visible_gpus(2)
    model_path = resolve_local_model_path(args.model_path)
    source_device = torch.device("cuda", 0)
    group_name = "openforge_weight_update_test"
    master_port = find_free_port()

    source_model = load_source_model(model_path, device=source_device)
    worker, _cfg, placement_groups = start_single_engine(model_path=model_path)
    process_group = None
    try:
        name, updated_tensor = select_update_tensor(source_model)
        snapshot_weights(worker)
        assert_success(
            ray.get(worker.pause_generation.remote(mode="abort")),
            context="pause_generation",
        )

        ray.get(
            worker.begin_init_weights_update_group.remote(
                master_address="127.0.0.1",
                master_port=master_port,
                rank_offset=1,
                world_size=2,
                group_name=group_name,
                backend="nccl",
            )
        )
        process_group = init_trainer_update_group(
            master_port=master_port,
            group_name=group_name,
            device=source_device,
        )
        assert_success(
            ray.get(worker.wait_pending_runtime_call.remote()),
            context="init_weights_update_group",
        )

        ray.get(
            worker.begin_update_weights_from_distributed.remote(
                names=[name],
                dtypes=[torch_dtype_name(updated_tensor.dtype)],
                shapes=[list(updated_tensor.shape)],
                group_name=group_name,
                policy_version=args.policy_version,
                flush_cache=True,
            )
        )
        dist.broadcast(updated_tensor, src=0, group=process_group)
        assert_success(
            ray.get(worker.wait_pending_runtime_call.remote()),
            context="update_weights_from_distributed",
        )

        assert_success(
            ray.get(worker.destroy_weights_update_group.remote(group_name=group_name)),
            context="destroy_weights_update_group",
        )
        dist.destroy_process_group(process_group)
        process_group = None

        assert_weight_version(worker, expected_version=args.policy_version)
        assert_weights_changed(worker, context="distributed/compare")
        assert_success(
            ray.get(worker.continue_generation.remote()),
            context="continue_generation",
        )
        print(f"[distributed] SUCCESS updated {name}")
        return 0
    finally:
        if process_group is not None:
            try:
                dist.destroy_process_group(process_group)
            except Exception:
                pass
            try:
                ray.get(
                    worker.destroy_weights_update_group.remote(group_name=group_name)
                )
            except Exception:
                pass
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutdown_single_engine(worker, placement_groups)


if __name__ == "__main__":
    raise SystemExit(main())
