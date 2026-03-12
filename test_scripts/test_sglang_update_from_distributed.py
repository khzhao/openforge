# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os
import threading

import torch
import torch.distributed as dist

from _sglang_weight_update_common import (
    DEFAULT_MODEL,
    assert_success,
    assert_weight_version,
    init_trainer_update_group,
    load_source_model,
    require_visible_gpus,
    resolve_local_model_path,
    select_update_tensor,
    start_runtime,
    torch_dtype_name,
    find_free_port,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=301)
    return parser.parse_args()


def run_runtime_call_during_local_step(runtime_call, local_step):
    result: dict[str, object] = {}
    request_error: list[BaseException] = []

    def worker() -> None:
        try:
            result["value"] = runtime_call()
        except BaseException as exc:
            request_error.append(exc)

    # The HTTP handler only returns after it finishes the paired NCCL op.
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    local_result = None
    local_error: BaseException | None = None
    try:
        local_result = local_step()
    except BaseException as exc:
        local_error = exc

    thread.join(timeout=35.0)
    if thread.is_alive():
        raise RuntimeError(
            "SGLang request did not finish. The matching NCCL collective likely "
            "never completed."
        )
    if request_error:
        raise request_error[0]
    if local_error is not None:
        raise local_error
    return result["value"], local_result


def destroy_update_group(runtime, process_group, *, group_name: str) -> None:
    if process_group is not None:
        dist.destroy_process_group(process_group)
    assert_success(
        runtime.destroy_weights_update_group(group_name=group_name),
        context="destroy_weights_update_group",
    )


def main() -> int:
    args = parse_args()
    require_visible_gpus(2)
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    model_path = resolve_local_model_path(args.model_path)
    source_device = torch.device("cuda", 0)
    group_name = "openforge_weight_update_test"
    master_port = find_free_port()

    runtime, _cfg = start_runtime(
        model_path=model_path,
        name="sglang-distributed-update-test",
        train_total_gpus=1,
    )
    source_model = load_source_model(model_path, device=source_device)
    process_group = None
    try:
        name, updated_tensor = select_update_tensor(source_model)

        assert_success(
            runtime.pause_generation(mode="abort"),
            context="pause_generation",
        )
        init_response, process_group = run_runtime_call_during_local_step(
            lambda: runtime.init_weights_update_group(
                "127.0.0.1",
                master_port,
                1,
                2,
                group_name,
                backend="nccl",
            ),
            lambda: init_trainer_update_group(
                master_port=master_port,
                group_name=group_name,
                device=source_device,
            ),
        )
        assert_success(
            init_response,
            context="init_weights_update_group",
        )

        update_response, _ = run_runtime_call_during_local_step(
            lambda: runtime.update_weights_from_distributed(
                names=[name],
                dtypes=[torch_dtype_name(updated_tensor.dtype)],
                shapes=[list(updated_tensor.shape)],
                group_name=group_name,
                flush_cache=True,
                weight_version=str(args.policy_version),
            ),
            lambda: dist.broadcast(
                updated_tensor,
                src=0,
                group=process_group,
            ),
        )
        assert_success(
            update_response,
            context="update_weights_from_distributed",
        )
        destroy_update_group(runtime, process_group, group_name=group_name)
        process_group = None

        assert_weight_version(runtime, expected_version=args.policy_version)
        assert_success(
            runtime.continue_generation(),
            context="continue_generation",
        )
        print(f"[distributed] SUCCESS updated {name}")
        return 0
    finally:
        if process_group is not None:
            try:
                destroy_update_group(runtime, process_group, group_name=group_name)
            except Exception:
                pass
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())

    # Marker
