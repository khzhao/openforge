# Copyright 2026 openforge

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from _sglang_weight_update_common import (
    DEFAULT_MODEL,
    assert_success,
    find_free_port,
    init_trainer_update_group,
    load_source_model,
    require_visible_gpus,
    resolve_local_model_path,
    start_runtime,
    torch_dtype_name,
)
from test_sglang_update_from_distributed import (
    destroy_update_group,
    run_runtime_call_during_local_step,
)


def main() -> int:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    require_visible_gpus(2)

    model_path = resolve_local_model_path(DEFAULT_MODEL)
    source_device = torch.device("cuda", 0)
    group_name = "openforge_debug_bucket"
    master_port = find_free_port()
    default_group_port = find_free_port()

    if os.environ.get("OPENFORGE_INIT_DEFAULT_PG") == "1":
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{default_group_port}",
            world_size=1,
            rank=0,
            timeout=timedelta(seconds=60),
        )
        torch.cuda.set_device(source_device)
        print("initialized default process group", flush=True)

    runtime, _cfg = start_runtime(
        model_path=model_path,
        name="sglang-debug-bucket",
        train_total_gpus=1,
    )
    source_model = load_source_model(model_path, device=source_device)
    process_group = None
    try:
        allowed_names = {name for name, _ in source_model.named_parameters()}
        named_tensors: list[tuple[str, torch.Tensor]] = []
        total_bytes = 0
        for name, tensor in source_model.state_dict().items():
            if name not in allowed_names or not tensor.is_floating_point():
                continue
            device_tensor = tensor.detach().to(source_device).contiguous()
            named_tensors.append((name, device_tensor))
            total_bytes += device_tensor.numel() * device_tensor.element_size()
            if total_bytes >= (4 << 20):
                break

        print(
            f"bucket_len={len(named_tensors)} bytes={total_bytes} first={named_tensors[0][0]}",
            flush=True,
        )

        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flat = bucket.get_flattened_tensor()
        names = [name for name, _ in named_tensors]
        dtypes = [torch_dtype_name(tensor.dtype) for _, tensor in named_tensors]
        shapes = [list(tensor.shape) for _, tensor in named_tensors]

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
        assert_success(init_response, context="init_weights_update_group")

        update_response, _ = run_runtime_call_during_local_step(
            lambda: runtime.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                load_format="flattened_bucket",
                flush_cache=True,
                weight_version="999",
            ),
            lambda: dist.broadcast(flat, src=0, group=process_group),
        )
        print(update_response, flush=True)
        assert_success(update_response, context="update_weights_from_distributed")

        destroy_update_group(runtime, process_group, group_name=group_name)
        process_group = None
        print("OK", flush=True)
        return 0
    finally:
        if process_group is not None:
            try:
                destroy_update_group(runtime, process_group, group_name=group_name)
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())
