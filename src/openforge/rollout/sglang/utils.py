# Copyright 2026 openforge

import multiprocessing
import os
from multiprocessing.process import BaseProcess
from typing import Any

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from openforge.configs.models import OpenForgeConfig


def get_sglang_global_gpu_offset(
    cfg: OpenForgeConfig,
    sglang_engine_rank: int,
    colocated: bool = False,
) -> int:
    """Return the first global GPU id assigned to one SGLang engine replica."""
    base_offset = 0 if colocated else cfg.train.total_gpus
    per_engine_gpus = [
        engine.num_gpus
        for engine in cfg.rollout.engines
        for _ in range(engine.replicas)
    ]
    base_offset = 0 if colocated else cfg.train.total_gpus
    return base_offset + sum(per_engine_gpus[:sglang_engine_rank])


def get_local_gpu_id(physical_gpu_id: int) -> int:
    """Get the local GPU id for the given physical GPU id."""
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not CUDA_VISIBLE_DEVICES:
        return physical_gpu_id
    gpu_ids = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))
    if physical_gpu_id in gpu_ids:
        return gpu_ids.index(physical_gpu_id)
    if 0 <= physical_gpu_id < len(gpu_ids):
        return physical_gpu_id
    raise RuntimeError(
        f"physical GPU id: {physical_gpu_id} is not valid under "
        f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}"
    )


def serve_sglang(server_args_payload: dict[str, Any]) -> None:
    """Serve SGLang HTTP server."""
    server_args = ServerArgs(**server_args_payload)
    server_args.host = str(server_args.host).strip("[]")
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_sglang_process(server_args_payload: dict[str, Any]) -> BaseProcess:
    """Launch a SGLang HTTP server process."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=serve_sglang, args=(server_args_payload,))
    process.start()
    return process


def kill_sglang_process_tree(pid: int) -> bool:
    """Kill a SGLang HTTP server process and its children."""
    try:
        kill_process_tree(pid, include_parent=True)
    except Exception:
        return False
    return True
