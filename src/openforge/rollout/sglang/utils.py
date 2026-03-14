# Copyright 2026 openforge

import multiprocessing
import os
from multiprocessing.process import BaseProcess
from typing import Any

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from openforge.rollout.types import EngineAddr, EngineSpec


def get_local_gpu_id(global_gpu_id: int) -> int:
    """Get the local GPU id for the given physical GPU id."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return global_gpu_id
    gpu_ids = list(map(int, cuda_visible_devices.split(",")))
    if global_gpu_id in gpu_ids:
        return gpu_ids.index(global_gpu_id)
    if 0 <= global_gpu_id < len(gpu_ids):
        return global_gpu_id
    raise RuntimeError(
        f"global GPU id: {global_gpu_id} is not valid under "
        f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}"
    )


def _coerce_server_args(server_args: ServerArgs | dict[str, Any]) -> ServerArgs:
    """Normalize a ServerArgs payload to a concrete ServerArgs object."""
    if isinstance(server_args, ServerArgs):
        return server_args
    return ServerArgs(**server_args)


def serve_sglang(server_args: ServerArgs | dict[str, Any]) -> None:
    """Serve SGLang HTTP server."""
    server_args = _coerce_server_args(server_args)
    server_args.host = str(server_args.host).strip("[]")
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_sglang_process(server_args: ServerArgs | dict[str, Any]) -> BaseProcess:
    """Launch a SGLang HTTP server process."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=serve_sglang, args=(server_args,))
    process.start()
    return process


def kill_sglang_process_tree(pid: int) -> bool:
    """Kill a SGLang HTTP server process and its children."""
    try:
        kill_process_tree(pid, include_parent=True)
    except Exception:
        return False
    return True


def generate_sglang_server_args(
    engine_spec: EngineSpec,
    engine_addr: EngineAddr,
) -> ServerArgs:
    """Generate SGLang HTTP server arguments for a rollout engine replica."""
    base_gpu_id = get_local_gpu_id(engine_spec.base_gpu_id)
    server_args_payload = {
        # Model
        "model_path": engine_spec.cfg.model.model_name_or_path,
        "trust_remote_code": True,
        # Server
        "host": engine_addr.host,
        "port": engine_addr.port,
        "nnodes": engine_spec.num_nodes,
        "node_rank": engine_spec.node_rank,
        "dist_init_addr": engine_addr.dist_init_addr,
        "nccl_port": engine_addr.nccl_port,
        # Distributed
        "base_gpu_id": base_gpu_id,
        "gpu_id_step": 1,
        "dp_size": engine_spec.parallelism.data_parallel_size,
        "pp_size": engine_spec.parallelism.pipeline_parallel_size,
        "tp_size": engine_spec.parallelism.tensor_parallel_size,
        "ep_size": engine_spec.parallelism.expert_parallel_size,
        # Misc
        "enable_memory_saver": engine_spec.enable_memory_saver,
        "skip_server_warmup": True,
        "enable_draft_weights_cpu_backup": True,
    }

    if engine_spec.sglang_server_overrides:
        server_args_payload.update(engine_spec.sglang_server_overrides)
    return ServerArgs(**server_args_payload)
