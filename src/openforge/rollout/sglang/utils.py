# Copyright 2026 openforge

import multiprocessing
import os
from itertools import accumulate
from multiprocessing.process import BaseProcess
from typing import Any

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from openforge.configs.models import OpenForgeConfig
from openforge.configs.topology import ParallelismConfig


def get_sglang_global_gpu_offset_for_replica(
    cfg: OpenForgeConfig,
    engine_replica_index: int,
    colocated: bool = False,
) -> int:
    """Return the first global GPU id assigned to one rollout engine replica."""
    assert engine_replica_index >= 0, "engine_replica_index must be non-negative"
    assert engine_replica_index < cfg.rollout.num_engine_replicas, (
        "engine_replica_index is out of range"
    )
    base_offset = 0 if colocated else cfg.train.total_gpus
    per_replica_gpus = [
        engine_group.num_gpus_per_replica
        for engine_group in cfg.rollout.engine_groups
        for _ in range(engine_group.replicas)
    ]
    offsets = list(accumulate([base_offset, *per_replica_gpus]))
    return offsets[engine_replica_index]


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
    cfg: OpenForgeConfig,
    engine_replica_index: int,
    colocated: bool = False,
    *,
    model_path: str,
    host: str,
    port: int,
    num_nodes: int,
    node_rank: int,
    dist_init_addr: str,
    nccl_port: int,
    parallelism_config: ParallelismConfig,
    enable_memory_saver: bool = False,
    override_server_args: dict[str, Any] | None = None,
) -> ServerArgs:
    """Generate SGLang HTTP server arguments for a rollout engine replica."""
    global_gpu_offset = get_sglang_global_gpu_offset_for_replica(
        cfg,
        engine_replica_index,
        colocated,
    )
    local_gpu_id = get_local_gpu_id(global_gpu_offset)

    server_args_payload = {
        # Model
        "model_path": model_path,
        "trust_remote_code": True,
        # Server
        "host": host,
        "port": port,
        "nnodes": num_nodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "nccl_port": nccl_port,
        # Distributed
        "base_gpu_id": local_gpu_id,
        "gpu_id_step": 1,
        "dp_size": parallelism_config.data_parallel_size,
        "pp_size": parallelism_config.pipeline_parallel_size,
        "tp_size": parallelism_config.tensor_parallel_size,
        "ep_size": parallelism_config.expert_parallel_size,
        # Misc
        "enable_memory_saver": enable_memory_saver,
        "skip_server_warmup": True,
        "enable_draft_weights_cpu_backup": True,
    }

    if override_server_args is not None:
        server_args_payload.update(override_server_args)
    return ServerArgs(**server_args_payload)
