# Copyright 2026 openforge

import multiprocessing
import os
from multiprocessing.process import BaseProcess
from typing import Any

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang_router.router_args import RouterArgs

from openforge.rollout.types import EngineAddr, EngineSpec, RouterSpec


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


def _coerce_router_args(router_args: RouterArgs | dict[str, Any]) -> RouterArgs:
    """Normalize a RouterArgs payload to a concrete RouterArgs object."""
    if isinstance(router_args, RouterArgs):
        return router_args
    return RouterArgs(**router_args)


def serve_sglang(server_args: ServerArgs | dict[str, Any]) -> None:
    """Serve SGLang HTTP server."""
    from sglang.srt.entrypoints.http_server import launch_server

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


def generate_sglang_server_args(
    engine_spec: EngineSpec,
    engine_addr: EngineAddr,
) -> ServerArgs:
    """Generate SGLang HTTP server arguments for a rollout engine replica."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        cvds = [int(device.strip()) for device in cuda_visible_devices.split(",")]
        assert cvds == sorted(cvds), (
            "CUDA_VISIBLE_DEVICES must be ordered by physical GPU id: "
            f"{cuda_visible_devices}"
        )

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


def generate_sglang_router_args(router_spec: RouterSpec) -> RouterArgs:
    """Generate SGLang router arguments for a rollout engine group."""
    router_args_payload = {
        "worker_urls": router_spec.worker_urls,
        "host": router_spec.router_ip,
        "port": router_spec.router_port,
        "policy": router_spec.policy,
        "request_timeout_secs": router_spec.request_timeout_secs,
        "worker_startup_timeout_secs": router_spec.worker_startup_timeout_secs,
        "worker_startup_check_interval": router_spec.worker_startup_check_interval,
        "health_check_timeout_secs": router_spec.health_check_timeout_secs,
        "health_check_interval_secs": router_spec.health_check_interval_secs,
    }
    return RouterArgs(**router_args_payload)


def serve_sglang_router(router_args: RouterArgs | dict[str, Any]) -> None:
    """Serve SGLang router."""
    from sglang_router.launch_router import launch_router

    router_args = _coerce_router_args(router_args)
    router_args.host = str(router_args.host).strip("[]")
    try:
        launch_router(router_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_sglang_router(router_args: RouterArgs | dict[str, Any]) -> BaseProcess:
    """Launch a SGLang router process."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=serve_sglang_router, args=(router_args,))
    process.start()
    return process
