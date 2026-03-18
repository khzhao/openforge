#!/usr/bin/env python3

"""Launch real SGLang replicas plus an OpenForge router and send live requests.

This script:
1. Builds an ``OpenForgeConfig`` with a multi-replica rollout engine group.
2. Launches the SGLang engine replicas through ``start_sglang_engines``.
3. Starts an ``openforge.rollout.sglang.router.Router`` over those worker URLs.
4. Sends real ``/generate`` requests through the router and validates responses.

Intended usage on a single node with 4 visible GPUs:
    python test_scripts/test_openforge_router_live.py --train-gpus 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import snapshot_download

import openforge.configs.rollout as rollout_config_module

rollout_config_module.Any = Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FILTER_ENV = "OPENFORGE_FILTERED_VISIBLE_DEVICES"
FILTER_THRESHOLD_ENV = "OPENFORGE_FILTERED_VISIBLE_DEVICES_MIN_FREE_GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    parser.add_argument("--router-host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=0)
    parser.add_argument(
        "--router-policy",
        choices=("round_robin", "cache_aware"),
        default="round_robin",
    )
    parser.add_argument(
        "--router-log-level",
        default=None,
        help="Optional SGLang router log level, for example warn or error.",
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--min-free-gpu-memory-gb",
        type=float,
        default=0.0,
        help=(
            "Restrict CUDA_VISIBLE_DEVICES to GPUs with at least this much free "
            "memory before Ray starts."
        ),
    )
    return parser.parse_args()


def _parse_visible_device_filter(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    device_ids = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must contain numeric GPU ids when "
                "--min-free-gpu-memory-gb is used"
            )
        device_ids.add(int(token))
    return device_ids


def _query_free_gpu_memory() -> list[tuple[int, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi is required to filter GPUs by free memory"
        ) from exc

    free_memory = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index_raw, free_raw = [part.strip() for part in line.split(",", maxsplit=1)]
        free_memory.append((int(index_raw), int(free_raw)))
    return free_memory


def maybe_filter_visible_devices(args: argparse.Namespace) -> None:
    min_free_mb = max(int(args.min_free_gpu_memory_gb * 1024), 0)
    if min_free_mb <= 0 or os.environ.get(FILTER_ENV):
        return

    visible_filter = _parse_visible_device_filter(os.environ.get("CUDA_VISIBLE_DEVICES"))
    eligible_devices = []
    free_memory = {}
    for device_id, free_mb in _query_free_gpu_memory():
        free_memory[device_id] = free_mb
        if visible_filter is not None and device_id not in visible_filter:
            continue
        if free_mb >= min_free_mb:
            eligible_devices.append(device_id)

    required_devices = (
        args.train_gpus + args.rollout_replicas * args.gpus_per_replica
        if args.rollout_replicas is not None
        else args.train_gpus + args.gpus_per_replica
    )
    if len(eligible_devices) < required_devices:
        free_summary = ", ".join(
            f"{device_id}:{free_memory[device_id] / 1024:.1f}GB"
            for device_id in sorted(free_memory)
            if visible_filter is None or device_id in visible_filter
        )
        raise RuntimeError(
            "Not enough visible GPUs satisfy the free-memory requirement: "
            f"need {required_devices}, found {len(eligible_devices)} with "
            f"at least {args.min_free_gpu_memory_gb:.1f}GB free. "
            f"Visible device free memory: {free_summary}"
        )

    filtered_devices = ",".join(str(device_id) for device_id in eligible_devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = filtered_devices
    env[FILTER_ENV] = filtered_devices
    env[FILTER_THRESHOLD_ENV] = str(args.min_free_gpu_memory_gb)
    os.execvpe(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)


if __name__ == "__main__":
    maybe_filter_visible_devices(parse_args())

import ray
import torch

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.types import RouterSpec
from openforge.utils.networking import get_free_port

OpenForgeConfig.model_rebuild(force=True)


def require_visible_gpus(min_count: int) -> int:
    visible = torch.cuda.device_count()
    if visible < min_count:
        raise RuntimeError(
            f"Expected at least {min_count} visible GPUs, found {visible}"
        )
    return visible


def resolve_model_path(model_path_or_id: str) -> str:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model_path_or_id)


def build_cfg(
    *,
    model_path: str,
    visible_gpus: int,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    cpus_per_rollout_replica: int,
) -> OpenForgeConfig:
    checkpoint_root = tempfile.mkdtemp(prefix="openforge-router-live-")
    total_cpus = train_gpus + rollout_replicas * cpus_per_rollout_replica
    return OpenForgeConfig.model_validate(
        {
            "data": {},
            "gateway": {"host": "127.0.0.1", "port": 0},
            "model": {
                "model_name_or_path": model_path,
                "tokenizer_name_or_path": model_path,
                "attn_implementation": "sdpa",
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": visible_gpus,
                "cpus_per_node": max(total_cpus, 1),
            },
            "train": {
                "backend": "fsdp2",
                "config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": False,
                    "mixed_precision": {
                        "param_dtype": "bfloat16",
                        "reduce_dtype": "float32",
                    },
                    "offload": {"mode": "none", "pin_memory": False},
                    "amp": {
                        "enabled": False,
                        "precision": "float32",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 1e-5,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 1,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                    },
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints": checkpoint_root,
                "cpus_per_worker": 1,
                "parallel": {
                    "data_parallel_size": train_gpus,
                    "fsdp_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": 8,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                    "spaces_between_words": True,
                },
                "engine_groups": [
                    {
                        "name": "regular",
                        "worker_type": "regular",
                        "replicas": rollout_replicas,
                        "num_gpus_per_replica": gpus_per_replica,
                        "num_cpus_per_replica": cpus_per_rollout_replica,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": gpus_per_replica,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "enable_memory_saver": False,
                    }
                ],
            },
        }
    )


def wait_for_healthy_clients(
    clients: list["SGLangClient"],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(client.health_generate(timeout=5.0) for client in clients):
            return
        time.sleep(1.0)
    raise TimeoutError("Timed out waiting for all SGLang engines to become healthy")


def build_generate_payload_from_input_ids(
    *,
    input_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": False,
            "skip_special_tokens": True,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        },
        "return_logprob": True,
    }


def run_router_generate(
    router_url: str,
    *,
    request_index: int,
    prompt: str,
    tokenizer: Any,
    max_new_tokens: int,
    request_timeout: float,
) -> tuple[int, dict[str, Any]]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response = requests.post(
        f"{router_url}/generate",
        json=build_generate_payload_from_input_ids(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        ),
        timeout=request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Bad router generate response: {payload!r}")
    meta_info = payload.get("meta_info")
    if not isinstance(meta_info, dict):
        raise RuntimeError(f"Missing meta_info in router generate response: {payload!r}")
    return request_index, payload


def stress_router(
    router_url: str,
    *,
    tokenizer: Any,
    total_requests: int,
    concurrency: int,
    max_new_tokens: int,
    request_timeout: float,
) -> tuple[int, int, float, list[dict[str, Any]]]:
    started = time.monotonic()
    total_chars = 0
    response_payloads: list[dict[str, Any] | None] = [None] * total_requests

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = [
            executor.submit(
                run_router_generate,
                router_url,
                request_index=index,
                prompt=(
                    f"Router request {index}. "
                    "Write one concise sentence about distributed inference."
                ),
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                request_timeout=request_timeout,
            )
            for index in range(total_requests)
        ]
        for future in as_completed(futures):
            request_index, payload = future.result()
            response_payloads[request_index] = payload
            total_chars += len(payload["text"])

    return (
        total_requests,
        total_chars,
        time.monotonic() - started,
        [payload for payload in response_payloads if payload is not None],
    )


def main() -> int:
    from openforge.rollout.manager import start_sglang_engines
    from openforge.rollout.sglang.client import SGLangClient
    from openforge.rollout.sglang.router import Router
    from openforge.utils.ray import create_placement_groups
    from transformers import AutoTokenizer

    args = parse_args()
    filtered_devices = os.environ.get(FILTER_ENV)
    if filtered_devices:
        print(
            "Using filtered CUDA_VISIBLE_DEVICES="
            f"{filtered_devices} "
            f"(min_free_gpu_memory_gb={os.environ.get(FILTER_THRESHOLD_ENV, '0')})"
        )

    visible_gpus = require_visible_gpus(1)
    rollout_replicas = args.rollout_replicas
    if rollout_replicas is None:
        rollout_replicas = max(
            (visible_gpus - args.train_gpus) // args.gpus_per_replica,
            1,
        )

    total_requested = args.train_gpus + rollout_replicas * args.gpus_per_replica
    visible_gpus = require_visible_gpus(total_requested)
    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = build_cfg(
        model_path=model_path,
        visible_gpus=visible_gpus,
        train_gpus=args.train_gpus,
        rollout_replicas=rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
    )

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    placement_groups = None
    router = None
    engine_info: dict[str, Any] | None = None
    total_requests = 0
    total_chars = 0
    elapsed = 0.0
    response_payloads: list[dict[str, Any]] = []
    router_port = args.router_port or get_free_port(start=30000)
    try:
        placement_groups = create_placement_groups(cfg)
        engine_info = start_sglang_engines(cfg, placement_groups)

        engine_urls = [
            engine_info["engine_addrs"][spec.engine_name].url
            for spec in engine_info["engine_specs"]
        ]
        clients = [SGLangClient(url) for url in engine_urls]
        wait_for_healthy_clients(clients, timeout=float(args.startup_timeout))

        for client in clients:
            assert client.get_model_info(timeout=10.0)
            assert client.get_server_info(timeout=10.0)

        router_spec = RouterSpec(
            router_name="openforge-live-router",
            router_ip=args.router_host,
            router_port=router_port,
            policy=args.router_policy,
            worker_urls=[],
            request_timeout_secs=max(int(args.request_timeout), 1),
            worker_startup_timeout_secs=args.startup_timeout,
            worker_startup_check_interval=1,
            health_check_timeout_secs=5,
            health_check_interval_secs=5,
            log_level=args.router_log_level,
        )
        router = Router()
        router.initialize(router_spec)
        for url in engine_urls:
            router.add_worker(url)
        router.launch()

        total_requests, total_chars, elapsed, response_payloads = stress_router(
            router.url,
            tokenizer=tokenizer,
            total_requests=args.num_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout,
        )

        for client in clients:
            assert client.flush_cache(timeout=10.0)
    finally:
        if router is not None:
            router.shutdown()
        if engine_info is not None:
            workers = engine_info["engine_workers"]
            try:
                ray.get([worker.stop.remote() for worker in workers])
            except Exception:
                pass
            for worker in workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
        if placement_groups is not None:
            try:
                ray.util.remove_placement_group(placement_groups["actor"][0])
            except Exception:
                pass
        ray.shutdown()

    for index, payload in enumerate(response_payloads):
        print(f"RESPONSE_JSON[{index}]")
        print(json.dumps(payload, sort_keys=True))

    print(
        "SUCCESS "
        f"model={Path(model_path).name} "
        f"router={router.url if router is not None else 'unavailable'} "
        f"replicas={rollout_replicas} "
        f"gpus_per_replica={args.gpus_per_replica} "
        f"total_requests={total_requests} "
        f"total_chars={total_chars} "
        f"elapsed_s={elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
