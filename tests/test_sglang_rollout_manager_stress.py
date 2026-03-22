# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ray
from huggingface_hub import snapshot_download
from ray._private import worker as ray_worker

import openforge.configs.rollout as rollout_config_module

rollout_config_module.Any = Any

from _script_test_utils import free_gpu_ids, require_free_gpu_ids, start_test_ray_cluster
from openforge.configs.models import OpenForgeConfig

OpenForgeConfig.model_rebuild(force=True)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAUNCH_RE = re.compile(
    r"Launching SGLang server on EngineAddr\(host='([^']+)', port=(\d+), "
    r"nccl_port=(\d+), dist_init_addr='([^']+)'\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--requests-per-engine", type=int, default=8)
    parser.add_argument("--prompt-words", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--shutdown-timeout", type=float, default=60.0)
    parser.add_argument("--log-timeout", type=float, default=30.0)
    return parser.parse_args()
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
    checkpoint_root = tempfile.mkdtemp(prefix="openforge-rollout-stress-")
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


def read_ray_logs(session_dir: str) -> str:
    logs_dir = Path(session_dir) / "logs"
    parts = []
    for pattern in ("*.out", "*.err"):
        for path in sorted(logs_dir.rglob(pattern)):
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def wait_for_engine_addrs(
    *,
    session_dir: str,
    expected_replicas: int,
    timeout: float,
) -> list[str]:
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        addrs = {
            f"http://{match.group(1)}:{int(match.group(2))}"
            for match in LAUNCH_RE.finditer(read_ray_logs(session_dir))
        }
        last_count = len(addrs)
        if last_count >= expected_replicas:
            return sorted(addrs)[:expected_replicas]
        time.sleep(1.0)
    raise TimeoutError(
        f"Expected {expected_replicas} launch log lines, found {last_count}"
    )


def wait_for_healthy_clients(
    clients: list[SGLangClient],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(client.health_generate(timeout=5.0) for client in clients):
            return
        time.sleep(1.0)
    raise TimeoutError("Timed out waiting for all SGLang servers to become healthy")


def wait_for_clients_to_stop(
    clients: list[SGLangClient],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(not client.health_generate(timeout=1.0) for client in clients):
            return
        time.sleep(1.0)
    raise TimeoutError("Timed out waiting for SGLang servers to stop")


def build_payload(
    *,
    prompt: str,
    max_new_tokens: int,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    return {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
        },
    }


def run_generate(
    client: SGLangClient,
    *,
    prompt: str,
    max_new_tokens: int,
    request_timeout: float,
) -> int:
    response = client.generate(
        **build_payload(prompt=prompt, max_new_tokens=max_new_tokens),
        timeout=request_timeout,
    )
    text = response["text"]
    assert isinstance(text, str) and text.strip(), f"Bad generate response: {response}"
    return len(text)


def assert_completion_length(response: dict[str, Any], *, expected_tokens: int) -> None:
    output_ids = response.get("output_ids")
    if not isinstance(output_ids, list):
        raise RuntimeError(f"Bad generate response: {response!r}")
    if len(output_ids) != expected_tokens:
        raise RuntimeError(
            f"expected {expected_tokens} completion tokens, got {len(output_ids)}: {response!r}"
        )

    meta_info = response.get("meta_info")
    if isinstance(meta_info, dict):
        completion_tokens = meta_info.get("completion_tokens")
        if completion_tokens is not None and int(completion_tokens) != expected_tokens:
            raise RuntimeError(
                "client did not respect sampling_params.max_new_tokens: "
                f"expected {expected_tokens}, got completion_tokens={completion_tokens}"
            )


def validate_client_sampling_params(
    clients: list[SGLangClient],
    *,
    request_timeout: float,
) -> None:
    prompt = "Validate sampling params on the direct SGLang client path."
    for client in clients:
        for max_new_tokens in (1, 4):
            response = client.generate(
                **build_payload(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=True,
                ),
                timeout=request_timeout,
            )
            assert_completion_length(response, expected_tokens=max_new_tokens)


def stress_clients(
    clients: list[SGLangClient],
    *,
    rounds: int,
    requests_per_engine: int,
    prompt_words: int,
    max_new_tokens: int,
    request_timeout: float,
) -> tuple[int, int, float]:
    prompt_prefix = " ".join(["distributed"] * prompt_words)
    total_requests = 0
    total_chars = 0
    start = time.monotonic()

    for round_idx in range(rounds):
        with ThreadPoolExecutor(
            max_workers=len(clients) * requests_per_engine
        ) as executor:
            futures = []
            for engine_idx, client in enumerate(clients):
                for request_idx in range(requests_per_engine):
                    prompt = (
                        f"{prompt_prefix}\n"
                        f"Round {round_idx}, engine {engine_idx}, request {request_idx}. "
                        "Write two short sentences about GPU inference."
                    )
                    futures.append(
                        executor.submit(
                            run_generate,
                            client,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            request_timeout=request_timeout,
                        )
                    )

            for future in as_completed(futures):
                total_chars += future.result()
                total_requests += 1

    return total_requests, total_chars, time.monotonic() - start


def main() -> int:
    args = parse_args()

    from openforge.rollout.manager import RolloutManager
    from openforge.rollout.sglang.client import SGLangClient
    from openforge.utils.ray import create_placement_groups

    gpu_ids = free_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("Expected at least 1 free GPU, found 0")
    rollout_replicas = args.rollout_replicas
    if rollout_replicas is None:
        rollout_replicas = max(
            (len(gpu_ids) - args.train_gpus) // args.gpus_per_replica,
            1,
        )

    total_requested = args.train_gpus + rollout_replicas * args.gpus_per_replica
    gpu_ids = require_free_gpu_ids(total_requested)
    model_path = resolve_model_path(args.model_path)
    cfg = build_cfg(
        model_path=model_path,
        visible_gpus=len(gpu_ids),
        train_gpus=args.train_gpus,
        rollout_replicas=rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
    )

    start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=max(len(gpu_ids), 1),
    )

    manager = None
    total_requests = 0
    total_chars = 0
    elapsed = 0.0
    placement_groups = None
    try:
        session_dir = ray_worker._global_node.get_session_dir_path()
        placement_groups = create_placement_groups(cfg)
        manager = RolloutManager(cfg, placement_groups)
        manager.initialize()

        urls = wait_for_engine_addrs(
            session_dir=session_dir,
            expected_replicas=rollout_replicas,
            timeout=args.log_timeout,
        )
        clients = [SGLangClient(url) for url in urls]
        wait_for_healthy_clients(clients, timeout=args.log_timeout)

        for client in clients:
            assert client.get_model_info(timeout=10.0)
            assert client.get_server_info(timeout=10.0)
        validate_client_sampling_params(
            clients,
            request_timeout=args.request_timeout,
        )

        total_requests, total_chars, elapsed = stress_clients(
            clients,
            rounds=args.rounds,
            requests_per_engine=args.requests_per_engine,
            prompt_words=args.prompt_words,
            max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout,
        )

        for client in clients:
            assert client.flush_cache(timeout=10.0)

        manager.shutdown()
        manager = None
        wait_for_clients_to_stop(clients, timeout=args.shutdown_timeout)
    finally:
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
        if placement_groups is not None:
            try:
                ray.util.remove_placement_group(placement_groups["actor"][0])
            except Exception:
                pass
        ray.shutdown()

    print(
        "SUCCESS "
        f"model={Path(model_path).name} "
        f"replicas={rollout_replicas} "
        f"gpus_per_replica={args.gpus_per_replica} "
        f"total_requests={total_requests} "
        f"total_chars={total_chars} "
        f"elapsed_s={elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
