#!/usr/bin/env python
# Copyright 2026 openforge
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import requests
import torch

from openforge.configs.models import GatewayServerConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import Runtime
from openforge.gateway.service import Service
from openforge.gateway.types import (
    ChatCompletionCreateRequest,
    ChatMessage,
    RuntimeConfig,
)
from openforge.train.server import TrainServerClient

VISIBLE_GPUS_REQUIRED = 4
SMALL_GROUP_SIZE = 2
LARGE_GROUP_SIZE = 4
GLOBAL_BATCH_SIZE = SMALL_GROUP_SIZE + LARGE_GROUP_SIZE
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TIMEOUT_SECONDS = 1800.0
POLL_INTERVAL_SECONDS = 0.2
ROUTER_STATUS_TIMEOUT_SECONDS = 30.0


def build_server_config() -> GatewayServerConfig:
    """Build the gateway-owned runtime shell config."""
    return GatewayServerConfig.model_validate(
        {
            "data": {},
            "gateway": {
                "host": "127.0.0.1",
                "port": 9999,
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": 4,
                "cpus_per_node": max(8, os.cpu_count() or 8),
            },
        }
    )


def build_runtime_config(checkpoint_dir: Path) -> RuntimeConfig:
    """Build the runtime config for async rollout + training."""
    return RuntimeConfig.model_validate(
        {
            "algo": {
                "name": "grpo_tis",
                "tis_cap": 2.0,
            },
            "model": {
                "model_name_or_path": MODEL_NAME,
                "tokenizer_name_or_path": MODEL_NAME,
                "attn_implementation": "sdpa",
            },
            "train": {
                "backend": "fsdp2",
                "config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": True,
                    "mixed_precision": {
                        "param_dtype": "bfloat16",
                        "reduce_dtype": "float32",
                    },
                    "offload": {
                        "mode": "none",
                        "pin_memory": False,
                    },
                    "amp": {
                        "enabled": True,
                        "precision": "bfloat16",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 1e-6,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 0,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                    },
                },
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "ppo_epochs": 4,
                "max_rollout_policy_lag": 2,
                "checkpoints": str(checkpoint_dir),
                "cpus_per_worker": 1,
                "parallel": {
                    "data_parallel_size": 1,
                    "fsdp_parallel_size": 2,
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
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 64,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": False,
                    "no_stop_trim": False,
                },
                "engine_groups": [
                    {
                        "name": "rollout",
                        "worker_type": "regular",
                        "replicas": 2,
                        "num_gpus_per_replica": 1,
                        "num_cpus_per_replica": 1,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "enable_memory_saver": False,
                        "sglang_server_overrides": {
                            "context_length": 1024,
                            "mem_fraction_static": 0.6,
                            "disable_cuda_graph": True,
                        },
                    }
                ],
            },
        }
    )


def make_generate_request(
    *,
    session_id: str,
    trajectory_id: str,
    group_id: str,
    prompt: str,
) -> ChatCompletionCreateRequest:
    """Build one gateway generate request."""
    return ChatCompletionCreateRequest.model_validate(
        {
            "_openforge": {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "group_id": group_id,
            },
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_completion_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )


async def wait_for(
    label: str,
    predicate,
    *,
    timeout: float = TIMEOUT_SECONDS,
):
    """Poll until a synchronous predicate returns a truthy value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"timed out waiting for {label}")


def router_status(runtime: Runtime) -> dict:
    """Return the current router status payload."""
    slot = runtime._slot
    assert slot is not None
    return slot.rollout_manager.router.status(timeout=ROUTER_STATUS_TIMEOUT_SECONDS)


def router_workers(runtime: Runtime) -> list[dict[str, object]]:
    """Return rollout worker metadata from the router."""
    workers = list(router_status(runtime)["workers"].values())
    assert len(workers) == 2
    return workers


def synced_rollout_version(runtime: Runtime) -> int | None:
    """Return the synced rollout version once all workers are active."""
    workers = router_workers(runtime)
    versions = [int(worker["weight_version"]) for worker in workers]
    states = [str(worker["state"]) for worker in workers]
    if all(state == "active" for state in states) and len(set(versions)) == 1:
        return versions[0]
    return None


async def wait_for_all_trained(
    store: SQLiteOpenForgeStore,
    session_id: str,
    *,
    expected_trajectories: int,
) -> None:
    """Wait until all session trajectories are marked trained."""
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        trajectories = await store.list_trajectories(session_id)
        if len(trajectories) == expected_trajectories and all(
            trajectory.status == "trained" for trajectory in trajectories
        ):
            return
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError("timed out waiting for all trajectories to train")


async def start_group(
    *,
    service: Service,
    session_id: str,
    count: int = SMALL_GROUP_SIZE,
) -> tuple[str, list[str]]:
    """Start one group and return its group id and trajectory ids."""
    group_id = f"group_{uuid4().hex[:8]}"
    started = await service.start_trajectory_groups(
        session_id=session_id,
        counts=[count],
        group_ids=[group_id],
    )
    return group_id, started.trajectory_ids[0]


async def generate_one(
    *,
    service: Service,
    session_id: str,
    trajectory_id: str,
    group_id: str,
    prompt: str,
) -> int:
    """Generate one trajectory and return the rollout policy version."""
    response = await service.generate(
        request=make_generate_request(
            session_id=session_id,
            trajectory_id=trajectory_id,
            group_id=group_id,
            prompt=prompt,
        )
    )
    return int(response.metadata["rollout_model_version"])


async def complete_one(
    *,
    service: Service,
    session_id: str,
    trajectory_id: str,
    reward: float,
) -> None:
    """Mark one trajectory completed."""
    await service.end_trajectories(
        session_id=session_id,
        trajectory_ids=[trajectory_id],
        final_rewards=[reward],
    )


async def run_group(
    *,
    service: Service,
    session_id: str,
    prompts: list[str],
    rewards: list[float],
) -> tuple[list[str], list[int]]:
    """Start one group, generate responses, and complete the trajectories."""
    assert len(prompts) == len(rewards)
    group_id, trajectory_ids = await start_group(
        service=service,
        session_id=session_id,
        count=len(prompts),
    )
    versions = await asyncio.gather(
        *[
            generate_one(
                service=service,
                session_id=session_id,
                trajectory_id=trajectory_id,
                group_id=group_id,
                prompt=prompt,
            )
            for trajectory_id, prompt in zip(trajectory_ids, prompts, strict=True)
        ]
    )
    await service.end_trajectories(
        session_id=session_id,
        trajectory_ids=trajectory_ids,
        final_rewards=rewards,
    )
    return trajectory_ids, list(versions)


def hold_trajectory(
    runtime: Runtime,
    trajectory_id: str,
    prompt: str,
) -> tuple[str, int]:
    """Pin one direct router trajectory to a rollout worker and leave it active."""
    slot = runtime._slot
    assert slot is not None
    input_ids = runtime.tokenize_messages(
        [ChatMessage.model_validate({"role": "user", "content": prompt})]
    )
    sampling_params = runtime._runtime_cfg.rollout.request.model_dump()
    payload = slot.rollout_manager.router.generate(
        sampling_params,
        input_ids=input_ids,
        trajectory_id=trajectory_id,
        return_logprob=True,
    )
    assert isinstance(payload, dict)
    version = int(payload["meta_info"]["weight_version"])
    trajectory = router_status(runtime)["trajectories"][trajectory_id]
    return str(trajectory["worker_name"]), version


def assert_all_ok(payload: dict) -> None:
    """Assert a train sync payload reported success for every worker."""
    results = payload["results"]
    assert isinstance(results, list)
    assert results
    for result in results:
        assert bool(result["ok"])


def assert_all_error(payload: dict, text: str) -> None:
    """Assert a train sync payload reported the given error for every worker."""
    results = payload["results"]
    assert isinstance(results, list)
    for result in results:
        assert text in str(result["error"])


def independent_update_status(
    runtime: Runtime,
    hold_trajectory_id: str,
) -> dict | None:
    """Return router status once one worker has updated and the held worker has not."""
    status = router_status(runtime)
    trajectory = status["trajectories"].get(hold_trajectory_id)
    if trajectory is None:
        return None
    hold_worker_name = str(trajectory["worker_name"])
    hold_worker = status["workers"][hold_worker_name]
    if not (
        int(hold_worker["weight_version"]) == 1
        and str(hold_worker["state"]) == "draining"
        and int(hold_worker["active_trajectory_count"]) >= 1
    ):
        return None
    other_workers = [
        worker
        for worker_name, worker in status["workers"].items()
        if worker_name != hold_worker_name
    ]
    if len(other_workers) != 1:
        return None
    other_worker = other_workers[0]
    if not (
        int(other_worker["weight_version"]) >= 2
        and str(other_worker["state"]) == "active"
    ):
        return None
    return status


async def main() -> None:
    """Run a live end-to-end async train + rollout script."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cuda_visible_devices, (
        "launch with CUDA_VISIBLE_DEVICES set to exactly four GPUs, "
        "for example CUDA_VISIBLE_DEVICES=1,2,6,7"
    )
    visible_gpu_ids = [part.strip() for part in cuda_visible_devices.split(",") if part]
    assert len(visible_gpu_ids) == VISIBLE_GPUS_REQUIRED, (
        f"expected exactly {VISIBLE_GPUS_REQUIRED} visible GPUs, got "
        f"{len(visible_gpu_ids)} from CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
    )

    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    assert torch.cuda.device_count() == VISIBLE_GPUS_REQUIRED, (
        f"torch sees {torch.cuda.device_count()} GPUs, expected "
        f"{VISIBLE_GPUS_REQUIRED}"
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="openforge-e2e-async-live-"))
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteOpenForgeStore(temp_dir / "store.sqlite")
    runtime = Runtime(cfg=build_server_config())
    service = Service(store=store, runtime=runtime)
    keep_temp_dir = False

    try:
        session = await service.start_session(
            runtime_config=build_runtime_config(checkpoint_dir)
        )
        session_id = session.session_id
        train_runtime = runtime.train()
        train_server_url = train_runtime.server_url
        train_client = TrainServerClient(train_server_url)
        router = runtime._slot.rollout_manager.router
        manager = train_runtime.manager
        original_new_id = service._new_id
        scripted_trajectory_ids = iter(
            [
                "traj_0000",
                "traj_0001",
                "traj_0010",
                "traj_0011",
                "traj_0012",
                "traj_0013",
                "traj_0100",
                "traj_0101",
                "traj_0200",
                "traj_0201",
                "traj_0202",
                "traj_0203",
                "traj_0900",
                "traj_0901",
                "traj_1000",
                "traj_1001",
                "traj_1002",
                "traj_1003",
            ]
        )

        def scripted_new_id(prefix: str) -> str:
            if prefix == "traj":
                return next(scripted_trajectory_ids)
            return original_new_id(prefix)

        service._new_id = scripted_new_id

        health = requests.get(f"{train_server_url}/health", timeout=30.0)
        health.raise_for_status()

        workers = [
            {
                "worker_url": str(worker["worker_url"]),
                "world_size": int(worker["world_size"]),
            }
            for worker in router_workers(runtime)
        ]

        empty_payload = train_client.request_weight_sync(
            workers=[],
            target_version=0,
        )
        assert empty_payload["results"] == []

        mismatch_payload = train_client.request_weight_sync(
            workers=workers,
            target_version=1,
        )
        assert_all_error(mismatch_payload, "target_version does not match")

        assert manager._sync_lock.acquire(blocking=False)
        try:
            in_progress_payload = train_client.request_weight_sync(
                workers=workers,
                target_version=0,
            )
        finally:
            manager._sync_lock.release()
        assert_all_error(in_progress_payload, "training_in_progress")

        group_a_ids, group_a_versions = await run_group(
            service=service,
            session_id=session_id,
            prompts=[
                "Explain PPO in one sentence.",
                "Explain KL regularization in one sentence.",
            ],
            rewards=[1.0, 0.0],
        )
        group_b_ids, group_b_versions = await run_group(
            service=service,
            session_id=session_id,
            prompts=[
                "List two causes of policy lag.",
                "List two causes of CUDA OOM.",
                "List two reasons asynchronous training can help throughput.",
                "List two reasons checkpoint export can be slow.",
            ],
            rewards=[0.8, 0.2, 0.6, 0.4],
        )
        assert set(group_a_versions) == {0}
        assert set(group_b_versions) == {0}

        await wait_for(
            "policy version 1",
            lambda: train_runtime.policy_version >= 1,
        )
        await wait_for(
            "rollout workers synced to version 1",
            lambda: synced_rollout_version(runtime) == 1,
        )

        hold_trajectory_id = f"hold_{uuid4().hex[:8]}"
        hold_worker_name, hold_version = hold_trajectory(
            runtime,
            hold_trajectory_id,
            "Keep this trajectory pinned to one rollout worker.",
        )
        assert hold_version == 1

        stale_group_id, stale_group_ids = await start_group(
            service=service,
            session_id=session_id,
        )
        stale_group_first_version = await generate_one(
            service=service,
            session_id=session_id,
            trajectory_id=stale_group_ids[0],
            group_id=stale_group_id,
            prompt="Explain why stale data can still be useful briefly.",
        )
        assert stale_group_first_version == 1
        await complete_one(
            service=service,
            session_id=session_id,
            trajectory_id=stale_group_ids[0],
            reward=0.65,
        )

        group_c_ids, group_c_versions = await run_group(
            service=service,
            session_id=session_id,
            prompts=[
                "Explain trajectory pinning in one sentence.",
                "Explain rollout draining in one sentence.",
            ],
            rewards=[0.9, 0.1],
        )
        group_d_ids, group_d_versions = await run_group(
            service=service,
            session_id=session_id,
            prompts=[
                "Explain asynchronous policy updates briefly.",
                "Explain TIS briefly.",
                "Explain why worker-local draining is useful briefly.",
                "Explain why rollout workers need explicit versions briefly.",
            ],
            rewards=[0.7, 0.3, 0.6, 0.4],
        )
        assert min(group_c_versions) >= 1
        assert min(group_d_versions) >= 1

        stale_group_second_version = await generate_one(
            service=service,
            session_id=session_id,
            trajectory_id=stale_group_ids[1],
            group_id=stale_group_id,
            prompt="Explain why rollout lag should be bounded briefly.",
        )
        assert stale_group_second_version == 1
        await complete_one(
            service=service,
            session_id=session_id,
            trajectory_id=stale_group_ids[1],
            reward=0.35,
        )
        stale_group_versions = [
            stale_group_first_version,
            stale_group_second_version,
        ]

        await wait_for(
            "policy version 2",
            lambda: train_runtime.policy_version >= 2,
        )

        independent_status = await wait_for(
            "independent rollout worker update",
            lambda: independent_update_status(runtime, hold_trajectory_id),
        )
        independent_versions = {
            name: (
                int(worker["weight_version"]),
                str(worker["state"]),
                int(worker["active_trajectory_count"]),
            )
            for name, worker in independent_status["workers"].items()
        }

        fresh_group_ids, fresh_group_versions = await run_group(
            service=service,
            session_id=session_id,
            prompts=[
                "List three benefits of worker-local draining.",
                "List three benefits of stable policy versions.",
                "List three benefits of async train and rollout decoupling.",
                "List three reasons to keep the train server thin.",
            ],
            rewards=[0.55, 0.45, 0.7, 0.3],
        )
        assert set(fresh_group_versions) == {2}

        router.release_trajectories(trajectory_ids=[hold_trajectory_id])

        await wait_for(
            "policy version 3",
            lambda: train_runtime.policy_version >= 3,
        )
        await wait_for(
            "rollout workers fully synced to version 3",
            lambda: synced_rollout_version(runtime) == 3,
        )
        await wait_for_all_trained(
            store,
            session_id,
            expected_trajectories=18,
        )
        assert [
            (await store.list_turns(trajectory_id))[0].rollout_model_version
            for trajectory_id in stale_group_ids
        ] == [1, 1]
        assert [
            (await store.list_turns(trajectory_id))[0].rollout_model_version
            for trajectory_id in fresh_group_ids
        ] == [2, 2, 2, 2]

        current_workers = [
            {
                "worker_url": str(worker["worker_url"]),
                "world_size": int(worker["world_size"]),
            }
            for worker in router_workers(runtime)
        ]
        distributed_payload = train_client.request_weight_sync(
            workers=current_workers,
            target_version=3,
            mode="distributed",
        )
        assert_all_ok(distributed_payload)

        disk_payload = train_client.request_weight_sync(
            workers=current_workers,
            target_version=3,
            mode="disk",
        )
        assert_all_ok(disk_payload)

        checkpoint = await service.export_checkpoint(session_id=session_id)
        assert checkpoint.policy_version == 3
        assert Path(checkpoint.checkpoint_path).exists()
        keep_temp_dir = True

        await service.end_session(session_id=session_id)

        print("PASS end-to-end async train + rollout live script")
        print(f"cuda_visible_devices={cuda_visible_devices}")
        print(f"train_server_url={train_server_url}")
        print(
            "versions="
            f"{{'a': {group_a_versions}, 'b': {group_b_versions}, "
            f"'c': {group_c_versions}, 'd': {group_d_versions}, "
            f"'stale': {stale_group_versions}, "
            f"'fresh': {fresh_group_versions}}}"
        )
        print(f"independent_update={independent_versions}")
        print(f"hold_worker={hold_worker_name}")
        print(f"checkpoint={checkpoint.checkpoint_path}")
        print(f"artifacts_dir={temp_dir}")
    finally:
        await runtime.shutdown()
        if not keep_temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
