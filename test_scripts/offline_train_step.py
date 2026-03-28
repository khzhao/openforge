#!/usr/bin/env python3
# Copyright 2026 openforge

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import ray
import yaml
from ray.util.placement_group import placement_group

from openforge.configs.models import GatewayServerConfig, OpenForgeConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.types import RuntimeConfig
from openforge.train.loop import TrainLoop
from openforge.train.manager import TrainManager
from openforge.utils.networking import get_free_port, get_host_ip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated OpenForge train update from an existing DB.",
    )
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--train-gpus", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--mini-batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--max-tokens-per-micro-batch", type=int, default=None)
    parser.add_argument(
        "--reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def load_config(
    *,
    gateway_config_path: str,
    runtime_config_path: str,
    train_gpus: int,
    global_batch_size: int | None,
    mini_batch_size: int | None,
    micro_batch_size: int | None,
    max_tokens_per_micro_batch: int | None,
    reshard_after_forward: bool | None,
) -> OpenForgeConfig:
    gateway_cfg = GatewayServerConfig.from_yaml(gateway_config_path)
    runtime_cfg = RuntimeConfig.from_yaml(runtime_config_path)
    runtime_cfg.train.parallel.data_parallel_size = 1
    runtime_cfg.train.parallel.fsdp_parallel_size = train_gpus
    if global_batch_size is not None:
        runtime_cfg.train.global_batch_size = global_batch_size
    if mini_batch_size is not None:
        runtime_cfg.train.mini_batch_size = mini_batch_size
    if micro_batch_size is not None:
        runtime_cfg.train.micro_batch_size = micro_batch_size
    if max_tokens_per_micro_batch is not None:
        runtime_cfg.train.max_tokens_per_micro_batch = max_tokens_per_micro_batch
    if reshard_after_forward is not None:
        runtime_cfg.train.config.reshard_after_forward = reshard_after_forward

    world_size = runtime_cfg.train.parallel.world_size
    if runtime_cfg.train.global_batch_size % world_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by train world size for the current "
            f"train loop: {runtime_cfg.train.global_batch_size} % {world_size} != 0"
        )
    if runtime_cfg.train.global_batch_size % (
        world_size * runtime_cfg.train.mini_batch_size
    ) != 0:
        raise ValueError(
            "global_batch_size must be divisible by world_size * mini_batch_size: "
            f"{runtime_cfg.train.global_batch_size} % "
            f"({world_size} * {runtime_cfg.train.mini_batch_size}) != 0"
        )

    # This offline isolate only exercises the train path, so rollout/cluster
    # capacity validation would be misleading here.
    return OpenForgeConfig.model_construct(
        data=gateway_cfg.data,
        gateway=gateway_cfg.gateway,
        cluster=gateway_cfg.cluster,
        algo=runtime_cfg.algo,
        model=runtime_cfg.model,
        train=runtime_cfg.train,
        rollout=runtime_cfg.rollout,
    )


async def run_step(args: argparse.Namespace) -> int:
    cfg = load_config(
        gateway_config_path=args.gateway_config,
        runtime_config_path=args.runtime_config,
        train_gpus=args.train_gpus,
        global_batch_size=args.global_batch_size,
        mini_batch_size=args.mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_tokens_per_micro_batch=args.max_tokens_per_micro_batch,
        reshard_after_forward=args.reshard_after_forward,
    )

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    session_id = args.session_id

    workdir = Path(tempfile.mkdtemp(prefix="openforge-offline-train-step-"))
    db_copy = workdir / "gateway.sqlite3"
    src_conn = sqlite3.connect(db_path)
    dst_conn = sqlite3.connect(db_copy)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    print(f"workdir={workdir}")
    print(f"db_copy={db_copy}")
    print(f"session_id={session_id}")
    print(
        "train_cfg="
        f"world_size={cfg.train.parallel.world_size} "
        f"mini_batch_size={cfg.train.mini_batch_size} "
        f"micro_batch_size={cfg.train.micro_batch_size} "
        f"max_tokens_per_micro_batch={cfg.train.max_tokens_per_micro_batch} "
        f"reshard_after_forward={cfg.train.config.reshard_after_forward}"
    )

    store = SQLiteOpenForgeStore(db_copy)
    manager = TrainManager()
    started_ray = False
    try:
        if not ray.is_initialized():
            ray.init(
                address="local",
                num_cpus=max(cfg.train.total_cpus, args.train_gpus),
                num_gpus=args.train_gpus,
                log_to_driver=False,
            )
            started_ray = True

        bundles = [
            {"GPU": 1, "CPU": max(cfg.train.cpus_per_worker, 1)}
            for _ in range(args.train_gpus)
        ]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        manager.initialize(
            cfg,
            master_addr=get_host_ip(),
            master_port=get_free_port(start=20000),
            placement_group={
                "actor": (
                    pg,
                    list(range(args.train_gpus)),
                    list(range(args.train_gpus)),
                )
            },
        )

        loop = TrainLoop(
            session_id=session_id,
            store=store,
            train_manager=manager,
        )
        started = time.monotonic()
        trained = await loop.train_once()
        elapsed = time.monotonic() - started

        statuses = {}
        for trajectory in await store.list_trajectories(session_id):
            statuses[trajectory.status] = statuses.get(trajectory.status, 0) + 1

        print(f"trained={trained}")
        print(f"elapsed_s={elapsed:.2f}")
        print(f"global_step={loop.global_step}")
        print(f"policy_version={loop.policy_version}")
        print(f"statuses={statuses}")
        return 0 if trained else 2
    finally:
        try:
            await store.close()
        finally:
            if manager.workers:
                manager.shutdown()
            if started_ray and ray.is_initialized():
                ray.shutdown()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_step(args))


if __name__ == "__main__":
    raise SystemExit(main())
