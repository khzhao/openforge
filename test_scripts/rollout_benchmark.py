#!/usr/bin/env python3
# Copyright 2026 openforge

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import ray
import yaml
from ray.util.placement_group import placement_group

from openforge.configs.models import GatewayServerConfig, OpenForgeConfig
from openforge.gateway.types import RuntimeConfig
from openforge.rollout.manager import RolloutManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark rollout throughput on saved prompt prefixes.",
    )
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--rollout-gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    return parser.parse_args()


def load_prompts(*, db_path: str, session_id: str, limit: int) -> list[tuple[str, list[int]]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT t.trajectory_id, t.prompt_length, t.token_ids_json
            FROM turns AS t
            JOIN trajectories AS tr
            ON tr.trajectory_id = t.trajectory_id
            WHERE tr.session_id = ?
              AND t.turn_index = 0
            ORDER BY t.trajectory_id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()

    prompts: list[tuple[str, list[int]]] = []
    import json

    for trajectory_id, prompt_length, token_ids_json in rows:
        token_ids = list(json.loads(token_ids_json))
        prompts.append((str(trajectory_id), token_ids[: int(prompt_length)]))
    if len(prompts) != limit:
        raise ValueError(f"expected {limit} prompts, got {len(prompts)}")
    return prompts


def load_config(
    *,
    gateway_config_path: str,
    runtime_config_path: str,
    rollout_gpus: int,
) -> OpenForgeConfig:
    gateway_cfg = GatewayServerConfig.from_yaml(gateway_config_path)
    runtime_cfg = RuntimeConfig.from_yaml(runtime_config_path)
    runtime_cfg.rollout.engine_groups[0].replicas = rollout_gpus

    return OpenForgeConfig.model_construct(
        data=gateway_cfg.data,
        gateway=gateway_cfg.gateway,
        cluster=gateway_cfg.cluster,
        algo=runtime_cfg.algo,
        model=runtime_cfg.model,
        train=runtime_cfg.train,
        rollout=runtime_cfg.rollout,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(
        gateway_config_path=args.gateway_config,
        runtime_config_path=args.runtime_config,
        rollout_gpus=args.rollout_gpus,
    )
    prompts = load_prompts(
        db_path=args.db,
        session_id=args.session_id,
        limit=args.num_prompts,
    )

    started_ray = False
    manager: RolloutManager | None = None
    try:
        if not ray.is_initialized():
            ray.init(
                address="local",
                num_cpus=max(args.rollout_gpus, 1),
                num_gpus=args.rollout_gpus,
                log_to_driver=False,
            )
            started_ray = True

        pg = placement_group(
            [{"GPU": 1, "CPU": 1} for _ in range(args.rollout_gpus)],
            strategy="PACK",
        )
        ray.get(pg.ready())
        placement_groups = {
            "rollout": (
                pg,
                list(range(args.rollout_gpus)),
                list(range(args.rollout_gpus)),
            )
        }
        manager = RolloutManager(cfg, placement_groups)
        manager.initialize(log_level="warning")

        sampling_params = cfg.rollout.request.model_dump()
        trajectory_ids = [trajectory_id for trajectory_id, _ in prompts]
        input_ids = [prompt_ids for _trajectory_id, prompt_ids in prompts]

        started = time.monotonic()
        payloads = manager.router.generate(
            sampling_params,
            trajectory_ids=trajectory_ids,
            input_ids=input_ids,
        )
        elapsed = time.monotonic() - started

        if isinstance(payloads, dict):
            payloads = [payloads]
        output_lengths: list[int] = []
        for payload in payloads:
            token_ids = payload.get("output_ids")
            if not isinstance(token_ids, list):
                token_ids = payload.get("token_ids")
            if not isinstance(token_ids, list):
                raise ValueError("generate payload missing output_ids or token_ids")
            output_lengths.append(len(token_ids))
        total_output_tokens = sum(output_lengths)
        print(f"rollout_gpus={args.rollout_gpus}")
        print(f"num_prompts={args.num_prompts}")
        print(f"elapsed_s={elapsed:.2f}")
        print(f"total_output_tokens={total_output_tokens}")
        print(f"tokens_per_s={total_output_tokens / elapsed:.2f}")
        print(
            f"mean_output_tokens={total_output_tokens / max(len(output_lengths), 1):.2f}"
        )
        return 0
    finally:
        if manager is not None:
            manager.shutdown()
        if started_ray and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
