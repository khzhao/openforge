# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

from loguru import logger

import openforge.ninja as ninja
from openforge import active_state
from openforge.gateway.types import RuntimeConfig

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def make_artifact_dir(path: str | None, *, prefix: str) -> Path:
    """Create or reuse the artifact directory for an example run."""
    if path is not None:
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=ARTIFACT_ROOT))


def save_summary(path: Path, payload: dict[str, object]) -> None:
    """Write a summary payload as formatted JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def add_train_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_group_size: int,
    default_total_epochs: int = 15,
) -> argparse.ArgumentParser:
    """Add the shared OpenForge example training CLI options."""
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument(
        "--runtime-config",
        default=None,
        help=(
            "Optional runtime YAML. If omitted, the script uses the active "
            "session runtime recorded by OpenForge."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-size", type=int, default=default_group_size)
    parser.add_argument("--total-epochs", type=int, default=default_total_epochs)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--wait-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--train-group-parallelism",
        type=int,
        default=None,
        help=(
            "Override Ninja execute parallelism in total in-flight rollouts. "
            "Prompt-group concurrency is roughly this value divided by group_size."
        ),
    )
    parser.add_argument("--train-group-retries", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--validation-every-updates", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    return parser


def load_runtime_config(path: str | None) -> RuntimeConfig:
    """Load the explicit runtime YAML or the active runtime state."""
    if path is None:
        return active_state.load_active_runtime_config()
    return RuntimeConfig.from_yaml(path)


def print_train_update(update: dict[str, object]) -> None:
    """Emit one machine-readable train progress event."""
    logger.info("TRAIN_UPDATE {}", json.dumps(update, sort_keys=True))


def print_validation_update(update: dict[str, object]) -> None:
    """Emit one machine-readable validation progress event."""
    logger.info("VALIDATION_UPDATE {}", json.dumps(update, sort_keys=True))


def plan_train_batches(
    *,
    runtime_config: RuntimeConfig,
    inputs: list[dict[str, Any]],
    group_size: int,
    epochs: int,
    seed: int,
    max_updates: int | None = None,
) -> dict[str, object]:
    """Shuffle prompt groups into one-update batches sized to global_batch_size."""
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if not inputs:
        raise ValueError("inputs must not be empty")

    global_batch_size = int(runtime_config.train.global_batch_size)
    if global_batch_size <= 0:
        raise ValueError("runtime_config.train.global_batch_size must be > 0")
    if global_batch_size % group_size != 0:
        raise ValueError(
            "runtime_config.train.global_batch_size must be divisible by group_size"
        )

    prompt_groups_per_update = global_batch_size // group_size
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for _epoch in range(epochs):
        epoch_inputs = list(inputs)
        rng.shuffle(epoch_inputs)
        schedule.extend(epoch_inputs)

    expected_updates = len(schedule) // prompt_groups_per_update
    if max_updates is not None:
        expected_updates = min(expected_updates, max_updates)
    if expected_updates <= 0:
        raise ValueError(
            "not enough prompt groups to perform one update: "
            f"have {len(schedule)}, need {prompt_groups_per_update}"
        )

    planned_prompt_groups = expected_updates * prompt_groups_per_update
    return {
        "update_inputs": [
            schedule[
                update_index * prompt_groups_per_update : (update_index + 1)
                * prompt_groups_per_update
            ]
            for update_index in range(expected_updates)
        ],
        "expected_updates": expected_updates,
        "prompt_groups_per_update": prompt_groups_per_update,
        "global_batch_size": global_batch_size,
        "train_groups": len(schedule),
        "train_groups_planned": planned_prompt_groups,
        "train_groups_dropped": len(schedule) - planned_prompt_groups,
    }


def run_train(
    agent_func: Any,
    *,
    runtime_config: RuntimeConfig,
    inputs: list[dict[str, Any]],
    group_size: int,
    epochs: int,
    seed: int,
    parallelism: int | None = None,
    retries: int = 0,
    wait_timeout: float = 7200.0,
    max_updates: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run grouped on-policy training for a registered Ninja agent."""
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if parallelism is not None and parallelism <= 0:
        raise ValueError("parallelism must be > 0")
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if not inputs:
        raise ValueError("inputs must not be empty")

    global_batch_size = int(runtime_config.train.global_batch_size)
    if global_batch_size <= 0:
        raise ValueError("runtime_config.train.global_batch_size must be > 0")
    if global_batch_size % group_size != 0:
        raise ValueError(
            "runtime_config.train.global_batch_size must be divisible by group_size"
        )

    prompt_groups_per_update = global_batch_size // group_size
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for _epoch in range(epochs):
        epoch_inputs = list(inputs)
        rng.shuffle(epoch_inputs)
        schedule.extend(epoch_inputs)

    available_updates = len(schedule) // prompt_groups_per_update
    if max_updates is not None:
        available_updates = min(available_updates, max_updates)
    if available_updates <= 0:
        raise ValueError(
            "not enough prompt groups to perform one update: "
            f"have {len(schedule)}, need {prompt_groups_per_update}"
        )

    final_policy_version: int | None = None
    last_update: dict[str, Any] | None = None
    train_updates: list[dict[str, Any]] = []
    consumed_groups = 0
    failed_prompt_groups_total = 0
    stopped_early_reason: str | None = None
    resolved_concurrency = ninja._resolve_concurrency(
        concurrency=parallelism,
        job_count=prompt_groups_per_update * group_size,
    )
    with agent_func._session() as session:
        final_policy_version = int(session.current_train_policy_version())
        for update_offset in range(available_updates):
            update_results: list[ninja._GroupedExecutionResult] = []
            update_failed_groups = 0
            update_start_consumed = consumed_groups
            while len(update_results) < prompt_groups_per_update:
                remaining_groups = prompt_groups_per_update - len(update_results)
                batch_inputs = schedule[
                    consumed_groups : consumed_groups + remaining_groups
                ]
                if not batch_inputs:
                    break
                consumed_groups += len(batch_inputs)
                call_specs, _single_request = ninja._normalize_requests(
                    args=(),
                    kwargs={},
                    requests=batch_inputs,
                )
                for request_index, (call_args, call_kwargs) in enumerate(call_specs):
                    agent_func._validate_call(
                        call_args,
                        call_kwargs,
                        request_index=request_index,
                    )
                grouped_results, failures = ninja._execute_grouped_results(
                    agent_func,
                    session,
                    call_specs,
                    group_size=group_size,
                    concurrency=resolved_concurrency,
                    retries=retries,
                    raise_on_failure=False,
                )
                update_results.extend(grouped_results)
                update_failed_groups += len(failures)
                failed_prompt_groups_total += len(failures)
                if (
                    consumed_groups >= len(schedule)
                    and len(update_results) < prompt_groups_per_update
                ):
                    break

            if len(update_results) < prompt_groups_per_update:
                stopped_early_reason = (
                    "insufficient successful prompt groups to complete another update"
                )
                break

            trajectory_ids = [
                trajectory_id
                for result in update_results
                for trajectory_id in result.trajectory_ids
            ]
            ninja._wait_for_trained_trajectories(
                session,
                trajectory_ids,
                timeout=wait_timeout,
            )
            final_policy_version = int(session.current_train_policy_version())
            rewards = [reward for result in update_results for reward in result.rewards]
            group_mean_rewards = [
                sum(result.rewards) / len(result.rewards) for result in update_results
            ]
            last_update = {
                "policy_version": final_policy_version,
                "prompt_groups": len(update_results),
                "samples": len(rewards),
                "max_group_reward": max(
                    max(result.rewards) for result in update_results
                ),
                "mean_group_reward": sum(group_mean_rewards) / len(group_mean_rewards),
                "sample_mean_reward": sum(rewards) / len(rewards),
                "failed_prompt_groups": update_failed_groups,
                "attempted_prompt_groups": consumed_groups - update_start_consumed,
                "update_index": update_offset + 1,
            }
            train_updates.append(dict(last_update))
            if progress_callback is not None:
                progress_callback(dict(last_update))

    return {
        "completed_updates": len(train_updates),
        "expected_updates": available_updates,
        "final_checkpoint": agent_func.save(),
        "final_policy_version": (
            0 if final_policy_version is None else final_policy_version
        ),
        "last_train_update": last_update,
        "failed_prompt_groups": failed_prompt_groups_total,
        "stopped_early_reason": stopped_early_reason,
        "train_updates": train_updates,
        "prompt_groups_per_update": prompt_groups_per_update,
        "global_batch_size": global_batch_size,
        "train_groups": len(schedule),
        "train_groups_consumed": consumed_groups,
        "train_groups_dropped": len(schedule) - consumed_groups,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "slurm_step_gpus": str(os.environ.get("SLURM_STEP_GPUS")),
    }
