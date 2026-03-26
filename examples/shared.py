# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

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
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--train-group-parallelism",
        type=int,
        default=None,
        help="Override Ninja execute parallelism. Defaults to framework auto mode.",
    )
    parser.add_argument("--train-group-retries", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=None)
    return parser


def load_runtime_config(path: str | None) -> RuntimeConfig:
    """Load the explicit runtime YAML or the active runtime state."""
    if path is None:
        return active_state.load_active_runtime_config()
    return RuntimeConfig.from_yaml(path)


def print_train_update(update: dict[str, object]) -> None:
    """Emit one machine-readable train progress event."""
    print(
        "TRAIN_UPDATE",
        json.dumps(update, sort_keys=True),
        flush=True,
    )


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
    wait_timeout: float = 1800.0,
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
    for update_offset in range(available_updates):
        batch_inputs = schedule[
            consumed_groups : consumed_groups + prompt_groups_per_update
        ]
        train_summary = ninja.train(
            agent_func,
            inputs=batch_inputs,
            group_size=group_size,
            concurrency=parallelism,
            retries=retries,
            wait_timeout=wait_timeout,
        )
        final_policy_version = int(train_summary["final_policy_version"])
        consumed_groups += prompt_groups_per_update
        last_update = {
            "policy_version": final_policy_version,
            "prompt_groups": int(train_summary["prompt_groups"]),
            "samples": int(train_summary["samples"]),
            "max_group_reward": float(train_summary["max_group_reward"]),
            "mean_group_reward": float(train_summary["mean_group_reward"]),
            "sample_mean_reward": float(train_summary["sample_mean_reward"]),
            "update_index": update_offset + 1,
        }
        train_updates.append(dict(last_update))
        if progress_callback is not None:
            progress_callback(dict(last_update))

    return {
        "completed_updates": available_updates,
        "expected_updates": available_updates,
        "final_checkpoint": agent_func.save(),
        "final_policy_version": (
            0 if final_policy_version is None else final_policy_version
        ),
        "last_train_update": last_update,
        "train_updates": train_updates,
        "prompt_groups_per_update": prompt_groups_per_update,
        "global_batch_size": global_batch_size,
        "train_groups": len(schedule),
        "train_groups_consumed": consumed_groups,
        "train_groups_dropped": len(schedule) - consumed_groups,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "slurm_step_gpus": str(os.environ.get("SLURM_STEP_GPUS")),
    }
