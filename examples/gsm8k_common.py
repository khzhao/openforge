# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import datasets

import openforge.ninja as ninja
from openforge import active_state
from openforge.benchmarks.gsm8k import build_gsm8k_prompt, extract_gsm8k_ground_truth
from openforge.gateway.types import RuntimeConfig

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(slots=True)
class GSM8kExample:
    """One GSM8K training example prepared for prompting."""

    question: str
    answer: str
    prompt: str
    ground_truth: str


def get_free_port(host: str) -> int:
    """Reserve and return one free TCP port on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def make_artifact_dir(path: str | None, *, prefix: str) -> Path:
    """Create or reuse the artifact directory for an example run."""
    if path is not None:
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=ARTIFACT_ROOT))


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load the runtime config from YAML."""
    return RuntimeConfig.from_yaml(path)


def load_train_examples(
    *,
    seed: int,
    max_examples: int | None,
) -> list[GSM8kExample]:
    """Load and prepare shuffled GSM8K training examples."""
    rows = datasets.load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=seed)
    examples: list[GSM8kExample] = []
    for row in rows:
        question = str(row["question"])
        answer = str(row["answer"])
        examples.append(
            GSM8kExample(
                question=question,
                answer=answer,
                prompt=build_gsm8k_prompt(question),
                ground_truth=extract_gsm8k_ground_truth(answer),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a sequence of JSON rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_split(path: Path, examples: list[GSM8kExample]) -> None:
    """Persist a prepared split as JSONL."""
    write_jsonl(path, [asdict(example) for example in examples])


def response_text(response: Any) -> str:
    """Extract the assistant text from a chat-completions response."""
    return str(response.choices[0].message.content)


def save_summary(path: Path, payload: dict[str, object]) -> None:
    """Write a summary payload as formatted JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_train_arg_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser for GSM8K training examples."""
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--total-epochs", type=int, default=15)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--train-top-p", type=float, default=1.0)
    parser.add_argument("--train-top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--train-group-parallelism",
        type=int,
        default=None,
        help="Override Ninja execute parallelism. Defaults to framework auto mode.",
    )
    parser.add_argument("--train-group-retries", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=None)
    return parser


def parse_train_args() -> argparse.Namespace:
    """Parse CLI flags for the shared GSM8K training example."""
    return build_train_arg_parser().parse_args()


def prepare_train_setup(args: argparse.Namespace) -> dict[str, Any]:
    """Load configs, dataset inputs, and metadata for GSM8K training."""
    artifact_dir = make_artifact_dir(args.artifact_dir, prefix="gsm8k-ninja-")
    runtime_config = (
        active_state.load_active_runtime_config()
        if args.runtime_config is None
        else load_runtime_config(args.runtime_config)
    )

    train_examples = load_train_examples(
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    save_split(artifact_dir / "data" / "train.jsonl", train_examples)

    sampling_params = {
        "temperature": args.train_temperature,
        "top_p": args.train_top_p,
        "top_k": args.train_top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": runtime_config.rollout.request.max_new_tokens,
    }
    inputs = [
        {
            "prompt": example.prompt,
            "ground_truth": example.ground_truth,
        }
        for example in train_examples
    ]
    summary = {
        "artifact_dir": str(artifact_dir),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "group_size": args.group_size,
        "max_updates": args.max_updates,
        "model_path": runtime_config.model.model_name_or_path,
        "reward_method": "strict",
        "global_batch_size": runtime_config.train.global_batch_size,
        "slurm_step_gpus": str(os.environ.get("SLURM_STEP_GPUS")),
        "total_epochs": args.total_epochs,
        "train_examples": len(train_examples),
        "train_group_parallelism": (
            "auto"
            if args.train_group_parallelism is None
            else args.train_group_parallelism
        ),
        "train_group_retries": args.train_group_retries,
        "train_sampling": sampling_params,
    }
    return {
        "runtime_config": runtime_config,
        "inputs": inputs,
        "sampling_params": sampling_params,
        "summary": summary,
        "summary_path": artifact_dir / "summary.json",
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
        "final_policy_version": 0 if final_policy_version is None else final_policy_version,
        "last_train_update": last_update,
        "train_updates": train_updates,
        "prompt_groups_per_update": prompt_groups_per_update,
        "global_batch_size": global_batch_size,
        "train_groups": len(schedule),
        "train_groups_consumed": consumed_groups,
        "train_groups_dropped": len(schedule) - consumed_groups,
    }
