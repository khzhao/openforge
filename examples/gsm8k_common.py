# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import datasets
import httpx
import yaml

from openforge.benchmarks.gsm8k import build_gsm8k_prompt, extract_gsm8k_ground_truth
from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import RuntimeConfig

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"
GATEWAY_CONFIG_PATH = Path(__file__).with_name("gsm8k_gateway.yaml")
RUNTIME_CONFIG_PATH = Path(__file__).with_name("gsm8k_runtime.yaml")


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


def load_gateway_config(
    path: str | Path,
    *,
    artifact_dir: Path,
) -> GatewayServerConfig:
    """Load the gateway config and relocate its sqlite path into artifacts."""
    cfg = GatewayServerConfig.from_yaml(path)
    cfg.data.path = str(artifact_dir / "gateway.sqlite3")
    return cfg


def load_runtime_config(
    path: str | Path,
    *,
    artifact_dir: Path,
) -> RuntimeConfig:
    """Load the runtime config and relocate checkpoint output into artifacts."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["train"]["checkpoints"] = str(artifact_dir / "checkpoints")
    return RuntimeConfig.model_validate(payload)


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


def response_text(response: dict[str, object]) -> str:
    """Extract the assistant text from a Ninja generate response."""
    return str(response["choices"][0]["message"]["content"])


def save_summary(path: Path, payload: dict[str, object]) -> None:
    """Write a summary payload as formatted JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_train_arg_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser for GSM8K training examples."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--gateway-config", default=str(GATEWAY_CONFIG_PATH))
    parser.add_argument("--runtime-config", default=str(RUNTIME_CONFIG_PATH))
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
    gateway_config = load_gateway_config(args.gateway_config, artifact_dir=artifact_dir)
    runtime_config = load_runtime_config(args.runtime_config, artifact_dir=artifact_dir)

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
        "samples_per_update": runtime_config.train.global_batch_size,
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
        "gateway_config": gateway_config,
        "runtime_config": runtime_config,
        "inputs": inputs,
        "sampling_params": sampling_params,
        "summary": summary,
        "summary_path": artifact_dir / "summary.json",
    }


@contextmanager
def _active_session(
    gateway_config: GatewayServerConfig,
) -> Any:
    base_url = f"http://{gateway_config.gateway.host}:{gateway_config.gateway.port}"
    client = httpx.Client(base_url=base_url, timeout=300.0)
    try:
        response = client.get("/current_session")
        if response.status_code == 404:
            raise AssertionError("no active session")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        yield client, str(payload["session_id"])
    finally:
        client.close()


def _parse_policy_version(value: object) -> int:
    text = str(value)
    return int(text) if text.isdigit() else 0


def _wait_for_rollout_policy_version(
    gateway_config: GatewayServerConfig,
    *,
    target_version: int,
    timeout: float,
) -> int:
    deadline = time.monotonic() + timeout
    probe_sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 8,
    }
    while time.monotonic() < deadline:
        with _active_session(gateway_config) as (client, session_id):
            trajectory_id = f"traj_{uuid4().hex}"
            response = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "group_id": None,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "sampling_params": probe_sampling,
                },
            )
            response.raise_for_status()
            payload = response.json()
            discard = client.post(
                "/discard_trajectory",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                },
            )
            discard.raise_for_status()
        version = _parse_policy_version(
            payload.get("metadata", {}).get("rollout_model_version")
        )
        if version >= target_version:
            return version
        time.sleep(1.0)
    raise TimeoutError(
        f"rollout policy_version did not reach {target_version} within {timeout} seconds"
    )


def run_train(
    agent_func: Any,
    *,
    gateway_config: GatewayServerConfig,
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

    samples_per_update = int(runtime_config.train.global_batch_size)
    if samples_per_update <= 0:
        raise ValueError("runtime_config.train.global_batch_size must be > 0")
    if samples_per_update % group_size != 0:
        raise ValueError(
            "runtime_config.train.global_batch_size must be divisible by group_size"
        )

    prompt_groups_per_update = samples_per_update // group_size
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

    initial_policy_version = agent_func.policy_version()
    final_policy_version = initial_policy_version
    last_update: dict[str, Any] | None = None
    train_updates: list[dict[str, Any]] = []
    consumed_groups = 0
    for update_offset in range(available_updates):
        batch_inputs = schedule[
            consumed_groups : consumed_groups + prompt_groups_per_update
        ]
        reward_groups = agent_func.sample(
            requests=batch_inputs,
            num_rollouts=group_size,
            concurrency=parallelism,
            retries=retries,
        )
        if group_size == 1:
            reward_groups = [[float(reward)] for reward in reward_groups]

        group_results = [
            {
                "group_index": consumed_groups + index + 1,
                "group_size": len(rewards),
                "max_reward": max(rewards),
                "mean_reward": sum(rewards) / len(rewards),
                "rewards": rewards,
            }
            for index, rewards in enumerate(reward_groups)
        ]
        consumed_groups += prompt_groups_per_update
        final_policy_version = _wait_for_rollout_policy_version(
            gateway_config,
            target_version=initial_policy_version + update_offset + 1,
            timeout=wait_timeout,
        )
        rewards = [
            reward
            for group in group_results
            for reward in group["rewards"]
        ]
        last_update = {
            "policy_version": final_policy_version,
            "prompt_groups": len(group_results),
            "samples": len(rewards),
            "max_group_reward": max(group["max_reward"] for group in group_results),
            "mean_group_reward": sum(
                group["mean_reward"] for group in group_results
            )
            / len(group_results),
            "sample_mean_reward": sum(rewards) / len(rewards),
            "update_index": update_offset + 1,
        }
        train_updates.append(dict(last_update))
        if progress_callback is not None:
            progress_callback(dict(last_update))

    return {
        "completed_updates": available_updates,
        "expected_updates": available_updates,
        "final_checkpoint": agent_func.save(),
        "final_policy_version": final_policy_version,
        "last_train_update": last_update,
        "train_updates": train_updates,
        "prompt_groups_per_update": prompt_groups_per_update,
        "samples_per_update": samples_per_update,
        "train_groups": len(schedule),
        "train_groups_consumed": consumed_groups,
        "train_groups_dropped": len(schedule) - consumed_groups,
    }
