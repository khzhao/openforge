# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
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


def require_visible_gpus(min_count: int) -> int:
    """Assert that at least the requested number of GPUs is visible."""
    visible = torch.cuda.device_count()
    assert visible >= min_count, (
        f"Expected at least {min_count} visible GPUs, found {visible}"
    )
    return visible


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


def parse_train_args() -> argparse.Namespace:
    """Parse CLI flags for the shared GSM8K training example."""
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
    parser.add_argument("--train-group-parallelism", type=int, default=32)
    parser.add_argument("--train-group-retries", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=None)
    return parser.parse_args()


def prepare_train_setup(args: argparse.Namespace) -> dict[str, Any]:
    """Load configs, dataset inputs, and metadata for GSM8K training."""
    artifact_dir = make_artifact_dir(args.artifact_dir, prefix="gsm8k-ninja-")
    gateway_config = load_gateway_config(args.gateway_config, artifact_dir=artifact_dir)
    runtime_config = load_runtime_config(args.runtime_config, artifact_dir=artifact_dir)
    total_gpus = runtime_config.train.total_gpus + runtime_config.rollout.total_gpus
    visible_gpus = require_visible_gpus(total_gpus)
    if gateway_config.cluster.total_gpus < visible_gpus:
        gateway_config.cluster.gpus_per_node = visible_gpus

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
        "train_group_parallelism": args.train_group_parallelism,
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
