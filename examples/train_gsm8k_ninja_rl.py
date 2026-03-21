# Copyright 2026 openforge

"""Train a GSM8k GRPO agent through the openforge.ninja API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
from huggingface_hub import snapshot_download

from openforge.benchmarks.gsm8k import (
    build_gsm8k_prompt,
    compute_gsm8k_score,
    extract_gsm8k_ground_truth,
    extract_gsm8k_solution,
)
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.types import RuntimeConfig
from openforge.ninja import Gateway, Session

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"


@dataclass(slots=True)
class GSM8kExample:
    split: str
    index: int
    question: str
    answer: str
    prompt: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpus-per-node", type=int, default=32)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=3)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-benchmark-examples", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--train-top-p", type=float, default=1.0)
    parser.add_argument("--train-top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--eval-parallelism", type=int, default=4)
    parser.add_argument("--train-group-parallelism", type=int, default=1)
    parser.add_argument("--train-group-retries", type=int, default=2)
    parser.add_argument("--ppo-mini-batch-size-prompts", type=int, default=64)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--kl-coef", type=float, default=1.0e-3)
    return parser.parse_args()


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
    try:
        return snapshot_download(repo_id=model_path_or_id, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=model_path_or_id)


def get_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def make_artifact_dir(path: str | None) -> Path:
    if path is not None:
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gsm8k-ninja-rl-", dir=ARTIFACT_ROOT))


def build_gateway_config(
    *,
    artifact_dir: Path,
    gateway_host: str,
    gateway_port: int,
    visible_gpus: int,
    cpus_per_node: int,
) -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=str(artifact_dir / "gateway.sqlite3")),
        gateway=GatewayConfig(host=gateway_host, port=gateway_port),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=visible_gpus,
            cpus_per_node=cpus_per_node,
        ),
    )


def build_runtime_config(
    *,
    model_path: str,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    checkpoint_root: Path,
    train_batch_size: int,
    group_size: int,
    ppo_mini_batch_size_prompts: int,
    ppo_micro_batch_size_per_gpu: int,
    ppo_epochs: int,
    max_new_tokens: int,
    learning_rate: float,
    kl_coef: float,
) -> RuntimeConfig:
    samples_per_update = train_batch_size * group_size
    mini_batch_size = ppo_mini_batch_size_prompts * group_size
    if mini_batch_size % train_gpus != 0:
        raise ValueError(
            "ppo_mini_batch_size_prompts * group_size must be divisible by train_gpus"
        )
    mini_batch_size //= train_gpus
    if samples_per_update % (mini_batch_size * train_gpus) != 0:
        raise ValueError(
            "train_batch_size must be divisible by ppo_mini_batch_size_prompts"
        )
    if mini_batch_size % ppo_micro_batch_size_per_gpu != 0:
        raise ValueError(
            "mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu"
        )
    return RuntimeConfig.model_validate(
        {
            "algo": {
                "name": "grpo",
                "clip_range": 0.2,
                "normalize_group_std": True,
                "entropy_coef": 0.0,
                "kl_coef": kl_coef,
            },
            "model": {
                "model_name_or_path": model_path,
                "reference_model_name_or_path": model_path,
                "tokenizer_name_or_path": model_path,
                "attn_implementation": "sdpa",
            },
            "train": {
                "backend": "fsdp2",
                "config": {
                    "gradient_checkpointing": True,
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
                        "lr": learning_rate,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1.0e-8,
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
                "global_batch_size": samples_per_update,
                "mini_batch_size": mini_batch_size,
                "micro_batch_size": ppo_micro_batch_size_per_gpu,
                "ppo_epochs": ppo_epochs,
                "checkpoints": str(checkpoint_root),
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
                    "repetition_penalty": 1.0,
                    "max_new_tokens": max_new_tokens,
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
                        "num_cpus_per_replica": 1,
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


def build_examples(rows, *, split: str) -> list[GSM8kExample]:
    examples: list[GSM8kExample] = []
    for index, row in enumerate(rows):
        question = str(row["question"])
        answer = str(row["answer"])
        examples.append(
            GSM8kExample(
                split=split,
                index=index,
                question=question,
                answer=answer,
                prompt=build_gsm8k_prompt(question),
                ground_truth=extract_gsm8k_ground_truth(answer),
            )
        )
    return examples


def maybe_limit(examples: list[GSM8kExample], limit: int | None) -> list[GSM8kExample]:
    if limit is None or limit >= len(examples):
        return list(examples)
    return list(examples[:limit])


def build_train_schedule(
    train_examples: list[GSM8kExample],
    *,
    total_epochs: int,
    seed: int,
) -> list[tuple[int, GSM8kExample]]:
    schedule: list[tuple[int, GSM8kExample]] = []
    rng = random.Random(seed)
    for epoch in range(total_epochs):
        epoch_examples = list(train_examples)
        rng.shuffle(epoch_examples)
        schedule.extend((epoch, example) for example in epoch_examples)
    return schedule


def load_gsm8k_splits(
    *,
    seed: int,
    val_size: int,
    max_train_examples: int | None,
    max_val_examples: int | None,
    max_benchmark_examples: int | None,
) -> tuple[list[GSM8kExample], list[GSM8kExample], list[GSM8kExample]]:
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    split = dataset["train"].train_test_split(test_size=val_size, seed=seed)
    train_rows = split["train"].shuffle(seed=seed)
    val_rows = split["test"].shuffle(seed=seed)
    benchmark_rows = dataset["test"].shuffle(seed=seed)
    train = maybe_limit(build_examples(train_rows, split="train"), max_train_examples)
    val = maybe_limit(build_examples(val_rows, split="val"), max_val_examples)
    benchmark = maybe_limit(
        build_examples(benchmark_rows, split="benchmark"),
        max_benchmark_examples,
    )
    return train, val, benchmark


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_split(path: Path, examples: list[GSM8kExample]) -> None:
    write_jsonl(path, [asdict(example) for example in examples])


def response_text(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def response_version(response: dict[str, Any]) -> str:
    return str(response["metadata"]["rollout_model_version"])


def parse_policy_version(version: str) -> int:
    if version.isdigit():
        return int(version)
    return 0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    append_jsonl_many(path, [row])


def append_jsonl_many(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def collect_update_groups(
    session: Session,
    *,
    group_specs: list[tuple[int, int, GSM8kExample]],
    group_size: int,
    sampling_params: dict[str, Any],
    sampling_seed_base: int | None,
    log_path: Path,
    parallelism: int,
    retries: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    groups_per_request = max(1, parallelism)
    for start in range(0, len(group_specs), groups_per_request):
        batch_specs = group_specs[start : start + groups_per_request]
        results.extend(
            run_training_group_batch(
                session,
                group_specs=batch_specs,
                group_size=group_size,
                sampling_params=sampling_params,
                sampling_seed_base=sampling_seed_base,
                log_path=log_path,
                max_retries=retries,
            )
        )

    results.sort(key=lambda row: int(row["group_index"]))
    return results


def summarize_train_update(
    *,
    update_index: int,
    policy_version: int,
    groups: list[dict[str, Any]],
    collection_seconds: float | None = None,
    policy_sync_seconds: float | None = None,
) -> dict[str, Any]:
    records = [record for group in groups for record in group["records"]]
    rewards = [float(record["reward"]) for record in records]
    strict_format = sum(1 for record in records if record["prediction"] is not None)
    result = {
        "metric": "train_update",
        "policy_version": policy_version,
        "result": {
            "groups": len(groups),
            "max_group_reward": max(float(group["max_reward"]) for group in groups),
            "mean_group_reward": sum(float(group["mean_reward"]) for group in groups)
            / len(groups),
            "nonzero_rollouts": sum(1 for reward in rewards if reward > 0.0),
            "samples": len(records),
            "sample_mean_reward": sum(rewards) / len(rewards),
            "strict_format_rollouts": strict_format,
            "strict_format_rate": strict_format / len(records),
        },
        "step": update_index,
    }
    if collection_seconds is not None:
        result["result"]["update_collection_seconds"] = collection_seconds
        result["result"]["prompt_groups_per_second"] = len(groups) / max(
            collection_seconds, 1.0e-9
        )
        result["result"]["rollouts_per_second"] = len(records) / max(
            collection_seconds, 1.0e-9
        )
    if policy_sync_seconds is not None:
        result["result"]["policy_sync_seconds"] = policy_sync_seconds
    return result


def run_training_group_batch(
    session: Session,
    *,
    group_specs: list[tuple[int, int, GSM8kExample]],
    group_size: int,
    sampling_params: dict[str, Any],
    sampling_seed_base: int | None,
    log_path: Path,
    max_retries: int,
) -> list[dict[str, Any]]:
    del sampling_seed_base
    if not group_specs:
        return []

    first_group_index = group_specs[0][0]
    last_group_index = group_specs[-1][0]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        groups: list[dict[str, Any]] = []
        all_children = []
        finished_children: set[str] = set()
        try:
            group_ids = [f"train-group-{group_index}" for group_index, _, _ in group_specs]
            child_groups = session.trajectory_groups(
                counts=[group_size for _ in group_specs],
                group_ids=group_ids,
            )
            for (group_index, epoch, example), group_id, children in zip(
                group_specs,
                group_ids,
                child_groups,
                strict=True,
            ):
                groups.append(
                    {
                        "children": children,
                        "epoch": epoch,
                        "example": example,
                        "group_id": group_id,
                        "group_index": group_index,
                    }
                )
                all_children.extend(children)

            generate_jobs = [
                (
                    child,
                    [{"role": "user", "content": str(group["example"].prompt)}],
                )
                for group in groups
                for child in group["children"]
            ]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(generate_jobs))
            ) as executor:
                responses = list(
                    executor.map(
                        lambda job: job[0].generate(
                            job[1],
                            sampling_params=sampling_params,
                        ),
                        generate_jobs,
                    )
                )

            results: list[dict[str, Any]] = []
            rows_to_log: list[dict[str, Any]] = []
            all_rewards: list[float] = []
            response_offset = 0
            for group in groups:
                example = group["example"]
                children = group["children"]
                group_records: list[dict[str, Any]] = []
                group_responses = responses[
                    response_offset : response_offset + len(children)
                ]
                response_offset += len(children)
                for response, child in zip(
                    group_responses,
                    children,
                    strict=True,
                ):
                    text = response_text(response)
                    reward = compute_gsm8k_score(
                        text,
                        example.ground_truth,
                        method="strict",
                        format_score=0.0,
                    )
                    all_rewards.append(float(reward))
                    group_records.append(
                        {
                            "prediction": extract_gsm8k_solution(
                                text,
                                method="strict",
                            ),
                            "reward": float(reward),
                            "rollout_model_version": response_version(response),
                            "trajectory_id": child.trajectory_id,
                        }
                    )
                group_records.sort(key=lambda record: str(record["trajectory_id"]))
                nonzero_rollouts = sum(
                    1 for record in group_records if float(record["reward"]) > 0.0
                )
                strict_format_rollouts = sum(
                    1
                    for record in group_records
                    if record["prediction"] is not None
                )
                results.append(
                    {
                        "group_size": group_size,
                        "max_reward": max(record["reward"] for record in group_records),
                        "mean_reward": sum(
                            record["reward"] for record in group_records
                        )
                        / len(group_records),
                        "epoch": group["epoch"],
                        "group_id": group["group_id"],
                        "group_index": group["group_index"],
                        "records": group_records,
                        "split": example.split,
                        "step_index": example.index,
                    }
                )
                rows_to_log.append(
                    {
                        "epoch": group["epoch"],
                        "group_id": group["group_id"],
                        "group_index": group["group_index"],
                        "example_index": example.index,
                        "group_size": group_size,
                        "max_reward": max(
                            record["reward"] for record in group_records
                        ),
                        "mean_reward": sum(
                            record["reward"] for record in group_records
                        )
                        / len(group_records),
                        "nonzero_rollouts": nonzero_rollouts,
                        "reward_sum": sum(
                            float(record["reward"]) for record in group_records
                        ),
                        "rewards": [
                            float(record["reward"]) for record in group_records
                        ],
                        "split": example.split,
                        "strict_format_rollouts": strict_format_rollouts,
                        "trajectory_ids": [
                            str(record["trajectory_id"]) for record in group_records
                        ],
                    }
                )

            session.end_clients(all_children, rewards=all_rewards)
            finished_children.update(child.trajectory_id for child in all_children)
            append_jsonl_many(log_path, rows_to_log)
            return results
        except Exception as exc:
            last_error = exc
            for child in all_children:
                if child.trajectory_id in finished_children:
                    child.close()
            pending_children = [
                child
                for child in all_children
                if child.trajectory_id not in finished_children
            ]
            if pending_children:
                try:
                    session.fail_clients(pending_children)
                except Exception:
                    for child in pending_children:
                        try:
                            child.fail()
                        except Exception:
                            pass
            if attempt >= max_retries:
                raise
            print(
                f"Retrying train groups {first_group_index}-{last_group_index} "
                f"after attempt {attempt + 1} failed: {exc!r}",
                flush=True,
            )
            time.sleep(min(2.0, 0.5 * (2**attempt)))

    assert last_error is not None
    raise last_error


def run_eval(
    session: Session,
    *,
    examples: list[GSM8kExample],
    sampling_params: dict[str, Any],
    split_name: str,
    log_path: Path,
    parallelism: int = 1,
) -> dict[str, Any]:
    def eval_one(example: GSM8kExample) -> dict[str, Any]:
        client = session.client()
        try:
            response = client.generate(
                [{"role": "user", "content": example.prompt}],
                sampling_params=sampling_params,
            )
        finally:
            # Eval trajectories should not count as training samples or failures.
            client.discard()
        text = response_text(response)
        reward = compute_gsm8k_score(text, example.ground_truth, method="strict")
        prediction = extract_gsm8k_solution(text, method="strict")
        return {
            "example_index": example.index,
            "ground_truth": example.ground_truth,
            "prediction": prediction,
            "question": example.question,
            "reward": reward,
            "rollout_model_version": response_version(response),
            "split": split_name,
            "text": text,
        }

    rows: list[dict[str, Any]] = []
    worker_count = max(1, min(parallelism, len(examples)))
    if worker_count == 1:
        rows = [eval_one(example) for example in examples]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [executor.submit(eval_one, example) for example in examples]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: int(row["example_index"]))
    for row in rows:
        append_jsonl(log_path, row)

    correct = sum(1 for row in rows if float(row["reward"]) >= 1.0)
    total = len(rows)
    versions = [
        parse_policy_version(str(row["rollout_model_version"])) for row in rows
    ]
    accuracy = 0.0 if total == 0 else correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "examples": total,
        "max_policy_version": max(versions, default=0),
        "split": split_name,
    }


def wait_for_policy_version(
    session: Session,
    *,
    target_version: int,
    timeout: float,
) -> int:
    deadline = time.monotonic() + timeout
    probe_sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 16,
    }
    while time.monotonic() < deadline:
        client = session.client()
        try:
            response = client.generate(
                [
                    {
                        "role": "user",
                        "content": build_gsm8k_prompt("What is 1 + 1?"),
                    }
                ],
                sampling_params=probe_sampling,
            )
        finally:
            # Version probes are control traffic, not failed rollouts.
            client.discard()
        version = parse_policy_version(response_version(response))
        if version >= target_version:
            return version
        time.sleep(1.0)
    raise TimeoutError(
        f"rollout policy_version did not reach {target_version} within {timeout} seconds"
    )


def save_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_path != DEFAULT_MODEL:
        raise ValueError(
            f"This recipe is fixed to {DEFAULT_MODEL!r} to match the verl GSM8K setup"
        )
    if args.group_size <= 0:
        raise ValueError("group_size must be > 0")
    if args.train_batch_size <= 0:
        raise ValueError("train_batch_size must be > 0")
    if args.total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    if args.eval_parallelism <= 0:
        raise ValueError("eval_parallelism must be > 0")
    if args.train_group_parallelism <= 0:
        raise ValueError("train_group_parallelism must be > 0")
    if args.train_group_retries < 0:
        raise ValueError("train_group_retries must be >= 0")
    if args.ppo_micro_batch_size_per_gpu <= 0:
        raise ValueError("ppo_micro_batch_size_per_gpu must be > 0")
    if args.ppo_mini_batch_size_prompts <= 0:
        raise ValueError("ppo_mini_batch_size_prompts must be > 0")
    if args.ppo_micro_batch_size_per_gpu > args.train_batch_size * args.group_size:
        raise ValueError(
            "ppo_micro_batch_size_per_gpu must be <= train_batch_size * group_size"
        )

    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    visible_gpus = require_visible_gpus(total_requested_gpus)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    checkpoint_root = artifact_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)

    train_examples, val_examples, benchmark_examples = load_gsm8k_splits(
        seed=args.seed,
        val_size=args.val_size,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        max_benchmark_examples=args.max_benchmark_examples,
    )
    save_split(artifact_dir / "data" / "train.jsonl", train_examples)
    save_split(artifact_dir / "data" / "val.jsonl", val_examples)
    save_split(artifact_dir / "data" / "benchmark.jsonl", benchmark_examples)

    gateway_config = build_gateway_config(
        artifact_dir=artifact_dir,
        gateway_host=args.gateway_host,
        gateway_port=gateway_port,
        visible_gpus=visible_gpus,
        cpus_per_node=args.cpus_per_node,
    )
    runtime_config = build_runtime_config(
        model_path=model_path,
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        checkpoint_root=checkpoint_root,
        train_batch_size=args.train_batch_size,
        group_size=args.group_size,
        ppo_mini_batch_size_prompts=args.ppo_mini_batch_size_prompts,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        ppo_epochs=args.ppo_epochs,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
    )
    train_schedule = build_train_schedule(
        train_examples,
        total_epochs=args.total_epochs,
        seed=args.seed,
    )
    groups_per_update = args.train_batch_size
    total_groups = len(train_schedule)
    expected_updates = total_groups // groups_per_update
    if expected_updates <= 0:
        raise ValueError(
            "Need at least train_batch_size prompt groups to produce one GRPO update; "
            f"got total_groups={total_groups}, train_batch_size={args.train_batch_size}"
        )
    trailing_groups = total_groups % groups_per_update
    if runtime_config.train.mini_batch_size % runtime_config.train.micro_batch_size != 0:
        raise ValueError(
            "mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu"
        )

    train_sampling = {
        "temperature": args.train_temperature,
        "top_p": args.train_top_p,
        "top_k": args.train_top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    eval_sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }

    summary_path = artifact_dir / "summary.json"
    metrics_log = artifact_dir / "metrics.jsonl"
    train_log = artifact_dir / "train_rollouts.jsonl"
    val_log = artifact_dir / "val_predictions.jsonl"
    benchmark_log = artifact_dir / "benchmark_predictions.jsonl"

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "benchmark_examples": len(benchmark_examples),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "expected_updates": expected_updates,
        "group_size": args.group_size,
        "model_path": model_path,
        "ppo_mini_batch_size_prompts": args.ppo_mini_batch_size_prompts,
        "reward_method": "strict",
        "samples_per_update": args.train_batch_size * args.group_size,
        "sampling_seed_base": args.seed,
        "train_sampling": train_sampling,
        "eval_sampling": eval_sampling,
        "slurm_step_gpus": str(os.environ.get("SLURM_STEP_GPUS")),
        "total_epochs": args.total_epochs,
        "train_batch_size": args.train_batch_size,
        "train_examples": len(train_examples),
        "train_groups": total_groups,
        "train_groups_dropped": trailing_groups,
        "val_examples": len(val_examples),
    }

    best_val_accuracy = -1.0
    best_checkpoint: dict[str, Any] | None = None

    with Gateway(gateway_config).start():
        with Session(runtime_config).start() as session:
            baseline = run_eval(
                session,
                examples=val_examples,
                sampling_params=eval_sampling,
                split_name="val_baseline",
                log_path=val_log,
                parallelism=args.eval_parallelism,
            )
            baseline_checkpoint = session.export_checkpoint()
            summary["baseline_val"] = baseline
            summary["baseline_checkpoint"] = baseline_checkpoint
            best_val_accuracy = float(baseline["accuracy"])
            best_checkpoint = baseline_checkpoint
            append_jsonl(
                metrics_log,
                {
                    "metric": "val",
                    "policy_version": baseline_checkpoint["policy_version"],
                    "result": baseline,
                    "stage": "baseline",
                },
            )
            save_summary(summary_path, summary)

            observed_updates = 0
            consumed_groups = 0
            while observed_updates < expected_updates:
                start = consumed_groups
                end = start + groups_per_update
                batch_specs = [
                    (group_index, epoch, example)
                    for group_index, (epoch, example) in enumerate(
                        train_schedule[start:end], start=start + 1
                    )
                ]
                if len(batch_specs) < groups_per_update:
                    break

                collection_started_at = time.monotonic()
                update_groups = collect_update_groups(
                    session,
                    group_specs=batch_specs,
                    group_size=args.group_size,
                    sampling_params=train_sampling,
                    sampling_seed_base=args.seed,
                    log_path=train_log,
                    parallelism=args.train_group_parallelism,
                    retries=args.train_group_retries,
                )
                collection_seconds = time.monotonic() - collection_started_at
                consumed_groups = end

                policy_sync_started_at = time.monotonic()
                current_version = wait_for_policy_version(
                    session,
                    target_version=observed_updates + 1,
                    timeout=args.wait_timeout,
                )
                policy_sync_seconds = time.monotonic() - policy_sync_started_at
                observed_updates = current_version
                train_update_metric = summarize_train_update(
                    update_index=observed_updates,
                    policy_version=current_version,
                    groups=update_groups,
                    collection_seconds=collection_seconds,
                    policy_sync_seconds=policy_sync_seconds,
                )
                append_jsonl(metrics_log, train_update_metric)
                summary["last_train_update"] = train_update_metric
                if observed_updates % args.eval_interval != 0 and observed_updates != expected_updates:
                    continue

                val_result = run_eval(
                    session,
                    examples=val_examples,
                    sampling_params=eval_sampling,
                    split_name=f"val_update_{observed_updates}",
                    log_path=val_log,
                    parallelism=args.eval_parallelism,
                )
                metric_record = {
                    "metric": "val",
                    "policy_version": current_version,
                    "result": val_result,
                    "step": observed_updates,
                }
                append_jsonl(metrics_log, metric_record)
                summary["last_val"] = metric_record

                if float(val_result["accuracy"]) > best_val_accuracy:
                    best_val_accuracy = float(val_result["accuracy"])
                    best_checkpoint = session.export_checkpoint()
                    summary["best_val"] = val_result
                    summary["best_checkpoint"] = best_checkpoint
                    save_summary(summary_path, summary)

            final_target = expected_updates
            final_version = wait_for_policy_version(
                session,
                target_version=final_target,
                timeout=args.wait_timeout,
            )
            final_val = run_eval(
                session,
                examples=val_examples,
                sampling_params=eval_sampling,
                split_name="val_final",
                log_path=val_log,
                parallelism=args.eval_parallelism,
            )
            benchmark = run_eval(
                session,
                examples=benchmark_examples,
                sampling_params=eval_sampling,
                split_name="benchmark_final",
                log_path=benchmark_log,
                parallelism=args.eval_parallelism,
            )
            final_checkpoint = session.export_checkpoint()

    summary.update(
        {
            "best_checkpoint": best_checkpoint,
            "best_val_accuracy": best_val_accuracy,
            "final_checkpoint": final_checkpoint,
            "final_policy_version": final_version,
            "final_val": final_val,
            "benchmark": benchmark,
        }
    )
    save_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
