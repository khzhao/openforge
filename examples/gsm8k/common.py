# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets

from examples.gsm8k.task import build_gsm8k_prompt, extract_gsm8k_ground_truth
from examples.shared import add_train_cli_args, load_runtime_config, make_artifact_dir


@dataclass(slots=True)
class GSM8kExample:
    """One GSM8K training example prepared for prompting."""

    question: str
    answer: str
    prompt: str
    ground_truth: str


def load_examples(
    *,
    split: str,
    seed: int,
    max_examples: int | None,
) -> list[GSM8kExample]:
    """Load and prepare shuffled GSM8K examples from one dataset split."""
    rows = datasets.load_dataset("openai/gsm8k", "main")[split].shuffle(seed=seed)
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


def agent_input(example: GSM8kExample) -> dict[str, str]:
    """Project one example down to the agent kwargs accepted by train/validate."""
    return {
        "prompt": example.prompt,
        "ground_truth": example.ground_truth,
    }


def save_split(path: Path, examples: list[GSM8kExample]) -> None:
    """Persist a prepared split as JSONL."""
    write_jsonl(path, [asdict(example) for example in examples])


def save_agent_input_split(path: Path, examples: list[GSM8kExample]) -> None:
    """Persist only the fields that match the GSM8K agent signature."""
    write_jsonl(path, [agent_input(example) for example in examples])


def response_text(response: Any) -> str:
    """Extract the assistant text from a chat-completions response."""
    return str(response.choices[0].message.content)


def build_train_arg_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser for GSM8K training examples."""
    parser = add_train_cli_args(
        argparse.ArgumentParser(),
        default_group_size=5,
    )
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--train-top-p", type=float, default=1.0)
    parser.add_argument("--train-top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser


def parse_train_args() -> argparse.Namespace:
    """Parse CLI flags for the shared GSM8K training example."""
    return build_train_arg_parser().parse_args()


def prepare_train_setup(args: argparse.Namespace) -> dict[str, Any]:
    """Load configs, dataset inputs, and metadata for GSM8K training."""
    artifact_dir = make_artifact_dir(args.artifact_dir, prefix="gsm8k-ninja-")
    runtime_config = load_runtime_config(args.runtime_config)

    train_examples = load_examples(
        split="train",
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    save_split(artifact_dir / "data" / "train.jsonl", train_examples)
    validation_path: Path | None = None
    validation_examples_count = 0
    if args.validation_every_updates > 0:
        validation_examples = load_examples(
            split="test",
            seed=args.seed,
            max_examples=args.max_validation_examples,
        )
        validation_path = artifact_dir / "data" / "validation.jsonl"
        save_agent_input_split(validation_path, validation_examples)
        validation_examples_count = len(validation_examples)

    sampling_params = {
        "temperature": args.train_temperature,
        "top_p": args.train_top_p,
        "top_k": args.train_top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": runtime_config.rollout.request.max_new_tokens,
    }
    inputs = [agent_input(example) for example in train_examples]
    summary = {
        "artifact_dir": str(artifact_dir),
        "group_size": args.group_size,
        "max_updates": args.max_updates,
        "model_path": runtime_config.model.model_name_or_path,
        "reward_method": "strict",
        "global_batch_size": runtime_config.train.global_batch_size,
        "total_epochs": args.total_epochs,
        "train_examples": len(train_examples),
        "validation_every_updates": args.validation_every_updates,
        "validation_examples": validation_examples_count,
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
        "validation_path": validation_path,
    }
