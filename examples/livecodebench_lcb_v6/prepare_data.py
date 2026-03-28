# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import datasets

from examples.livecodebench_lcb_v6.common import _normalize_problem_row
from examples.livecodebench_lcb_v6.task import (
    sample_livecodebench_train_reward_spec,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        required=True,
        help="Path to a local parquet/json/jsonl file with raw LiveCodeBench rows.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.jsonl, test.jsonl, and summary.json will be written.",
    )
    parser.add_argument(
        "--train-test-fraction",
        type=float,
        default=0.5,
        help="Fraction of tests to keep in train.jsonl. test.jsonl keeps all tests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--judge-timeout", type=float, default=6.0)
    parser.add_argument("--judge-memory-mb", type=int, default=1024)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.input_data)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    train_test_counts: list[int] = []
    full_test_counts: list[int] = []

    for index, row in enumerate(rows):
        if args.max_examples is not None and len(test_rows) >= args.max_examples:
            break
        example = _normalize_problem_row(
            dict(row),
            index=index,
            judge_timeout=args.judge_timeout,
            judge_memory_mb=args.judge_memory_mb,
        )
        full_reward_spec = dict(example.reward_spec)
        full_reward_spec["sparse_rewards"] = True
        train_reward_spec = sample_livecodebench_train_reward_spec(
            full_reward_spec,
            keep_fraction=args.train_test_fraction,
            seed=args.seed + index,
        )
        train_rows.append(
            {
                "prompt": example.prompt,
                "problem_id": example.problem_id,
                "reward_spec": train_reward_spec,
            }
        )
        test_rows.append(
            {
                "prompt": example.prompt,
                "problem_id": example.problem_id,
                "reward_spec": full_reward_spec,
            }
        )
        train_test_counts.append(len(train_reward_spec["tests"]))
        full_test_counts.append(len(full_reward_spec["tests"]))

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)

    summary = {
        "input_data": args.input_data,
        "output_dir": str(output_dir),
        "examples": len(test_rows),
        "train_test_fraction": args.train_test_fraction,
        "mean_tests_train": (
            0.0
            if not train_test_counts
            else sum(train_test_counts) / len(train_test_counts)
        ),
        "mean_tests_test": (
            0.0 if not full_test_counts else sum(full_test_counts) / len(full_test_counts)
        ),
        "judge_timeout": args.judge_timeout,
        "judge_memory_mb": args.judge_memory_mb,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_rows(path: str) -> datasets.Dataset:
    input_path = Path(path)
    suffix = input_path.suffix
    if suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files=str(input_path))["train"]
    if suffix in {".json", ".jsonl"}:
        return datasets.load_dataset("json", data_files=str(input_path))["train"]
    raise ValueError("input-data must be a .parquet, .json, or .jsonl file")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
