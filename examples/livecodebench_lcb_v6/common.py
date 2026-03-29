# Copyright 2026 openforge

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets

from examples.livecodebench_lcb_v6.task import (
    build_livecodebench_prompt,
    decode_livecodebench_private_test_cases,
)
from examples.shared import add_train_cli_args, load_runtime_config, make_artifact_dir
from openforge.utils.processing import load_tokenizer

_MAX_PROMPT_TOKENS = 2048
LIVECODEBENCH_V6_START_DATE = "2025-02-01"
LIVECODEBENCH_V6_END_DATE = "2025-05-01"


@dataclass(slots=True)
class LiveCodeBenchExample:
    """One normalized LiveCodeBench-style training example."""

    prompt: str
    reward_spec: dict[str, Any]
    problem_id: str


def build_train_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the LiveCodeBench training example."""
    parser = add_train_cli_args(
        argparse.ArgumentParser(),
        default_group_size=8,
        default_total_epochs=30,
    )
    parser.add_argument(
        "--prompt-data",
        required=True,
        help=(
            "Path to a local parquet/json/jsonl file or a directory containing "
            "train.parquet/train.jsonl/train.json."
        ),
    )
    parser.add_argument("--train-temperature", type=float, default=None)
    parser.add_argument("--train-top-p", type=float, default=None)
    parser.add_argument("--train-top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--train-max-new-tokens", type=int, default=None)
    parser.add_argument("--judge-timeout", type=float, default=6.0)
    parser.add_argument("--judge-memory-mb", type=int, default=1024)
    parser.add_argument("--validation-data", default=None)
    return parser


def parse_train_args() -> argparse.Namespace:
    """Parse CLI flags for the LiveCodeBench example."""
    return build_train_arg_parser().parse_args()


def prepare_train_setup(args: argparse.Namespace) -> dict[str, Any]:
    """Load config, prompts, and reward specs for LiveCodeBench training."""
    artifact_dir = make_artifact_dir(
        args.artifact_dir,
        prefix="livecodebench-ninja-",
    )
    runtime_config = load_runtime_config(args.runtime_config)
    train_examples = load_examples(
        path=args.prompt_data,
        split_name="train",
        seed=args.seed,
        max_examples=args.max_train_examples,
        judge_timeout=args.judge_timeout,
        judge_memory_mb=args.judge_memory_mb,
    )
    train_examples, dropped_train_examples = _filter_overlong_prompt_examples(
        train_examples,
        tokenizer_name_or_path=runtime_config.model.tokenizer_name_or_path,
        chat_template_kwargs=runtime_config.model.chat_template_kwargs,
        max_prompt_tokens=_MAX_PROMPT_TOKENS,
    )
    sampling_params = {
        "temperature": (
            runtime_config.rollout.request.temperature
            if args.train_temperature is None
            else args.train_temperature
        ),
        "top_p": (
            runtime_config.rollout.request.top_p
            if args.train_top_p is None
            else args.train_top_p
        ),
        "top_k": (
            runtime_config.rollout.request.top_k
            if args.train_top_k is None
            else args.train_top_k
        ),
        "repetition_penalty": (
            runtime_config.rollout.request.repetition_penalty
            if args.repetition_penalty is None
            else args.repetition_penalty
        ),
        "max_new_tokens": (
            runtime_config.rollout.request.max_new_tokens
            if args.train_max_new_tokens is None
            else args.train_max_new_tokens
        ),
    }
    inputs = [_example_request(example) for example in train_examples]
    validation_path: Path | None = None
    validation_examples = 0
    dropped_validation_examples = 0
    validation_data = (
        args.prompt_data if args.validation_data is None else args.validation_data
    )
    if args.validation_every_updates > 0:
        validation_rows = load_examples(
            path=validation_data,
            split_name="validation",
            seed=args.seed,
            max_examples=args.max_validation_examples,
            judge_timeout=args.judge_timeout,
            judge_memory_mb=args.judge_memory_mb,
        )
        validation_rows, dropped_validation_examples = _filter_overlong_prompt_examples(
            validation_rows,
            tokenizer_name_or_path=runtime_config.model.tokenizer_name_or_path,
            chat_template_kwargs=runtime_config.model.chat_template_kwargs,
            max_prompt_tokens=_MAX_PROMPT_TOKENS,
        )
        validation_inputs = [_example_request(example) for example in validation_rows]
        validation_path = artifact_dir / "data" / "validation.jsonl"
        _write_jsonl(validation_path, validation_inputs)
        validation_examples = len(validation_inputs)
    summary = {
        "artifact_dir": str(artifact_dir),
        "group_size": args.group_size,
        "max_updates": args.max_updates,
        "model_path": runtime_config.model.model_name_or_path,
        "global_batch_size": runtime_config.train.global_batch_size,
        "total_epochs": args.total_epochs,
        "train_examples": len(train_examples),
        "dropped_train_examples_over_prompt_limit": dropped_train_examples,
        "train_group_parallelism": (
            "auto"
            if args.train_group_parallelism is None
            else args.train_group_parallelism
        ),
        "train_group_retries": args.train_group_retries,
        "train_sampling": sampling_params,
        "prompt_data": args.prompt_data,
        "validation_data": validation_data,
        "validation_every_updates": args.validation_every_updates,
        "validation_examples": validation_examples,
        "dropped_validation_examples_over_prompt_limit": (
            dropped_validation_examples
        ),
        "max_prompt_tokens": _MAX_PROMPT_TOKENS,
        "judge_timeout": args.judge_timeout,
        "judge_memory_mb": args.judge_memory_mb,
    }
    return {
        "runtime_config": runtime_config,
        "inputs": inputs,
        "sampling_params": sampling_params,
        "summary": summary,
        "summary_path": artifact_dir / "summary.json",
        "validation_path": validation_path,
    }


def load_examples(
    *,
    path: str,
    split_name: str,
    seed: int,
    max_examples: int | None,
    judge_timeout: float,
    judge_memory_mb: int,
) -> list[LiveCodeBenchExample]:
    """Load and normalize rows from a local LiveCodeBench-style dataset."""
    input_path = _resolve_data_path(path, split_name=split_name)
    suffix = input_path.suffix
    row_iterable: list[dict[str, Any]] | datasets.Dataset
    if suffix == ".parquet":
        row_iterable = datasets.load_dataset("parquet", data_files=str(input_path))[
            "train"
        ]
    elif suffix == ".jsonl":
        row_iterable = _load_jsonl_rows(input_path)
    elif suffix == ".json":
        row_iterable = _load_json_rows(input_path)
    else:
        raise ValueError("prompt-data must be a .parquet, .json, or .jsonl file")

    if isinstance(row_iterable, list):
        rng = random.Random(seed)
        rng.shuffle(row_iterable)
        rows = row_iterable
    else:
        rows = row_iterable.shuffle(seed=seed)

    examples: list[LiveCodeBenchExample] = []
    for index, row in enumerate(rows):
        normalized_row = dict(row)
        if not keep_livecodebench_v6_row(normalized_row):
            continue
        examples.append(
            _normalize_problem_row(
                normalized_row,
                index=index,
                judge_timeout=judge_timeout,
                judge_memory_mb=judge_memory_mb,
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def _example_request(example: LiveCodeBenchExample) -> dict[str, Any]:
    return {
        "prompt": example.prompt,
        "reward_spec": example.reward_spec,
        "problem_id": example.problem_id,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _filter_overlong_prompt_examples(
    examples: list[LiveCodeBenchExample],
    *,
    tokenizer_name_or_path: str,
    chat_template_kwargs: dict[str, Any],
    max_prompt_tokens: int,
) -> tuple[list[LiveCodeBenchExample], int]:
    tokenizer = load_tokenizer(tokenizer_name_or_path, trust_remote_code=True)
    kept: list[LiveCodeBenchExample] = []
    dropped = 0
    for example in examples:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if len(token_ids) <= max_prompt_tokens:
            kept.append(example)
        else:
            dropped += 1
    return kept, dropped


def _normalize_problem_row(
    row: dict[str, Any],
    *,
    index: int,
    judge_timeout: float,
    judge_memory_mb: int,
) -> LiveCodeBenchExample:
    metadata = _coerce_json_like(row.get("metadata"))
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    prompt = row.get("prompt")
    if prompt is None:
        question = _first_text(
            row,
            "question_content",
            "question",
            "problem",
        )
        if question is None:
            raise ValueError("row is missing prompt/question_content")
        starter_code = _optional_text(row.get("starter_code"))
        prompt = build_livecodebench_prompt(
            question,
            starter_code=starter_code,
            time_limit_seconds=_extract_time_limit_seconds(row, metadata_dict),
        )

    reward_spec = _build_reward_spec(
        row,
        metadata_dict=metadata_dict,
        judge_timeout=judge_timeout,
        judge_memory_mb=judge_memory_mb,
    )
    problem_id = _optional_text(
        row.get("problem_id")
        or row.get("question_id")
        or row.get("task_id")
        or row.get("slug")
    )
    if problem_id is None:
        problem_id = f"problem-{index}"

    return LiveCodeBenchExample(
        prompt=str(prompt),
        reward_spec=reward_spec,
        problem_id=problem_id,
    )


def _build_reward_spec(
    row: dict[str, Any],
    *,
    metadata_dict: dict[str, Any] | None = None,
    judge_timeout: float,
    judge_memory_mb: int,
) -> dict[str, Any]:
    explicit = row.get("reward_spec")
    if explicit is not None:
        reward_spec = _coerce_json_like(explicit)
        if not isinstance(reward_spec, dict):
            raise ValueError("reward_spec must decode to an object")
        reward_spec = dict(reward_spec)
        reward_spec.setdefault("timeout_seconds", judge_timeout)
        reward_spec.setdefault("memory_limit_mb", judge_memory_mb)
        return reward_spec

    if metadata_dict is None:
        metadata = _coerce_json_like(row.get("metadata"))
        metadata_dict = metadata if isinstance(metadata, dict) else {}
    decoded_private_tests = _decode_private_tests(row, metadata_dict)
    entry_point = _optional_text(
        row.get("entry_point")
        or row.get("fn_name")
        or metadata_dict.get("entry_point")
        or metadata_dict.get("func_name")
        or metadata_dict.get("fn_name")
        or (
            decoded_private_tests.get("fn_name")
            if isinstance(decoded_private_tests, dict)
            else None
        )
    )

    test_cases = _collect_private_test_cases(
        row,
        decoded_private_tests=decoded_private_tests,
    )
    if entry_point is not None:
        return {
            "task_type": "function",
            "entry_point": entry_point,
            "tests": _normalize_function_tests(test_cases),
            "timeout_seconds": judge_timeout,
            "memory_limit_mb": judge_memory_mb,
        }
    return {
        "task_type": "stdin_stdout",
        "tests": _normalize_stdio_tests(test_cases),
        "timeout_seconds": judge_timeout,
        "memory_limit_mb": judge_memory_mb,
    }


def _collect_private_test_cases(
    row: dict[str, Any],
    *,
    decoded_private_tests: dict[str, Any] | None = None,
) -> list[Any]:
    raw_private_tests = row.get("private_test_cases")
    if decoded_private_tests is not None:
        loaded = decoded_private_tests
    elif raw_private_tests is not None:
        loaded = _coerce_json_like(raw_private_tests)
    else:
        raise ValueError("row is missing private_test_cases")
    if isinstance(loaded, dict) and "tests" in loaded:
        loaded = loaded["tests"]
    if isinstance(loaded, dict) and "inputs" in loaded and "outputs" in loaded:
        loaded = [
            {"input": item_input, "output": item_output}
            for item_input, item_output in zip(
                loaded["inputs"],
                loaded["outputs"],
                strict=True,
            )
        ]
    if not isinstance(loaded, list):
        raise ValueError("private_test_cases must decode to a list or test bundle")
    if not loaded:
        raise ValueError("row is missing private test cases")
    return loaded


def keep_livecodebench_v6_row(row: dict[str, Any]) -> bool:
    contest_date = _extract_contest_date(row)
    if contest_date is None:
        return True
    return LIVECODEBENCH_V6_START_DATE <= contest_date < LIVECODEBENCH_V6_END_DATE


def _normalize_function_tests(test_cases: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for case in test_cases:
        loaded = _coerce_json_like(case)
        if not isinstance(loaded, dict):
            raise ValueError("function test case must be an object")
        raw_args = loaded.get("args", loaded.get("input", loaded.get("inputs", [])))
        raw_args = _coerce_literal_like(raw_args)
        if isinstance(raw_args, tuple):
            args = list(raw_args)
        elif isinstance(raw_args, list):
            args = raw_args
        else:
            args = [raw_args]
        raw_kwargs = loaded.get("kwargs", {})
        if raw_kwargs is None:
            raw_kwargs = {}
        kwargs = _coerce_literal_like(raw_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("function test kwargs must be a mapping")
        expected = loaded.get("expected", loaded.get("output", loaded.get("outputs")))
        if expected is None:
            raise ValueError("function test case is missing expected output")
        normalized.append(
            {
                "args": args,
                "kwargs": kwargs,
                "expected": _coerce_literal_like(expected),
            }
        )
    return normalized


def _normalize_stdio_tests(test_cases: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for case in test_cases:
        loaded = _coerce_json_like(case)
        if not isinstance(loaded, dict):
            raise ValueError("stdin/stdout test case must be an object")
        stdin = loaded.get("stdin", loaded.get("input"))
        stdout = loaded.get("stdout", loaded.get("output", loaded.get("expected")))
        if stdin is None or stdout is None:
            raise ValueError("stdin/stdout test case must include input and output")
        normalized.append(
            {
                "stdin": _stringify_stream(stdin),
                "stdout": _stringify_stream(stdout),
            }
        )
    return normalized


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _optional_text(row.get(key))
        if value is not None:
            return value
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{\"'":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def _coerce_literal_like(value: Any) -> Any:
    value = _coerce_json_like(value)
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if re.fullmatch(r"-?\d+", stripped) and len(stripped) <= 64:
        return int(stripped)
    if (
        re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", stripped)
        and len(stripped) <= 64
    ):
        return float(stripped)
    if stripped[0] not in "[({'\"":
        return value
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


def _stringify_stream(value: Any) -> str:
    value = _coerce_literal_like(value)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _decode_private_tests(
    row: dict[str, Any],
    metadata_dict: dict[str, Any],
) -> dict[str, Any] | None:
    raw_private_tests = row.get("private_test_cases")
    if not isinstance(raw_private_tests, str):
        return None
    stripped = raw_private_tests.strip()
    if not stripped:
        return None
    loaded = _coerce_json_like(stripped)
    if loaded is not stripped:
        return loaded if isinstance(loaded, dict) else None
    try:
        return decode_livecodebench_private_test_cases(
            stripped,
            fn_name=str(metadata_dict.get("func_name", "") or ""),
        )
    except Exception:
        return None


def _extract_contest_date(row: dict[str, Any]) -> str | None:
    metadata = _coerce_json_like(row.get("metadata"))
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for candidate in (
        row.get("contest_date"),
        row.get("contestDate"),
        row.get("release_date"),
        metadata_dict.get("contest_date"),
        metadata_dict.get("contestDate"),
        metadata_dict.get("release_date"),
    ):
        date_text = _extract_date_token(candidate)
        if date_text is not None:
            return date_text
    return None


def _extract_date_token(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    if match is None:
        return None
    return match.group(0)


def _extract_time_limit_seconds(
    row: dict[str, Any],
    metadata_dict: dict[str, Any],
) -> float | None:
    for candidate in (
        row.get("time_limit"),
        row.get("time_limit_seconds"),
        metadata_dict.get("time_limit"),
        metadata_dict.get("time_limit_seconds"),
    ):
        seconds = _coerce_seconds(candidate)
        if seconds is not None:
            return seconds
    return None


def _coerce_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    return float(match.group(0))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain one JSON object per line")
            rows.append(loaded)
    return rows


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("train"), list):
        rows = payload["train"]
    else:
        raise ValueError(f"{path} must decode to a list of objects or train-wrapped JSON")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} must contain JSON objects")
        normalized_rows.append(row)
    return normalized_rows


def _resolve_data_path(path: str, *, split_name: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        for suffix in (".parquet", ".jsonl", ".json"):
            split_path = candidate / f"{split_name}{suffix}"
            if split_path.exists():
                return split_path
        raise ValueError(
            f"{path} is missing {split_name}.parquet/.jsonl/.json for LiveCodeBench"
        )
    return candidate
