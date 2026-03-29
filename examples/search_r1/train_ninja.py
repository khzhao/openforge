# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import datasets
import httpx

import openforge.ninja as ninja
from examples.shared import (
    add_train_cli_args,
    load_runtime_config,
    make_artifact_dir,
    plan_train_batches,
    print_train_update,
    print_validation_update,
    save_summary,
)

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search Wikipedia for short factual evidence snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a search-enabled question answering agent. "
    "Each assistant reply must use <think>...</think> for reasoning. "
    "If you need evidence, call the search tool with a concise query. "
    "When you have enough evidence, end with <answer>...</answer>."
)

_SEARCH_HEADERS = {
    "User-Agent": "OpenForge-SearchR1/1.0 (https://github.com/openai/openforge)",
}
_SEARCH_CACHE: dict[tuple[str, int], str] = {}
_SEARCH_LOCK = threading.Lock()
_SEARCH_NEXT_REQUEST_AT = 0.0
_SEARCH_MIN_INTERVAL_SECONDS = 0.5
_SEARCH_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SEARCH_FAILURE_RESULT = "Search backend unavailable."
_DEFAULT_PROMPT_DATA = "/home/guo/kzhao/data/kzhao/search_r1_train.parquet"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the SearchR1 Ninja example."""
    parser = add_train_cli_args(
        argparse.ArgumentParser(description=__doc__),
        default_group_size=8,
    )
    parser.add_argument("--prompt-data", default=_DEFAULT_PROMPT_DATA)
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--label-key", default="reward_model")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--search-timeout", type=float, default=30.0)
    parser.add_argument("--validation-data", default=None)
    return parser.parse_args()


def _load_prompt_rows(
    *,
    path: str,
    input_key: str,
    label_key: str,
    seed: int,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    suffix = Path(path).suffix
    if suffix == ".parquet":
        rows = datasets.load_dataset("parquet", data_files=path)["train"]
    elif suffix in {".json", ".jsonl"}:
        rows = datasets.load_dataset("json", data_files=path)["train"]
    else:
        raise ValueError("prompt-data must be a .parquet, .json, or .jsonl file")

    rows = rows.shuffle(seed=seed)
    examples: list[dict[str, Any]] = []
    for row in rows:
        examples.append(
            {
                "prompt": row[input_key],
                "reward_model": row[label_key],
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _search_wikipedia(
    *,
    query: str,
    top_k: int,
    timeout: float,
) -> str:
    global _SEARCH_NEXT_REQUEST_AT
    cache_key = (query, top_k)
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    for attempt in range(5):
        with _SEARCH_LOCK:
            cached = _SEARCH_CACHE.get(cache_key)
            if cached is not None:
                return cached
            now = time.monotonic()
            wait_seconds = _SEARCH_NEXT_REQUEST_AT - now
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)
            _SEARCH_NEXT_REQUEST_AT = time.monotonic() + _SEARCH_MIN_INTERVAL_SECONDS
        try:
            response = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                headers=_SEARCH_HEADERS,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": top_k,
                    "utf8": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _SEARCH_RETRYABLE_STATUS_CODES:
                return _SEARCH_FAILURE_RESULT
            time.sleep(min(2.0**attempt, 8.0))
            continue
        except httpx.HTTPError:
            time.sleep(min(2.0**attempt, 8.0))
            continue

        payload = response.json()
        hits = payload["query"]["search"]
        result = "\n".join(
            f"{index}. {hit['title']}: {_strip_html(hit['snippet'])}"
            for index, hit in enumerate(hits, start=1)
        )
        with _SEARCH_LOCK:
            _SEARCH_CACHE[cache_key] = result
        return result

    return _SEARCH_FAILURE_RESULT


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _extract_information_blocks(text: str) -> list[str]:
    return [
        match.strip()
        for match in re.findall(
            r"<information>(.*?)</information>", text, flags=re.DOTALL
        )
    ]


def _ground_truth_targets(reward_model: Any) -> list[str]:
    if isinstance(reward_model, str):
        return [reward_model]
    if isinstance(reward_model, dict):
        ground_truth = reward_model["ground_truth"]
        if isinstance(ground_truth, dict):
            targets = ground_truth["target"]
            if isinstance(targets, str):
                return [targets]
            return [str(target) for target in targets]
        if isinstance(ground_truth, str):
            return [ground_truth]
    raise ValueError("reward_model must be a string or a dict with ground_truth.target")


def _is_valid_sequence(text: str) -> bool:
    tags = re.findall(r"</?(?:think|search|information|answer)>", text)
    if not tags:
        return False
    state = "start"
    for tag in tags:
        if tag == "<think>" and state in {"start", "information"}:
            state = "in_think"
            continue
        if tag == "</think>" and state == "in_think":
            state = "after_think"
            continue
        if tag == "<search>" and state == "after_think":
            state = "in_search"
            continue
        if tag == "</search>" and state == "in_search":
            state = "after_search"
            continue
        if tag == "<information>" and state == "after_search":
            state = "in_information"
            continue
        if tag == "</information>" and state == "in_information":
            state = "information"
            continue
        if tag == "<answer>" and state == "after_think":
            state = "in_answer"
            continue
        if tag == "</answer>" and state == "in_answer":
            state = "end"
            continue
        return False
    return state == "end"


def _retrieval_correct(text: str, targets: list[str]) -> bool:
    normalized_targets = [_normalize_answer(target) for target in targets]
    for block in _extract_information_blocks(text):
        normalized_block = _normalize_answer(block)
        for target in normalized_targets:
            if target in normalized_block:
                return True
    return False


def _compute_reward(response_text: str, reward_model: Any) -> float:
    targets = _ground_truth_targets(reward_model)
    answer = _extract_answer(response_text)
    valid_format = _is_valid_sequence(response_text)
    retrieval_correct = valid_format and _retrieval_correct(response_text, targets)
    if answer is not None and any(
        _normalize_answer(answer) == _normalize_answer(target) for target in targets
    ):
        return 1.0 if valid_format else 0.8
    if valid_format:
        return 0.3 if retrieval_correct else 0.2
    return 0.0


def _assistant_payload(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if message.content is not None:
        payload["content"] = str(message.content)
    if message.tool_calls is not None:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _messages_from_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in prompt
    ]


def main() -> int:
    """Run the SearchR1 Ninja example with explicit train and validate calls."""
    args = parse_args()
    if args.max_turns <= 0:
        raise ValueError("max-turns must be > 0")
    if args.search_top_k <= 0:
        raise ValueError("search-top-k must be > 0")
    if args.validation_every_updates < 0:
        raise ValueError("validation-every-updates must be >= 0")

    artifact_dir = make_artifact_dir(args.artifact_dir, prefix="search-r1-ninja-")
    runtime_config = load_runtime_config(args.runtime_config)
    sampling_params = {
        "temperature": runtime_config.rollout.request.temperature,
        "top_p": runtime_config.rollout.request.top_p,
        "max_completion_tokens": runtime_config.rollout.request.max_new_tokens,
    }
    inputs = _load_prompt_rows(
        path=args.prompt_data,
        input_key=args.input_key,
        label_key=args.label_key,
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    _write_jsonl(artifact_dir / "data" / "train.jsonl", inputs)
    validation_path: Path | None = None
    validation_examples = 0
    if args.validation_every_updates > 0:
        validation_inputs = _load_prompt_rows(
            path=args.prompt_data
            if args.validation_data is None
            else args.validation_data,
            input_key=args.input_key,
            label_key=args.label_key,
            seed=args.seed,
            max_examples=args.max_validation_examples,
        )
        validation_path = artifact_dir / "data" / "validation.jsonl"
        _write_jsonl(validation_path, validation_inputs)
        validation_examples = len(validation_inputs)

    @ninja.agent()
    def search_agent(client, *, prompt: Any, reward_model: Any) -> float:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_messages_from_prompt(prompt),
        ]
        trace_parts: list[str] = []
        for _ in range(args.max_turns):
            response = client.chat.completions.create(
                model=runtime_config.model.model_name_or_path,
                messages=messages,
                tools=[_SEARCH_TOOL],
                tool_choice="auto",
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                max_completion_tokens=sampling_params["max_completion_tokens"],
            )
            message = response.choices[0].message
            messages.append(_assistant_payload(message))
            if message.content is not None:
                trace_parts.append(str(message.content))
            if message.tool_calls is None:
                return _compute_reward("".join(trace_parts), reward_model)

            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                query = str(arguments["query"])
                trace_parts.append(f"<search>{query}</search>")
                tool_result = _search_wikipedia(
                    query=query,
                    top_k=args.search_top_k,
                    timeout=args.search_timeout,
                )
                trace_parts.append(f"<information>{tool_result}</information>")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return _compute_reward("".join(trace_parts), reward_model)

    train_plan = plan_train_batches(
        runtime_config=runtime_config,
        inputs=inputs,
        group_size=args.group_size,
        epochs=args.total_epochs,
        seed=args.seed,
        max_updates=args.max_updates,
    )
    train_updates: list[dict[str, object]] = []
    validation_updates: list[dict[str, object]] = []
    for update_index, batch_inputs in enumerate(
        train_plan["update_inputs"],
        start=1,
    ):
        assert isinstance(batch_inputs, list)
        train_update = ninja.train_async(
            search_agent,
            inputs=batch_inputs,
            group_size=args.group_size,
            concurrency=args.train_group_parallelism,
            retries=args.train_group_retries,
        )
        train_event = {
            **train_update,
            "update_index": update_index,
        }
        train_updates.append(train_event)
        print_train_update(train_event)
        if (
            validation_path is not None
            and args.validation_every_updates > 0
            and update_index % args.validation_every_updates == 0
        ):
            validation_update = ninja.validate(
                search_agent,
                file_path=str(validation_path),
                wait_timeout=args.wait_timeout,
            )
            validation_event = {
                **validation_update,
                "update_index": update_index,
            }
            validation_updates.append(validation_event)
            print_validation_update(validation_event)

    final_checkpoint = search_agent.save()
    summary = {
        "artifact_dir": str(artifact_dir),
        "completed_updates": len(train_updates),
        "expected_updates": train_plan["expected_updates"],
        "final_checkpoint": final_checkpoint,
        "final_policy_version": final_checkpoint["policy_version"],
        "global_batch_size": train_plan["global_batch_size"],
        "group_size": args.group_size,
        "input_key": args.input_key,
        "label_key": args.label_key,
        "last_train_update": train_updates[-1] if train_updates else None,
        "last_validation_update": (
            validation_updates[-1] if validation_updates else None
        ),
        "max_turns": args.max_turns,
        "max_updates": args.max_updates,
        "model_path": runtime_config.model.model_name_or_path,
        "prompt_groups_per_update": train_plan["prompt_groups_per_update"],
        "prompt_data": args.prompt_data,
        "search_top_k": args.search_top_k,
        "train_groups": train_plan["train_groups"],
        "train_groups_consumed": len(train_updates)
        * int(train_plan["prompt_groups_per_update"]),
        "train_groups_dropped": train_plan["train_groups_dropped"],
        "train_examples": len(inputs),
        "train_updates": train_updates,
        "validation_data": (
            args.prompt_data if args.validation_data is None else args.validation_data
        ),
        "validation_every_updates": args.validation_every_updates,
        "validation_examples": validation_examples,
        "validation_updates": validation_updates,
    }
    save_summary(artifact_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
