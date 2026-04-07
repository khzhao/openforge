# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import httpx

from openforge import active_state

from examples.gsm8k.common import load_examples
from examples.gsm8k.task import (
    build_gsm8k_prompt,
    compute_gsm8k_score,
    extract_gsm8k_ground_truth,
)
from examples.openclaw.replay.common import post_turn
from examples.openclaw.session_state import SessionStateStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GSM8K-style replay episodes through the OpenClaw middleware"
    )
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.max_completion_tokens is not None and args.max_completion_tokens <= 0:
        raise SystemExit("--max-completion-tokens must be > 0")

    split = args.split or ("test" if args.mode == "eval" else "train")
    examples = load_examples(
        split=split,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    gateway_base_url = _gateway_base_url()
    model = args.model or _default_model()
    max_completion_tokens = (
        args.max_completion_tokens or _default_max_completion_tokens()
    )
    store = SessionStateStore(Path(args.state_db).expanduser())

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.request_timeout) as client:
        examples_per_epoch = len(examples)
        for epoch_index in range(1, args.epochs + 1):
            for index, example in enumerate(examples, start=1):
                global_episode_index = ((epoch_index - 1) * examples_per_epoch) + index
                results.append(
                    _run_episode(
                        client=client,
                        store=store,
                        middleware_base_url=args.base_url,
                        gateway_base_url=gateway_base_url,
                        model=model,
                        question=example.question,
                        answer=example.answer,
                        mode=args.mode,
                        epoch_index=epoch_index,
                        episode_index=index,
                        global_episode_index=global_episode_index,
                        max_completion_tokens=max_completion_tokens,
                    )
                )

    accuracy = (
        sum(1 for item in results if item["reward"] == 1.0) / len(results)
        if results
        else 0.0
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "split": split,
                "epochs": args.epochs,
                "examples_per_epoch": len(examples),
                "episodes": len(results),
                "accuracy": accuracy,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


def _run_episode(
    *,
    client: httpx.Client,
    store: SessionStateStore,
    middleware_base_url: str,
    gateway_base_url: str,
    model: str,
    question: str,
    answer: str,
    mode: str,
    epoch_index: int,
    episode_index: int,
    global_episode_index: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    prompt = build_gsm8k_prompt(question)
    session_id = _build_session_id(
        mode=mode,
        epoch_index=epoch_index,
        episode_index=episode_index,
    )

    response_payload = post_turn(
        client=client,
        base_url=middleware_base_url,
        model=model,
        session_id=session_id,
        messages=[{"role": "user", "content": prompt}],
        turn_type="side" if mode == "eval" else "main",
        max_completion_tokens=max_completion_tokens,
    )
    assistant_text = _assistant_text(response_payload)
    ground_truth = extract_gsm8k_ground_truth(answer)
    reward = float(
        compute_gsm8k_score(
            assistant_text,
            ground_truth,
            method="strict",
            format_score=0.0,
            score=1.0,
        )
    )

    if mode == "train":
        pending = store.get_pending(session_id)
        if pending is None:
            raise RuntimeError(f"no pending trajectory found for session {session_id}")
        gateway_client = httpx.Client(base_url=gateway_base_url, timeout=120.0)
        try:
            response = gateway_client.post(
                "/end_trajectory",
                json={
                    "session_id": pending.openforge_session_id,
                    "trajectory_id": pending.trajectory_id,
                    "final_reward": reward,
                },
            )
            response.raise_for_status()
        finally:
            gateway_client.close()
        store.complete_turn_record(
            trajectory_id=pending.trajectory_id,
            reward=reward,
            reason="exact_match" if reward == 1.0 else "incorrect_or_bad_format",
            feedback_text=f"target={ground_truth}",
        )
        store.clear_pending(session_id)

    return {
        "session_id": session_id,
        "epoch_index": epoch_index,
        "episode_index": episode_index,
        "global_episode_index": global_episode_index,
        "question": question,
        "ground_truth": ground_truth,
        "assistant_text": assistant_text,
        "reward": reward,
        "mode": mode,
    }


def _build_session_id(
    *,
    mode: str,
    epoch_index: int,
    episode_index: int,
) -> str:
    return (
        f"gsm8k-{mode}-epoch{epoch_index:03d}-"
        f"episode{episode_index:05d}-{uuid.uuid4().hex[:8]}"
    )


def _assistant_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _gateway_base_url() -> str:
    host, port = active_state.load_active_gateway_target()
    return f"http://{host}:{port}"


def _default_model() -> str:
    runtime = active_state.load_active_runtime_config()
    return runtime.model.model_name_or_path


def _default_max_completion_tokens() -> int:
    runtime = active_state.load_active_runtime_config()
    return int(runtime.rollout.request.max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
