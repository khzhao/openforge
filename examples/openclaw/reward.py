# Copyright 2026 openforge

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

from .session_state import PendingTurn

__all__ = [
    "RewardDecision",
    "build_prm_judge_prompt",
    "extract_feedback_state",
    "majority_vote",
    "parse_prm_score",
]


@dataclass(slots=True)
class RewardDecision:
    """One scalar reward plus a short explanation."""

    reward: float
    reason: str
    feedback_text: str


def extract_feedback_state(
    *,
    pending: PendingTurn,
    messages: list[dict[str, Any]],
) -> tuple[str, str]:
    """Extract follow-up state text and role for judging the previous turn."""
    appended = (
        messages[pending.request_message_count :]
        if len(messages) > pending.request_message_count
        else []
    )
    feedback_messages = [
        item for item in appended if str(item.get("role", "")) != "assistant"
    ]
    if not feedback_messages:
        feedback_messages = [
            item for item in messages if str(item.get("role", "")) != "assistant"
        ][-1:]

    feedback_text = "\n".join(
        _message_text(item) for item in feedback_messages if _message_text(item)
    ).strip()
    if not feedback_messages:
        return feedback_text, "user"
    next_state_role = str(feedback_messages[-1].get("role", "user"))
    return feedback_text, next_state_role


def build_prm_judge_prompt(
    *,
    pending: PendingTurn,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    """Build the OpenClaw-RL-style PRM judge prompt."""
    feedback_text, next_state_role = extract_feedback_state(
        pending=pending,
        messages=messages,
    )
    system_prompt = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output successfully fulfilled "
        "the user's intent at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked.\n"
        "This content was NOT available before the assistant's action.\n"
        "A successful, non-error tool output means the assistant's action worked "
        "correctly and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was "
        "wrong, incomplete, or unwanted.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous or unrelated.\n\n"
        "Key negative signals include redo requests, correction requests, rephrased "
        "requests, and tool/environment failures.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user_prompt = (
        f"## Assistant output\n{pending.assistant_text or '[empty]'}\n\n"
        f"## Next state [role: {next_state_role}]\n"
        f"{feedback_text or '[no textual follow-up]'}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        feedback_text,
    )


def parse_prm_score(text: str) -> int | None:
    """Parse the last boxed PRM score from judge output."""
    import re

    matches = re.findall(r"\\boxed\{([-+]?\d)\}", text)
    if not matches:
        return None
    value = int(matches[-1])
    if value in (-1, 0, 1):
        return value
    return None


def majority_vote(scores: list[int | None]) -> float:
    """Return the OpenClaw-RL-style majority-vote score."""
    valid = [score for score in scores if score is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top_score, top_count = counter.most_common(1)[0]
    if list(counter.values()).count(top_count) > 1:
        return 0.0
    return float(top_score)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
    return ""
