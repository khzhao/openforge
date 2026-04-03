# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .session_state import PendingTurn

__all__ = [
    "RewardDecision",
    "build_judge_messages",
    "extract_feedback_text",
    "parse_judge_output",
]


@dataclass(slots=True)
class RewardDecision:
    """One scalar reward plus a short explanation."""

    reward: float
    reason: str
    feedback_text: str


def extract_feedback_text(
    *,
    pending: PendingTurn,
    messages: list[dict[str, Any]],
) -> str:
    """Extract follow-up state text for judging the previous assistant turn."""
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

    return "\n".join(
        _message_text(item) for item in feedback_messages if _message_text(item)
    ).strip()


def build_judge_messages(
    *,
    pending: PendingTurn,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    """Build a judge prompt for the most recent pending assistant turn."""
    feedback_text = extract_feedback_text(
        pending=pending,
        messages=messages,
    )
    system_prompt = (
        "You are a strict reward judge for online RL.\n"
        "Decide whether the previous assistant response helped, hurt, or was unclear "
        "given the next state.\n"
        "Return compact JSON with keys: verdict, reward, reason.\n"
        'Use verdict in {"positive","negative","neutral"}.\n'
        "Use reward in [-1.0, 1.0].\n"
        "Be conservative. Use neutral when the next state is unrelated or ambiguous."
    )
    user_prompt = (
        "Evaluate the previous assistant turn.\n\n"
        f"Previous assistant response:\n{pending.assistant_text or '[empty]'}\n\n"
        f"Next-state feedback:\n{feedback_text or '[no textual follow-up]'}\n\n"
        "Return JSON only."
    )
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        feedback_text,
    )


def parse_judge_output(*, text: str, feedback_text: str) -> RewardDecision:
    """Parse judge output into a scalar reward decision."""
    import json

    normalized = text.strip()
    reward = 0.0
    reason = "judge_parse_fallback"
    if normalized:
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_reward = payload.get("reward", 0.0)
            try:
                reward = max(-1.0, min(1.0, float(raw_reward)))
            except (TypeError, ValueError):
                reward = 0.0
            raw_reason = payload.get("reason")
            if isinstance(raw_reason, str) and raw_reason.strip():
                reason = raw_reason.strip()
            else:
                verdict = payload.get("verdict")
                if isinstance(verdict, str) and verdict.strip():
                    reason = verdict.strip()
    return RewardDecision(
        reward=reward,
        reason=reason,
        feedback_text=feedback_text,
    )


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
