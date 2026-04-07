# Copyright 2026 openforge

from __future__ import annotations

import collections
import re
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

_UNTRUSTED_METADATA_BLOCK_RE = re.compile(
    r"(?ms)^(?:[^\n]*\(untrusted metadata\):\s*```json\n.*?\n```\s*)+"
)
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")


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
    del pending
    feedback_message = next(
        (
            item
            for item in reversed(messages)
            if str(item.get("role", "")) != "assistant"
        ),
        None,
    )
    if feedback_message is None:
        return "", "user"
    feedback_text = _strip_untrusted_metadata(_message_text(feedback_message)).strip()
    next_state_role = str(feedback_message.get("role", "user"))
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
    assistant_text = _strip_think_blocks(pending.assistant_text or "").strip()
    system_prompt = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully "
        "fulfilled** the user's intent at that step, using the next state as "
        "evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action — "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action "
        "worked correctly only if the assistant's final answer or action "
        "actually matches what the tool result confirms.\n\n"
        "## Scoring rules\n"
        "- Default to \\boxed{0} unless the next state is strong evidence.\n"
        "- \\boxed{1} (good): Use only for explicit confirmation that the "
        "assistant's final answer/action was correct. Good signals include:\n"
        "  * The user explicitly confirms satisfaction or success.\n"
        "  * A tool/environment result directly verifies the assistant's "
        "stated answer or intended action.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output "
        "was wrong, incomplete, or unwanted. **Key negative signals include:**\n"
        "  * The user asks the assistant to **redo, retry, or repeat** the "
        "same action (\"do it again\", \"try again\", \"one more time\").\n"
        "  * The user requests a **correction or modification** to what the "
        "assistant just did (\"change X to Y\", \"no, I meant …\", "
        "\"not that, …\", \"please fix …\").\n"
        "  * The user **rephrases or restates** the same request, implying the "
        "assistant did not understand or execute it correctly.\n"
        "  * The environment returns an **error, failure, or unexpected "
        "result** caused by the assistant's action.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous — e.g. the user "
        "gives any new request or topic change, the user keeps the "
        "conversation going without explicit approval, there is insufficient "
        "information to judge, or the user asks a question-only follow-up such "
        "as a prediction or clarification request. A fresh request like "
        "\"/start\" or \"what's in this directory?\" after the previous turn "
        "is usually neutral, not positive.\n\n"
        "## Important\n"
        "A change request IS negative feedback — it means the previous output "
        "did not meet the user's need. Do NOT treat it as a neutral new "
        "instruction. Do NOT reward a turn merely because the user kept "
        "talking or because a tool returned some output. Judge only the "
        "assistant's final visible answer/action, not hidden reasoning.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user_prompt = (
        f"## Assistant output\n{assistant_text or '[empty]'}\n\n"
        f"## Next state [role: {next_state_role}]\n"
        f"{feedback_text or '[no textual follow-up]'}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, "
        "(c) a question-only clarification / prediction request, or "
        "(d) ambiguous? Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
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
    matches = re.findall(r"\\boxed\{([-+]?\d)\}", text)
    if matches:
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


def _strip_untrusted_metadata(text: str) -> str:
    return _UNTRUSTED_METADATA_BLOCK_RE.sub("", text, count=1)


def _strip_think_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    return cleaned.strip()
