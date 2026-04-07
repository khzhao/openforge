# Copyright 2026 openforge

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "append_assistant_message",
    "load_replay_rows",
    "post_turn",
]


def load_replay_rows(path: str) -> list[dict[str, Any]]:
    """Load replay rows from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError("each replay row must be a JSON object")
            rows.append(payload)
    return rows


def append_assistant_message(
    messages: list[dict[str, Any]],
    response_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append the assistant response to the chat history."""
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response payload is missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("response choice is malformed")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response message is malformed")
    return [*messages, dict(message)]


def post_turn(
    *,
    client: httpx.Client,
    base_url: str,
    model: str,
    session_id: str,
    messages: list[dict[str, Any]],
    turn_type: str = "main",
    max_completion_tokens: int = 512,
) -> dict[str, Any]:
    """Send one middleware turn and return the JSON response."""
    response = client.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={
            "X-Session-Id": session_id,
            "X-Turn-Type": turn_type,
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_completion_tokens": max_completion_tokens,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("middleware returned a non-object response")
    return payload
