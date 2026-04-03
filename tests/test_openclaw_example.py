# Copyright 2026 openforge

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.openclaw.app import _assistant_text, _resolve_external_session_id
from examples.openclaw.reward import (
    build_judge_messages,
    extract_feedback_text,
    parse_judge_output,
)
from examples.openclaw.session_state import PendingTurn


def test_openclaw_resolve_external_session_id_prefers_header() -> None:
    request = SimpleNamespace(headers={"X-Session-Id": "sess-header"})
    payload = {"user": "sess-user"}
    assert _resolve_external_session_id(request, payload) == "sess-header"


def test_openclaw_resolve_external_session_id_falls_back_to_user() -> None:
    request = SimpleNamespace(headers={})
    payload = {"user": "sess-user"}
    assert _resolve_external_session_id(request, payload) == "sess-user"


def test_openclaw_assistant_text_extracts_first_choice_content() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "hello from openforge",
                }
            }
        ]
    }
    assert _assistant_text(payload) == "hello from openforge"


def test_openclaw_extract_feedback_text_uses_follow_up_message() -> None:
    pending = PendingTurn(
        external_session_id="ext-1",
        openforge_session_id="sess-1",
        trajectory_id="traj-1",
        model_name="Qwen/Qwen3-8B",
        request_message_count=2,
        assistant_text="draft answer",
        created_at=0.0,
    )
    feedback_text = extract_feedback_text(
        pending=pending,
        messages=[
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "draft answer"},
            {"role": "user", "content": "No, that's wrong. Please fix it."},
        ],
    )
    assert feedback_text == "No, that's wrong. Please fix it."


def test_openclaw_build_judge_messages_includes_pending_answer() -> None:
    pending = PendingTurn(
        external_session_id="ext-1",
        openforge_session_id="sess-1",
        trajectory_id="traj-1",
        model_name="Qwen/Qwen3-8B",
        request_message_count=2,
        assistant_text="draft answer",
        created_at=0.0,
    )
    judge_messages, feedback_text = build_judge_messages(
        pending=pending,
        messages=[
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "draft answer"},
            {"role": "user", "content": "No, that's wrong. Please fix it."},
        ],
    )
    assert feedback_text == "No, that's wrong. Please fix it."
    assert judge_messages[1]["content"].find("draft answer") >= 0


def test_openclaw_parse_judge_output_reads_json_reward() -> None:
    decision = parse_judge_output(
        text='{"verdict":"negative","reward":-0.75,"reason":"user_corrected_answer"}',
        feedback_text="No, that's wrong. Please fix it.",
    )
    assert decision.reward == -0.75
    assert decision.reason == "user_corrected_answer"
