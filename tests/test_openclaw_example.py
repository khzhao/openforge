# Copyright 2026 openforge

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.openclaw.app import (
    _assistant_text,
    _normalize_messages,
    _normalize_tool_call,
    _resolve_external_session_id,
    _stream_response_payload,
)
from examples.openclaw.reward import (
    build_prm_judge_prompt,
    extract_feedback_state,
    majority_vote,
    parse_prm_score,
)
from examples.openclaw.session_state import PendingTurn, SessionStateStore


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
    feedback_text, next_state_role = extract_feedback_state(
        pending=pending,
        messages=[
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "draft answer"},
            {"role": "user", "content": "No, that's wrong. Please fix it."},
        ],
    )
    assert feedback_text == "No, that's wrong. Please fix it."
    assert next_state_role == "user"


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
    judge_messages, feedback_text = build_prm_judge_prompt(
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
    score = parse_prm_score(r"analysis... \boxed{-1}")
    assert score == -1


def test_openclaw_majority_vote_prefers_unique_winner() -> None:
    assert majority_vote([1, 1, 0]) == 1.0
    assert majority_vote([1, -1]) == 0.0


def test_openclaw_stream_response_payload_emits_done() -> None:
    payload = {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                },
            }
        ],
    }
    chunks = list(_stream_response_payload(payload))
    assert chunks[0].startswith("data: {")
    assert '"chat.completion.chunk"' in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_openclaw_normalize_messages_flattens_text_blocks() -> None:
    messages = _normalize_messages(
        [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "system text"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello world"}],
            },
        ]
    )
    assert messages == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "hello world"},
    ]


def test_openclaw_normalize_tool_call_serializes_arguments_dict() -> None:
    tool_call = _normalize_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": {"path": "/tmp/demo.txt"},
            },
        }
    )
    assert tool_call["function"]["arguments"] == '{"path":"/tmp/demo.txt"}'


def test_openclaw_session_state_records_rewarded_turn() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SessionStateStore(Path(tmp_dir) / "openclaw.sqlite3")
        store.save_turn_record(
            trajectory_id="traj-1",
            external_session_id="ext-1",
            openforge_session_id="sess-1",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            turn_type="main",
            purpose="train",
            request_messages=[{"role": "user", "content": "hello"}],
            normalized_messages=[{"role": "user", "content": "hello"}],
            assistant_text="hi there",
        )
        store.complete_turn_record(
            trajectory_id="traj-1",
            reward=1.0,
            reason="positive",
            feedback_text="thanks",
        )
        records = store.list_turn_records(limit=5)
    assert len(records) == 1
    assert records[0].status == "rewarded"
    assert records[0].reward == 1.0
    assert records[0].feedback_text == "thanks"
