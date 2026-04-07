# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import examples.openclaw.app as openclaw_app
import pytest
from examples.openclaw.app import (
    _assistant_text,
    _judge_chat_completions_url,
    _judge_pending_sync,
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
from fastapi import HTTPException


def test_openclaw_resolve_external_session_id_prefers_header() -> None:
    request = SimpleNamespace(headers={"X-Session-Id": "sess-header"})
    assert _resolve_external_session_id(request) == "sess-header"


def test_openclaw_resolve_external_session_id_requires_header() -> None:
    request = SimpleNamespace(headers={})
    with pytest.raises(HTTPException) as exc_info:
        _resolve_external_session_id(request)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "missing external session id; set X-Session-Id in OpenClaw"


def test_openclaw_judge_env_reads_current_names(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_JUDGE_M", "3")
    monkeypatch.setenv("OPENCLAW_JUDGE_TEMPERATURE", "0.25")
    monkeypatch.setenv("OPENCLAW_JUDGE_MAX_TOKENS", "4096")
    reloaded_module = importlib.reload(openclaw_app)
    try:
        assert reloaded_module._JUDGE_M == 3
        assert reloaded_module._JUDGE_TEMPERATURE == 0.25
        assert reloaded_module._JUDGE_MAX_TOKENS == 4096
    finally:
        importlib.reload(openclaw_app)


def test_openclaw_judge_env_ignores_prm_aliases(monkeypatch) -> None:
    monkeypatch.delenv("OPENCLAW_JUDGE_M", raising=False)
    monkeypatch.delenv("OPENCLAW_JUDGE_TEMPERATURE", raising=False)
    monkeypatch.delenv("OPENCLAW_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.setenv("OPENCLAW_PRM_M", "7")
    monkeypatch.setenv("OPENCLAW_PRM_TEMPERATURE", "0.8")
    monkeypatch.setenv("OPENCLAW_PRM_MAX_TOKENS", "777")
    reloaded_module = importlib.reload(openclaw_app)
    try:
        assert reloaded_module._JUDGE_M == 1
        assert reloaded_module._JUDGE_TEMPERATURE == 0.0
        assert reloaded_module._JUDGE_MAX_TOKENS == 2048
    finally:
        importlib.reload(openclaw_app)


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


def test_openclaw_extract_feedback_text_only_uses_latest_message() -> None:
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
            {"role": "user", "content": "intermediate user turn"},
            {"role": "tool", "content": "latest tool output"},
        ],
    )
    assert feedback_text == "latest tool output"
    assert next_state_role == "tool"


def test_openclaw_extract_feedback_text_strips_untrusted_metadata_prefix() -> None:
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
            {
                "role": "user",
                "content": (
                    "Conversation info (untrusted metadata):\n"
                    "```json\n{\"message_id\":\"1\"}\n```\n\n"
                    "Sender (untrusted metadata):\n"
                    "```json\n{\"name\":\"Kevin\"}\n```\n\n"
                    "/start"
                ),
            }
        ],
    )
    assert feedback_text == "/start"
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
    assert judge_messages[0]["content"].find("question-only follow-up") >= 0


def test_openclaw_build_judge_messages_strips_think_blocks() -> None:
    pending = PendingTurn(
        external_session_id="ext-1",
        openforge_session_id="sess-1",
        trajectory_id="traj-1",
        model_name="Qwen/Qwen3-8B",
        request_message_count=2,
        assistant_text="<think>hidden reasoning</think>\nFinal answer",
        created_at=0.0,
    )
    judge_messages, _feedback_text = build_prm_judge_prompt(
        pending=pending,
        messages=[
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": "thanks"},
        ],
    )
    assert "hidden reasoning" not in judge_messages[1]["content"]
    assert "Final answer" in judge_messages[1]["content"]


def test_openclaw_build_judge_messages_defaults_new_requests_to_neutral() -> None:
    pending = PendingTurn(
        external_session_id="ext-1",
        openforge_session_id="sess-1",
        trajectory_id="traj-1",
        model_name="Qwen/Qwen3-8B",
        request_message_count=2,
        assistant_text="draft answer",
        created_at=0.0,
    )
    judge_messages, _feedback_text = build_prm_judge_prompt(
        pending=pending,
        messages=[
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "draft answer"},
            {"role": "user", "content": "/start"},
        ],
    )
    assert "Default to \\boxed{0}" in judge_messages[0]["content"]
    assert "usually neutral, not positive" in judge_messages[0]["content"]


def test_openclaw_parse_judge_output_reads_boxed_reward() -> None:
    score = parse_prm_score(r"reasoning... \boxed{0}")
    assert score == 0


def test_openclaw_parse_judge_output_keeps_boxed_fallback() -> None:
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


def test_openclaw_judge_pending_uses_local_active_session_path() -> None:
    original_active_session = openclaw_app._ActiveSession

    class _FakeAgentClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        model_dump=lambda mode="json", exclude_none=True: {
                            "choices": [
                                {
                                    "message": {
                                        "content": "Question-only follow-up, so neutral. \\boxed{0}",
                                    }
                                }
                            ]
                        }
                    )

            completions = _Completions()

        chat = _Chat()

        def close(self) -> None:
            return None

    class _FakeTrajectory:
        session_id = "sess-1"
        trajectory_id = "judge-traj"

    class _FakeSession:
        def __init__(self, gateway_target) -> None:
            self.session_id = "sess-1"

        def __enter__(self) -> "_FakeSession":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def client(self, *, purpose: str):
            assert purpose == "validation"
            return _FakeTrajectory()

        def agent_client(self, trajectory):
            assert trajectory.trajectory_id == "judge-traj"
            return _FakeAgentClient()

        def _retry_post(self, path: str, payload: dict[str, object]):
            assert path == "/discard_trajectory"
            return SimpleNamespace(raise_for_status=lambda: None)

    openclaw_app._ActiveSession = _FakeSession
    try:
        pending = PendingTurn(
            external_session_id="ext-1",
            openforge_session_id="sess-1",
            trajectory_id="traj-1",
            model_name="policy-model",
            request_message_count=2,
            assistant_text="It should happen tomorrow.",
            created_at=0.0,
        )
        decision = _judge_pending_sync(
            ("127.0.0.1", 0),
            pending,
            [
                {"role": "user", "content": "Will this happen?"},
                {"role": "assistant", "content": "It should happen tomorrow."},
                {"role": "user", "content": "Will this happen?"},
            ],
        )
    finally:
        openclaw_app._ActiveSession = original_active_session

    assert decision.reward == 0.0
    assert decision.reason == "prm_majority_vote"
    assert decision.feedback_text == "Will this happen?"


def test_openclaw_tool_followup_is_not_scored_until_user_reply() -> None:
    calls: list[str] = []

    async def fake_run_gateway_call(coro):
        return await coro

    async def fake_run_sync(fn, *args, **kwargs):
        calls.append(fn.__name__)
        if fn.__name__ == "_generate_sync":
            return ("sess-1", "traj-2", {"model": "Qwen/Qwen3-4B", "choices": [{"message": {"content": "tool step"}}]})
        if fn.__name__ == "_judge_pending_sync":
            raise AssertionError("tool follow-up should not be judged")
        if fn.__name__ == "_discard_trajectory_sync":
            return None
        if fn.__name__ == "_finish_trajectory_sync":
            raise AssertionError("tool follow-up should not finish trajectory")
        raise AssertionError(f"unexpected fn {fn.__name__}")

    class _FakeStore:
        def __init__(self) -> None:
            self.pending = PendingTurn(
                external_session_id="sess-ext",
                openforge_session_id="sess-1",
                trajectory_id="traj-1",
                model_name="Qwen/Qwen3-4B",
                request_message_count=1,
                assistant_text="previous answer",
                created_at=0.0,
            )
            self.completed = False
            self.cleared = False
            self.saved_pending = False
            self.marked_status: str | None = None

        def get_pending(self, external_session_id: str):
            return self.pending

        def complete_turn_record(self, **kwargs):
            self.completed = True

        def clear_pending(self, external_session_id: str):
            self.cleared = True

        def mark_turn_record_status(self, *, trajectory_id: str, status: str):
            assert trajectory_id == "traj-1"
            self.marked_status = status

        def save_turn_record(self, **kwargs):
            return None

        def save_pending(self, **kwargs):
            self.saved_pending = True

    original_run_gateway_call = openclaw_app._run_gateway_call
    original_run_sync = openclaw_app._run_sync
    store = _FakeStore()
    openclaw_app.app.state.store = store
    openclaw_app.app.state.gateway_target = ("127.0.0.1", 8042)
    openclaw_app._run_gateway_call = fake_run_gateway_call
    openclaw_app._run_sync = fake_run_sync
    try:
        payload = {
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "tool", "content": "tool output"}],
            "stream": False,
        }

        class _FakeRequest:
            headers = {"X-Session-Id": "sess-ext", "X-Turn-Type": "main"}

            async def json(self):
                return payload

        async def run():
            response = await openclaw_app.chat_completions(_FakeRequest())
            assert response.status_code == 200

        import asyncio

        asyncio.run(run())
    finally:
        openclaw_app._run_gateway_call = original_run_gateway_call
        openclaw_app._run_sync = original_run_sync

    assert store.completed is False
    assert store.marked_status == "skipped"
    assert store.cleared is True
    assert store.saved_pending is True
    assert calls == ["_discard_trajectory_sync", "_generate_sync"]


def test_openclaw_judge_chat_completion_url_builds_v1_path() -> None:
    assert (
        _judge_chat_completions_url("http://127.0.0.1:8042")
        == "http://127.0.0.1:8042/v1/chat/completions"
    )
