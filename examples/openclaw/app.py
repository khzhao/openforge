# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openforge import active_state
from openforge.ninja.session import _ActiveSession, _TrajectoryClient

from .reward import (
    RewardDecision,
    build_prm_judge_prompt,
    majority_vote,
    parse_prm_score,
)
from .session_state import SessionStateStore

__all__ = ["app", "main"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8012
_LOG = logging.getLogger(__name__)
_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}
_JUDGE_M = int(os.environ.get("OPENCLAW_JUDGE_M", "1"))
_JUDGE_TEMPERATURE = float(os.environ.get("OPENCLAW_JUDGE_TEMPERATURE", "0.0"))
_JUDGE_MAX_TOKENS = int(os.environ.get("OPENCLAW_JUDGE_MAX_TOKENS", "2048"))


def _flatten_message_content(content: Any) -> str:
    """Extract plain text from multimodal content lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenClaw message payloads into OpenForge-compatible messages."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "developer":
            item["role"] = "system"
        raw_content = item.get("content")
        if not isinstance(raw_content, str) and raw_content is not None:
            item["content"] = _flatten_message_content(raw_content)
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            item["tool_calls"] = [_normalize_tool_call(call) for call in tool_calls]
        normalized.append(item)
    return normalized


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Normalize assistant tool calls for OpenForge's typed request schema."""
    item = dict(tool_call)
    function = item.get("function")
    if isinstance(function, dict):
        normalized_function = dict(function)
        arguments = normalized_function.get("arguments")
        if isinstance(arguments, dict):
            normalized_function["arguments"] = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        item["function"] = normalized_function
    return item


def _normalize_forward_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenClaw request into the subset OpenForge accepts."""
    forward_payload = {
        key: value
        for key, value in payload.items()
        if key not in _NON_STANDARD_BODY_KEYS
    }
    messages = forward_payload.get("messages")
    if isinstance(messages, list):
        forward_payload["messages"] = _normalize_messages(
            [_as_message_dict(item) for item in messages]
        )
    forward_payload["stream"] = False
    forward_payload.pop("stream_options", None)
    return forward_payload


def _default_gateway_base_url() -> str:
    host, port = active_state.load_active_gateway_target()
    return f"http://{host}:{port}"


def _default_state_path() -> Path:
    cache_home = Path(
        os.environ.get("OPENFORGE_CACHE_HOME", Path.home() / ".cache")
    )
    return cache_home / "openforge" / "examples" / "openclaw.sqlite3"


@asynccontextmanager
async def _lifespan(_: FastAPI):
    gateway_base_url = os.environ.get("OPENFORGE_GATEWAY_BASE_URL")
    if gateway_base_url is None:
        gateway_base_url = _default_gateway_base_url()
    state_path = Path(os.environ.get("OPENCLAW_STATE_DB", _default_state_path()))
    gateway_target = _parse_gateway_target(gateway_base_url)
    store = SessionStateStore(state_path)
    app.state.gateway_base_url = gateway_base_url
    app.state.gateway_target = gateway_target
    app.state.store = store
    try:
        yield
    finally:
        pass


app = FastAPI(
    title="OpenClaw Middleware Example",
    lifespan=_lifespan,
)


@app.get("/health")
async def health() -> dict[str, object]:
    """Return middleware health."""
    return {
        "ok": True,
        "gateway_base_url": app.state.gateway_base_url,
        "judge_mode": "local",
    }


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Pass through the upstream models list."""
    payload = await _run_gateway_call(
        _run_sync(_list_models_sync, app.state.gateway_target)
    )
    return JSONResponse(payload)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """Accept OpenClaw provider requests and map them to OpenForge trajectories."""
    payload, stream_requested = await _read_json_payload(request)
    messages = payload.get("messages")
    _LOG.info(
        "openclaw request: stream=%r messages=%s x_session_id=%r x_turn_type=%r model=%r",
        stream_requested,
        len(messages) if isinstance(messages, list) else None,
        request.headers.get("X-Session-Id"),
        request.headers.get("X-Turn-Type"),
        payload.get("model"),
    )
    if not isinstance(messages, list) or not messages:
        _LOG.warning("rejecting request: messages must be a non-empty list")
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    payload = _normalize_forward_payload(payload)
    _LOG.warning(
        "openclaw normalized request: messages=%s max_completion_tokens=%r max_tokens=%r tools=%s model=%r",
        len(payload.get("messages", []))
        if isinstance(payload.get("messages"), list)
        else None,
        payload.get("max_completion_tokens"),
        payload.get("max_tokens"),
        len(payload.get("tools", []))
        if isinstance(payload.get("tools"), list)
        else 0,
        payload.get("model"),
    )

    external_session_id = _resolve_external_session_id(request)
    turn_type = request.headers.get("X-Turn-Type", "main").strip().lower() or "main"
    purpose = "train" if turn_type == "main" else "validation"
    _LOG.info(
        "openclaw routing: session_id=%r turn_type=%r purpose=%r",
        external_session_id,
        turn_type,
        purpose,
    )

    store: SessionStateStore = app.state.store

    if purpose == "train":
        pending = store.get_pending(external_session_id)
        if pending is not None:
            next_state_message = _as_message_dict(messages[-1])
            next_state_role = str(next_state_message.get("role", "user"))
            if next_state_role == "tool":
                await _run_gateway_call(
                    _run_sync(
                        _discard_trajectory_sync,
                        session=None,
                        session_id=pending.openforge_session_id,
                        trajectory_id=pending.trajectory_id,
                    )
                )
                store.mark_turn_record_status(
                    trajectory_id=pending.trajectory_id,
                    status="skipped",
                )
                store.clear_pending(external_session_id)
            else:
                decision = await _run_gateway_call(
                    _run_sync(
                        _judge_pending_sync,
                        app.state.gateway_target,
                        pending,
                        [next_state_message],
                    )
                )
                await _run_gateway_call(
                    _run_sync(
                        _finish_trajectory_sync,
                        app.state.gateway_target,
                        pending.openforge_session_id,
                        pending.trajectory_id,
                        decision.reward,
                    )
                )
                store.complete_turn_record(
                    trajectory_id=pending.trajectory_id,
                    reward=decision.reward,
                    reason=decision.reason,
                    feedback_text=decision.feedback_text,
                )
                store.clear_pending(external_session_id)

    session_id, trajectory_id, response_payload = await _run_gateway_call(
        _run_sync(
            _generate_sync,
            app.state.gateway_target,
            payload,
            purpose,
        )
    )
    store.save_turn_record(
        trajectory_id=trajectory_id,
        external_session_id=external_session_id,
        openforge_session_id=session_id,
        model_name=str(response_payload.get("model", payload.get("model", ""))),
        turn_type=turn_type,
        purpose=purpose,
        request_messages=[_as_message_dict(item) for item in messages],
        normalized_messages=[_as_message_dict(item) for item in payload["messages"]],
        assistant_text=_assistant_text(response_payload),
    )
    if purpose == "train":
        store.save_pending(
            external_session_id=external_session_id,
            openforge_session_id=session_id,
            trajectory_id=trajectory_id,
            model_name=str(response_payload.get("model", payload.get("model", ""))),
            request_message_count=len(messages),
            assistant_text=_assistant_text(response_payload),
        )
    else:
        store.mark_turn_record_status(
            trajectory_id=trajectory_id,
            status="discarded",
        )
    if stream_requested:
        return StreamingResponse(
            _stream_response_payload(response_payload),
            media_type="text/event-stream",
        )
    return JSONResponse(response_payload)


def main() -> int:
    """Run the OpenClaw middleware example."""
    uvicorn.run(
        "examples.openclaw.app:app",
        host=os.environ.get("OPENCLAW_MIDDLEWARE_HOST", _DEFAULT_HOST),
        port=int(
            os.environ.get("OPENCLAW_MIDDLEWARE_PORT", str(_DEFAULT_PORT))
        ),
        reload=False,
    )
    return 0


async def _run_gateway_call(coro):
    try:
        return await coro
    except AssertionError as exc:
        _LOG.exception("middleware upstream assertion failure")
        if "no active session recorded" in str(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "no active OpenForge session; start `openforge gateway` and "
                    "`openforge session` first"
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"OpenForge gateway interaction failed: {exc}",
        ) from exc
    except Exception as exc:
        _LOG.exception("middleware upstream request failed")
        raise HTTPException(
            status_code=502,
            detail=f"OpenForge gateway interaction failed: {exc}",
        ) from exc


async def _run_sync(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def _parse_gateway_target(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url.rstrip("/"))
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        raise RuntimeError(
            "OpenClaw example requires OPENFORGE_GATEWAY_BASE_URL with an explicit "
            f"host and port, got {base_url!r}"
        )
    return host, port


def _judge_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _list_models_sync(gateway_target: tuple[str, int]) -> dict[str, Any]:
    with _ActiveSession(gateway_target) as session:
        response = session.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return dict(payload)


def _generate_sync(
    gateway_target: tuple[str, int],
    payload: dict[str, Any],
    purpose: str,
) -> tuple[str, str, dict[str, Any]]:
    with _ActiveSession(gateway_target) as session:
        trajectory = session.client(purpose=purpose)
        try:
            trajectory._start()
        except Exception:
            _LOG.exception(
                "failed to start trajectory: purpose=%s trajectory_id=%s payload=%s",
                purpose,
                trajectory.trajectory_id,
                json.dumps(payload, ensure_ascii=False),
            )
            raise
        agent_client = session.agent_client(trajectory)
        try:
            response = agent_client.chat.completions.create(**payload)
        except Exception:
            _LOG.exception(
                "downstream chat completion failed: purpose=%s trajectory_id=%s payload=%s",
                purpose,
                trajectory.trajectory_id,
                json.dumps(payload, ensure_ascii=False),
            )
            if purpose == "train":
                try:
                    trajectory.fail()
                except Exception:
                    _LOG.exception(
                        "failed to mark errored train trajectory: %s",
                        trajectory.trajectory_id,
                    )
            else:
                try:
                    _discard_trajectory_sync(
                        session=session,
                        session_id=trajectory.session_id,
                        trajectory_id=trajectory.trajectory_id,
                    )
                except Exception:
                    _LOG.exception(
                        "failed to discard validation trajectory: %s",
                        trajectory.trajectory_id,
                    )
            raise
        finally:
            agent_client.close()

        if purpose != "train":
            _discard_trajectory_sync(
                session=session,
                session_id=trajectory.session_id,
                trajectory_id=trajectory.trajectory_id,
            )

        return (
            trajectory.session_id,
            trajectory.trajectory_id,
            response.model_dump(mode="json", exclude_none=True),
        )


def _judge_pending_sync(
    gateway_target: tuple[str, int],
    pending,
    messages: list[dict[str, Any]],
):
    judge_messages, feedback_text = build_prm_judge_prompt(
        pending=pending,
        messages=messages,
    )
    scores: list[int | None] = []
    outputs: list[str] = []
    with _ActiveSession(gateway_target) as session:
        if session.session_id != pending.openforge_session_id:
            raise RuntimeError(
                "active OpenForge session changed while the middleware still held a "
                f"pending trajectory: {session.session_id} != "
                f"{pending.openforge_session_id}"
            )
        for _index in range(_JUDGE_M):
            trajectory = session.client(purpose="validation")
            agent_client = session.agent_client(trajectory)
            try:
                response = agent_client.chat.completions.create(
                    model=pending.model_name,
                    messages=judge_messages,
                    temperature=_JUDGE_TEMPERATURE,
                    top_p=1.0,
                    top_k=1,
                    max_completion_tokens=_JUDGE_MAX_TOKENS,
                )
            except Exception:
                _LOG.exception(
                    "judge request failed: pending_trajectory_id=%s session_id=%s",
                    pending.trajectory_id,
                    pending.openforge_session_id,
                )
                raise
            finally:
                agent_client.close()
                _discard_trajectory_sync(
                    session=session,
                    session_id=trajectory.session_id,
                    trajectory_id=trajectory.trajectory_id,
                )
            judge_text = _assistant_text(
                response.model_dump(mode="json", exclude_none=True)
            )
            outputs.append(judge_text)
            scores.append(parse_prm_score(judge_text))
    _LOG.warning(
        "judge output: trajectory_id=%s scores=%r text=%r",
        pending.trajectory_id,
        scores,
        outputs[-1] if outputs else "",
    )
    reward = majority_vote(scores)
    reason = "prm_majority_vote"
    if not outputs:
        reason = "prm_no_output"
    elif all(score is None for score in scores):
        reason = "prm_parse_fallback"
    return RewardDecision(
        reward=reward,
        reason=reason,
        feedback_text=feedback_text,
    )


def _finish_trajectory_sync(
    gateway_target: tuple[str, int],
    session_id: str,
    trajectory_id: str,
    reward: float,
) -> None:
    with _ActiveSession(gateway_target) as session:
        if session.session_id != session_id:
            raise RuntimeError(
                "active OpenForge session changed while the middleware still held a "
                f"pending trajectory: {session.session_id} != {session_id}"
            )
        trajectory = _TrajectoryClient(
            post=session.post,
            retry_post=session._retry_post,
            session_id=session_id,
            trajectory_id=trajectory_id,
            used=True,
            purpose="train",
        )
        trajectory.finish(reward)


def _discard_trajectory_sync(
    *,
    session: _ActiveSession | None,
    session_id: str,
    trajectory_id: str,
) -> None:
    if session is not None:
        response = session._retry_post(
            "/discard_trajectory",
            {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
            },
        )
        response.raise_for_status()
        return
    host, port = active_state.load_active_gateway_target()
    with _ActiveSession((host, port)) as active_session:
        if active_session.session_id != session_id:
            raise RuntimeError(
                "active OpenForge session changed while discarding a skipped "
                f"trajectory: {active_session.session_id} != {session_id}"
            )
        response = active_session._retry_post(
            "/discard_trajectory",
            {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
            },
        )
        response.raise_for_status()


async def _read_json_payload(request: Request) -> tuple[dict[str, Any], bool]:
    try:
        payload = await request.json()
    except Exception as exc:
        _LOG.warning("rejecting request: invalid JSON payload: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"invalid JSON payload: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        _LOG.warning("rejecting request: request body must be a JSON object")
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    stream_requested = bool(payload.get("stream"))
    if stream_requested:
        _LOG.warning("coercing request: stream=true -> stream=false")
        payload["stream"] = False
    return payload, stream_requested


def _resolve_external_session_id(request: Request) -> str:
    external_session_id = request.headers.get("X-Session-Id")
    if external_session_id is None or not external_session_id.strip():
        _LOG.warning(
            "rejecting request: missing external session id (x_session_id=%r)",
            request.headers.get("X-Session-Id"),
        )
        raise HTTPException(
            status_code=400,
            detail="missing external session id; set X-Session-Id in OpenClaw",
        )
    return external_session_id.strip()


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


def _stream_response_payload(response_payload: dict[str, Any]):
    response_id = str(response_payload.get("id", "chatcmpl-openforge"))
    created = int(response_payload.get("created", 0))
    model = str(response_payload.get("model", ""))
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        yield "data: [DONE]\n\n"
        return

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    index = int(first_choice.get("index", 0))
    finish_reason = first_choice.get("finish_reason")
    message = first_choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    delta: dict[str, Any] = {"role": "assistant"}
    if isinstance(content, str) and content:
        delta["content"] = content
    if isinstance(tool_calls, list) and tool_calls:
        delta["tool_calls"] = tool_calls

    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk, separators=(',', ':'))}\n\n"

    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _as_message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
