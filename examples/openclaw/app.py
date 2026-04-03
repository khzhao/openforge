# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openforge import active_state
from openforge.ninja.session import _ActiveSession, _TrajectoryClient

from .reward import build_judge_messages, parse_judge_output
from .session_state import SessionStateStore

__all__ = ["app", "main"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8011


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
    gateway_base_url = os.environ.get("OPENCLAW_OPENFORGE_BASE_URL")
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
    payload = await _read_json_payload(request)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    external_session_id = _resolve_external_session_id(request, payload)
    turn_type = request.headers.get("X-Turn-Type", "main").strip().lower() or "main"
    purpose = "train" if turn_type == "main" else "validation"

    store: SessionStateStore = app.state.store

    if purpose == "train":
        pending = store.get_pending(external_session_id)
        if pending is not None:
            decision = await _run_gateway_call(
                _run_sync(
                    _judge_pending_sync,
                    app.state.gateway_target,
                    pending,
                    [_as_message_dict(item) for item in messages],
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
            store.clear_pending(external_session_id)

    session_id, trajectory_id, response_payload = await _run_gateway_call(
        _run_sync(
            _generate_sync,
            app.state.gateway_target,
            payload,
            purpose,
        )
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
    return JSONResponse(response_payload)


def main() -> int:
    """Run the OpenClaw middleware example."""
    uvicorn.run(
        "examples.openclaw.app:app",
        host=os.environ.get("OPENCLAW_HOST", _DEFAULT_HOST),
        port=int(os.environ.get("OPENCLAW_PORT", str(_DEFAULT_PORT))),
        reload=False,
    )
    return 0


async def _run_gateway_call(coro):
    try:
        return await coro
    except AssertionError as exc:
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
        raise HTTPException(
            status_code=502,
            detail=f"OpenForge gateway interaction failed: {exc}",
        ) from exc


async def _run_sync(fn, *args):
    return await asyncio.to_thread(fn, *args)


def _parse_gateway_target(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url.rstrip("/"))
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        raise RuntimeError(
            "OpenClaw example requires OPENCLAW_OPENFORGE_BASE_URL with an explicit "
            f"host and port, got {base_url!r}"
        )
    return host, port


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
        agent_client = session.agent_client(trajectory)
        try:
            response = agent_client.chat.completions.create(**payload)
        except Exception:
            if purpose == "train":
                trajectory.fail()
            else:
                _discard_trajectory_sync(
                    session=session,
                    session_id=trajectory.session_id,
                    trajectory_id=trajectory.trajectory_id,
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
    judge_messages, feedback_text = build_judge_messages(
        pending=pending,
        messages=messages,
    )
    with _ActiveSession(gateway_target) as session:
        if session.session_id != pending.openforge_session_id:
            raise RuntimeError(
                "active OpenForge session changed while the middleware still held a "
                f"pending trajectory: {session.session_id} != "
                f"{pending.openforge_session_id}"
            )
        trajectory = session.client(purpose="validation")
        agent_client = session.agent_client(trajectory)
        try:
            response = agent_client.chat.completions.create(
                model=pending.model_name,
                messages=judge_messages,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_completion_tokens=128,
            )
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
    return parse_judge_output(
        text=judge_text,
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
    session: _ActiveSession,
    session_id: str,
    trajectory_id: str,
) -> None:
    response = session._retry_post(
        "/discard_trajectory",
        {
            "session_id": session_id,
            "trajectory_id": trajectory_id,
        },
    )
    response.raise_for_status()


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"invalid JSON payload: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    if bool(payload.get("stream")):
        raise HTTPException(
            status_code=400,
            detail="stream=true is not supported by the OpenClaw example middleware",
        )
    return payload


def _resolve_external_session_id(request: Request, payload: dict[str, Any]) -> str:
    external_session_id = request.headers.get("X-Session-Id")
    if external_session_id is None:
        user_value = payload.get("user")
        if isinstance(user_value, str) and user_value.strip():
            external_session_id = user_value
    if external_session_id is None or not external_session_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "missing external session id; set X-Session-Id in OpenClaw or "
                "provide a stable user field"
            ),
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


def _as_message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
