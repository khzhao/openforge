# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import ValidationError

from openforge import active_state
from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import RuntimeConfig

DEFAULT_GATEWAY_TIMEOUT_SECONDS = 60.0


def _run_gateway_start(args: argparse.Namespace) -> int:
    import uvicorn

    from openforge.gateway.server import create_app

    cfg = GatewayServerConfig.from_yaml(args.config)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.gateway.host, port=cfg.gateway.port)
    return 0


def _run_gateway_stop(_: argparse.Namespace) -> int:
    pid = active_state.load_active_gateway_pid()
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    active_state.clear_active_gateway()
    return 0


def add_session_start_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the top-level session start arguments."""
    parser.add_argument(
        "--runtime-config",
        type=str,
        required=True,
        help="Path to the runtime YAML config to submit to the gateway.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GATEWAY_TIMEOUT_SECONDS,
        help="Timeout in seconds for waiting on and talking to the gateway.",
    )


def add_session_stop_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the top-level session stop arguments."""
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GATEWAY_TIMEOUT_SECONDS,
        help="Timeout in seconds for talking to the gateway.",
    )


def _try_request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, object] | None,
    timeout: float,
) -> tuple[int, dict[str, object]] | None:
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = int(response.status)
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib_error.URLError:
        return None

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gateway returned invalid JSON: {raw}") from exc
    if not isinstance(response, dict):
        raise SystemExit(f"gateway returned unexpected payload: {response!r}")
    return status, response


def _request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, object] | None,
    timeout: float,
) -> tuple[int, dict[str, object]]:
    response = _try_request_json(
        method=method,
        url=url,
        payload=payload,
        timeout=timeout,
    )
    if response is None:
        raise SystemExit(f"failed to reach gateway at {url}")
    return response


def _wait_for_gateway(*, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _try_request_json(
            method="GET",
            url=f"{base_url}/health",
            payload=None,
            timeout=min(timeout, 2.0),
        )
        if response is not None and response[0] == 200:
            return
        time.sleep(1.0)
    raise SystemExit(f"timed out waiting for gateway at {base_url}")


def _current_session(*, base_url: str, timeout: float) -> dict[str, object] | None:
    status, response = _request_json(
        method="GET",
        url=f"{base_url}/current_session",
        payload=None,
        timeout=timeout,
    )
    if status == 404:
        return None
    if status != 200:
        raise SystemExit(
            f"gateway preflight failed with status {status}: "
            f"{json.dumps(response, sort_keys=True)}"
        )
    return response


def _run_session_start(args: argparse.Namespace) -> int:
    runtime_config = RuntimeConfig.from_yaml(args.runtime_config)
    host, port = active_state.load_active_gateway_target()
    base_url = f"http://{host}:{port}"
    _wait_for_gateway(base_url=base_url, timeout=args.timeout)
    session = _current_session(base_url=base_url, timeout=args.timeout)
    if session is not None:
        raise SystemExit(
            f"gateway already has an active session: {session.get('session_id')}"
        )
    status, response = _request_json(
        method="POST",
        url=f"{base_url}/start_session",
        payload={"runtime": runtime_config.model_dump(mode="json")},
        timeout=args.timeout,
    )
    if status != 200:
        raise SystemExit(
            f"gateway request failed with status {status}: "
            f"{json.dumps(response, sort_keys=True)}"
        )
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


def _run_session_stop(args: argparse.Namespace) -> int:
    host, port = active_state.load_active_gateway_target()
    base_url = f"http://{host}:{port}"
    _wait_for_gateway(base_url=base_url, timeout=args.timeout)
    session = _current_session(base_url=base_url, timeout=args.timeout)
    if session is None:
        raise SystemExit("no active session")
    session_id = session.get("session_id")
    if not isinstance(session_id, str):
        raise SystemExit(f"gateway returned unexpected session payload: {session!r}")
    status, response = _request_json(
        method="POST",
        url=f"{base_url}/end_session",
        payload={"session_id": session_id},
        timeout=args.timeout,
    )
    if status != 200:
        raise SystemExit(
            f"gateway request failed with status {status}: "
            f"{json.dumps(response, sort_keys=True)}"
        )
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level OpenForge CLI parser."""
    parser = argparse.ArgumentParser(prog="openforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gateway_parser = subparsers.add_parser("gateway", help="Gateway commands.")
    gateway_subparsers = gateway_parser.add_subparsers(
        dest="gateway_command",
        required=True,
    )
    gateway_start_parser = gateway_subparsers.add_parser(
        "start",
        help="Run the gateway API server.",
    )
    gateway_start_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Gateway server YAML config file.",
    )
    gateway_start_parser.set_defaults(handler=_run_gateway_start)

    gateway_stop_parser = gateway_subparsers.add_parser(
        "stop",
        help="Stop the recorded gateway process.",
    )
    gateway_stop_parser.set_defaults(handler=_run_gateway_stop)

    session_parser = subparsers.add_parser("session", help="Session commands.")
    session_subparsers = session_parser.add_subparsers(
        dest="session_command",
        required=True,
    )
    session_start_parser = session_subparsers.add_parser(
        "start",
        help="Submit a runtime YAML config to /start_session.",
    )
    add_session_start_arguments(session_start_parser)
    session_start_parser.set_defaults(handler=_run_session_start)

    session_stop_parser = session_subparsers.add_parser(
        "stop",
        help="End the active session on the recorded gateway.",
    )
    add_session_stop_arguments(session_stop_parser)
    session_stop_parser.set_defaults(handler=_run_session_stop)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the OpenForge CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AssertionError, ValidationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
