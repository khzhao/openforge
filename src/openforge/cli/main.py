# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import socket
import shutil
import signal
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import ValidationError

from openforge import active_state
from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import RuntimeConfig
from openforge.logging import render_status, render_watch_error

DEFAULT_GATEWAY_TIMEOUT_SECONDS = 60.0
DEFAULT_SESSION_START_TIMEOUT_SECONDS = 600.0
DEFAULT_WATCH_INTERVAL_SECONDS = 1.009


class GatewayRequestTimeoutError(RuntimeError):
    """Raised when the gateway accepts a connection but does not reply in time."""


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
    else:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
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
        default=DEFAULT_SESSION_START_TIMEOUT_SECONDS,
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
    except (TimeoutError, socket.timeout) as exc:
        raise GatewayRequestTimeoutError(
            f"timed out waiting for gateway response from {url}"
        ) from exc
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
    try:
        response = _try_request_json(
            method=method,
            url=url,
            payload=payload,
            timeout=timeout,
        )
    except GatewayRequestTimeoutError as exc:
        raise SystemExit(str(exc)) from exc
    if response is None:
        raise SystemExit(f"failed to reach gateway at {url}")
    return response


def _wait_for_gateway(*, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = _try_request_json(
                method="GET",
                url=f"{base_url}/health",
                payload=None,
                timeout=min(timeout, 2.0),
            )
        except GatewayRequestTimeoutError:
            response = None
        if response is not None and response[0] == 200:
            return
        time.sleep(1.0)
    raise SystemExit(f"timed out waiting for gateway at {base_url}")


def _wait_for_active_gateway_target(*, timeout: float) -> tuple[str, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return active_state.load_active_gateway_target()
        except AssertionError:
            pass
        time.sleep(1.0)
    raise SystemExit(
        "no active gateway recorded; "
        "run `python -m openforge.cli.main gateway start --config ...` first"
    )


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
    host, port = _wait_for_active_gateway_target(timeout=args.timeout)
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
    host, port = _wait_for_active_gateway_target(timeout=args.timeout)
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


def _run_watch(args: argparse.Namespace) -> int:
    host, port = _wait_for_active_gateway_target(timeout=args.timeout)
    base_url = f"http://{host}:{port}"
    _wait_for_gateway(base_url=base_url, timeout=args.timeout)
    use_tty = sys.stdout.isatty()
    if use_tty and not args.once:
        from openforge.cli.watch_tui import run_textual_watch

        return run_textual_watch(
            fetch_snapshot=lambda: _fetch_watch_snapshot(
                base_url=base_url,
                timeout=args.timeout,
            ),
            interval=args.interval,
        )

    while True:
        terminal_width = shutil.get_terminal_size(fallback=(100, 30)).columns
        ok, payload = _fetch_watch_snapshot(base_url=base_url, timeout=args.timeout)
        output = (
            render_status(payload, use_color=use_tty, width=terminal_width)
            if ok
            else render_watch_error(
                str(payload), use_color=use_tty, width=terminal_width
            )
        )
        print(output, flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


def _fetch_watch_snapshot(
    *, base_url: str, timeout: float
) -> tuple[bool, dict[str, object] | str]:
    try:
        status, response = _request_json(
            method="GET",
            url=f"{base_url}/status",
            payload=None,
            timeout=timeout,
        )
    except SystemExit as exc:
        return False, str(exc)
    if status != 200:
        return (
            False,
            f"gateway status failed with status {status}: "
            f"{json.dumps(response, sort_keys=True)}",
        )
    return True, response


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

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll the gateway status endpoint and print a live session view.",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_WATCH_INTERVAL_SECONDS,
        help=(
            "Seconds to wait between status refreshes. Defaults to 1.009s "
            "(1009 ms, prime) to avoid phase-locking with whole-second worker "
            "health loops."
        ),
    )
    watch_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GATEWAY_TIMEOUT_SECONDS,
        help="Timeout in seconds for talking to the gateway.",
    )
    watch_parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch and print one status snapshot.",
    )
    watch_parser.set_defaults(handler=_run_watch)

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
