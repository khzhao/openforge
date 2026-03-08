#!/usr/bin/env python3
# Copyright 2026 openforge

"""Launch a real SGLang server through SGLangEngineRuntime.

This is a standalone smoke test for the runtime layer itself. It:
- builds an SGLangEngineSpec
- starts SGLangEngineRuntime
- waits for health + flush_cache readiness
- prints server info
- optionally sends one non-streaming /generate request

Example:
    python test_scripts/sglang_runtime_server_test_script.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --max-new-tokens 16
"""

import argparse
import json
import requests
import sys
import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import openforge.rollout.sglang.engine_runtime as runtime_module
from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.utils.networking import get_free_port

DISALLOWED_SERVER_ARG_OVERRIDES = {
    "host",
    "model_path",
    "port",
    "tokenizer_path",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Launch an SGLang backend server through SGLangEngineRuntime.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path or Hugging Face repo ID to serve.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path or repo ID. Defaults to --model-path.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Optional served model name exposed by the server.",
    )
    parser.add_argument(
        "--bind-host",
        type=str,
        default="0.0.0.0",
        help="Host the SGLang server binds to.",
    )
    parser.add_argument(
        "--connect-host",
        type=str,
        default="127.0.0.1",
        help="Host used by the local test client to reach the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Defaults to the first free port at or above 30000.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout used by the runtime client.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum time to wait for health + flush_cache readiness.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for startup readiness.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to SGLang server args.",
    )
    parser.add_argument(
        "--skip-server-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass skip_server_warmup to SGLang server args.",
    )
    parser.add_argument(
        "--weight-version",
        type=str,
        default="smoke-test",
        help="Weight version string passed to the server.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size.",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--attn-cp-size",
        type=int,
        default=1,
        help="Attention/context parallel size.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert parallel size.",
    )
    parser.add_argument(
        "--server-args-json",
        type=str,
        default=None,
        help=(
            "Extra SGLang ServerArgs entries as a JSON object. "
            f"Core keys {sorted(DISALLOWED_SERVER_ARG_OVERRIDES)!r} are reserved."
        ),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only launch and health-check the server; do not send /generate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write one short sentence about distributed training.",
        help="Prompt used for the test /generate request.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="max_new_tokens for the test /generate request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature for the test /generate request.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top_p for the test /generate request.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the server running until interrupted with Ctrl-C.",
    )
    return parser.parse_args()


def _decode_json(raw_body: str) -> Any:
    """Decode a JSON payload if possible."""
    if not raw_body:
        return None
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, Any]:
    """POST JSON and decode the response."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OSError(f"request POST {path} failed: {exc}") from exc

    raw_body = response.text
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"request POST {path} failed with status {response.status_code}: "
            f"{raw_body.strip()}"
        ) from exc
    return response.status_code, _decode_json(raw_body)


def _parse_server_arg_overrides(raw_json: str | None) -> dict[str, Any]:
    """Parse extra server args JSON."""
    if raw_json is None:
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, Mapping):
        raise ValueError("--server-args-json must decode to a JSON object")
    invalid_keys = DISALLOWED_SERVER_ARG_OVERRIDES.intersection(parsed)
    if invalid_keys:
        blocked = ", ".join(sorted(invalid_keys))
        raise ValueError(
            f"--server-args-json cannot override reserved keys: {blocked}"
        )
    return dict(parsed)


def _build_runtime_spec(args: argparse.Namespace) -> SGLangEngineSpec:
    """Construct the runtime spec used by the smoke test."""
    port = args.port if args.port is not None else get_free_port(start=30000)
    tokenizer_path = args.tokenizer_path or args.model_path
    served_model_name = args.served_model_name or args.model_path

    server_args: dict[str, Any] = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "host": args.bind_host,
        "port": port,
        "trust_remote_code": args.trust_remote_code,
        "skip_server_warmup": args.skip_server_warmup,
        "served_model_name": served_model_name,
        "weight_version": args.weight_version,
        "dp_size": args.dp_size,
        "pp_size": args.pp_size,
        "tp_size": args.tp_size,
        "attn_cp_size": args.attn_cp_size,
        "ep_size": args.ep_size,
    }
    server_args.update(_parse_server_arg_overrides(args.server_args_json))

    return SGLangEngineSpec(
        engine_id=0,
        name="sglang-runtime-smoke-test",
        role="regular",
        host=args.connect_host,
        port=port,
        bootstrap_port=None,
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        policy_version=None,
        server_args=server_args,
    )


def _build_generate_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Construct a minimal non-streaming /generate request."""
    return {
        "text": args.prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "stream": False,
    }


def _print_json(label: str, payload: Any) -> None:
    """Print a labeled JSON payload."""
    print(f"{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> int:
    """Run the SGLang runtime smoke test."""
    args = parse_args()

    runtime_module.HEALTHCHECK_TIMEOUT_SECONDS = args.startup_timeout_seconds
    runtime_module.HEALTHCHECK_POLL_INTERVAL_SECONDS = args.poll_interval_seconds

    spec = _build_runtime_spec(args)
    runtime = SGLangEngineRuntime(
        spec,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    print(f"Launching SGLangEngineRuntime at {spec.url}")
    _print_json("server_args", spec.server_args)

    started = False
    try:
        runtime.start()
        started = True

        process = runtime.process
        process_pid = None if process is None else process.pid
        print(f"Runtime healthy. child_pid={process_pid}")
        print(f"/health_generate={runtime.is_healthy()}")
        print(f"/flush_cache={runtime.flush_cache()}")
        print(f"weight_version={runtime.get_weight_version()!r}")
        _print_json("server_info", runtime.get_server_info())

        if not args.skip_generate:
            payload = _build_generate_payload(args)
            _print_json("generate_request", payload)
            status, response = _post_json(
                spec.url,
                "/generate",
                payload,
                timeout=max(args.request_timeout_seconds, 30.0),
            )
            if status != HTTPStatus.OK:
                raise RuntimeError(f"/generate returned unexpected status {status}")
            _print_json("generate_response", response)

        if args.keep_running:
            print("")
            print(f"Server is still running at {spec.url}")
            print("Press Ctrl-C to stop it.")
            while True:
                time.sleep(1.0)

        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down the runtime.")
        return 130
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime.process is not None:
            runtime.stop()
            print("Runtime stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
