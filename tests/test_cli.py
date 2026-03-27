# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import importlib
import json
import signal
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

openforge_cli = importlib.import_module("openforge.cli.main")
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.types import RuntimeConfig, StartSessionRequest


class _FakeHTTPResponse:
    def __init__(self, *, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


def _runtime_config() -> RuntimeConfig:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "reference_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "attn_implementation": "flash_attention_2",
                },
                "train": {
                    "backend": "fsdp2",
                    "config": {
                        "gradient_checkpointing": False,
                        "reshard_after_forward": False,
                        "mixed_precision": {
                            "param_dtype": "bfloat16",
                            "reduce_dtype": "float32",
                        },
                        "offload": {"mode": "none", "pin_memory": False},
                        "amp": {
                            "enabled": False,
                            "precision": "float32",
                            "use_grad_scaler": False,
                        },
                        "optim": {
                            "lr": 1.0e-5,
                            "adam_beta1": 0.9,
                            "adam_beta2": 0.95,
                            "adam_eps": 1.0e-8,
                            "weight_decay": 0.0,
                            "max_grad_norm": 1.0,
                        },
                        "scheduler": {
                            "type": "constant",
                            "warmup_steps": 0,
                            "min_lr": 0.0,
                            "num_cycles": 0.5,
                        },
                    },
                    "global_batch_size": 8,
                    "mini_batch_size": 4,
                    "micro_batch_size": 1,
                    "max_rollout_policy_lag": 0,
                    "checkpoints": "./checkpoints",
                    "cpus_per_worker": 1,
                    "parallel": {
                        "data_parallel_size": 1,
                        "fsdp_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "tensor_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_parallel_size": 1,
                    },
                },
                "rollout": {
                    "backend": "sglang",
                    "request": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "max_new_tokens": 32,
                        "stop": [],
                        "stop_token_ids": [],
                        "skip_special_tokens": True,
                        "no_stop_trim": False,
                    },
                    "engine_groups": [
                        {
                            "name": "regular",
                            "worker_type": "regular",
                            "replicas": 1,
                            "num_gpus_per_replica": 1,
                            "num_cpus_per_replica": 1,
                            "parallelism": {
                                "data_parallel_size": 1,
                                "fsdp_parallel_size": 1,
                                "pipeline_parallel_size": 1,
                                "tensor_parallel_size": 1,
                                "context_parallel_size": 1,
                                "expert_parallel_size": 1,
                            },
                            "enable_memory_saver": False,
                        }
                    ],
                },
            }
        }
    )
    return request.runtime


def test_openforge_gateway_start_runs_server() -> None:
    cfg = _server_config()
    seen: dict[str, object] = {}

    def fake_run(app, *, host: str, port: int) -> None:
        seen.update(
            {
                "app": app,
                "host": host,
                "port": port,
            }
        )

    with patch.object(
        openforge_cli.GatewayServerConfig,
        "from_yaml",
        classmethod(lambda cls, path: cfg),
    ):
        fake_server = types.SimpleNamespace(create_app=lambda config: "app")
        fake_uvicorn = types.SimpleNamespace(run=fake_run)
        with patch.dict(
            sys.modules,
            {
                "openforge.gateway.server": fake_server,
                "uvicorn": fake_uvicorn,
            },
        ):
            assert (
                openforge_cli.main(
                    [
                        "gateway",
                        "start",
                        "--config",
                        "examples/gsm8k/gateway.yaml",
                    ]
                )
                == 0
            )

    assert seen["app"] == "app"
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8000


def test_openforge_session_start_rejects_existing_session() -> None:
    urls: list[str] = []

    def fake_urlopen(request, timeout: float):
        assert timeout in (2.0, openforge_cli.DEFAULT_SESSION_START_TIMEOUT_SECONDS)
        urls.append(request.full_url)
        if request.full_url.endswith("/health"):
            return _FakeHTTPResponse(status=200, payload={"ok": True})
        if request.full_url.endswith("/current_session"):
            return _FakeHTTPResponse(
                status=200,
                payload={
                    "session_id": "sess-123",
                    "model": "model-a",
                    "policy_version": 0,
                },
            )
        raise AssertionError(request.full_url)

    with patch.object(
        openforge_cli.active_state,
        "load_active_gateway_target",
        lambda: ("127.0.0.1", 8000),
    ):
        with patch.object(
            openforge_cli.RuntimeConfig, "from_yaml", lambda path: _runtime_config()
        ):
            with patch.object(openforge_cli.urllib_request, "urlopen", fake_urlopen):
                with expect_raises(SystemExit, match="active session: sess-123"):
                    openforge_cli.main(
                        [
                            "session",
                            "start",
                            "--runtime-config",
                            "examples/gsm8k/runtime.yaml",
                        ]
                    )

    assert urls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/current_session",
    ]


def test_openforge_session_stop_ends_active_session() -> None:
    requests: list[tuple[str, str]] = []

    def fake_urlopen(request, timeout: float):
        assert timeout in (2.0, openforge_cli.DEFAULT_GATEWAY_TIMEOUT_SECONDS)
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/health"):
            return _FakeHTTPResponse(status=200, payload={"ok": True})
        if request.full_url.endswith("/current_session"):
            return _FakeHTTPResponse(
                status=200,
                payload={
                    "session_id": "sess-123",
                    "model": "model-a",
                    "policy_version": 0,
                },
            )
        if request.full_url.endswith("/end_session"):
            return _FakeHTTPResponse(
                status=200,
                payload={"session_id": "sess-123", "status": "completed"},
            )
        raise AssertionError(request.full_url)

    with patch.object(
        openforge_cli.active_state,
        "load_active_gateway_target",
        lambda: ("127.0.0.1", 8000),
    ):
        with patch.object(openforge_cli.urllib_request, "urlopen", fake_urlopen):
            assert openforge_cli.main(["session", "stop"]) == 0

    assert requests == [
        ("GET", "http://127.0.0.1:8000/health"),
        ("GET", "http://127.0.0.1:8000/current_session"),
        ("POST", "http://127.0.0.1:8000/end_session"),
    ]


def test_openforge_gateway_stop_terminates_recorded_process() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "gateway": {"host": "127.0.0.1", "pid": 4321, "port": 8000},
                    "session": None,
                }
            ),
            encoding="utf-8",
        )
        seen: dict[str, object] = {}

        with patch.object(
            openforge_cli.active_state,
            "active_state_path",
            lambda: state_path,
        ):
            process_alive = True

            def fake_kill(pid: int, sig: int) -> None:
                nonlocal process_alive
                if sig == signal.SIGTERM:
                    seen.update({"pid": pid, "signal": sig})
                    process_alive = False
                    return
                if sig == 0 and process_alive:
                    return
                raise ProcessLookupError

            with patch.object(
                openforge_cli.os,
                "kill",
                fake_kill,
            ):
                with patch.object(openforge_cli.time, "sleep", lambda _: None):
                    assert openforge_cli.main(["gateway", "stop"]) == 0

    assert seen == {"pid": 4321, "signal": signal.SIGTERM}
    assert state_path.exists() is False


def test_openforge_gateway_stop_escalates_to_sigkill() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "gateway": {"host": "127.0.0.1", "pid": 4321, "port": 8000},
                    "session": None,
                }
            ),
            encoding="utf-8",
        )
        signals: list[int] = []

        with patch.object(
            openforge_cli.active_state,
            "active_state_path",
            lambda: state_path,
        ):
            alive_checks = {"count": 0}

            def fake_kill(pid: int, sig: int) -> None:
                if sig == signal.SIGTERM:
                    signals.append(sig)
                    return
                if sig == 0:
                    alive_checks["count"] += 1
                    if alive_checks["count"] < 100:
                        return
                    raise ProcessLookupError
                if sig == signal.SIGKILL:
                    signals.append(sig)
                    return
                raise ProcessLookupError

            with patch.object(
                openforge_cli.os,
                "kill",
                fake_kill,
            ):
                with patch.object(openforge_cli.time, "sleep", lambda _: None):
                    timeline = iter([0.0, 10.0])
                    with patch.object(
                        openforge_cli.time,
                        "monotonic",
                        lambda: next(timeline),
                    ):
                        assert openforge_cli.main(["gateway", "stop"]) == 0

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert state_path.exists() is False


def test_openforge_watch_prints_status_once() -> None:
    requests: list[tuple[str, str]] = []

    def fake_urlopen(request, timeout: float):
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/health"):
            return _FakeHTTPResponse(status=200, payload={"ok": True})
        if request.full_url.endswith("/status"):
            return _FakeHTTPResponse(
                status=200,
                payload={
                    "session_id": "sess-123",
                    "wall_time_s": 3.5,
                    "gateway": {"heartbeat_age_s": 0.2, "pending_generate_count": 1},
                    "train": {
                        "active": True,
                        "heartbeat_age_s": 0.1,
                        "last_update_age_s": 1.2,
                        "global_step": 4,
                        "policy_version": 4,
                        "latest_update": {
                            "reward_mean": 0.5,
                            "grad_norm": 1.25,
                            "lr": 1.0e-5,
                        },
                    },
                    "rollout": {
                        "heartbeat_age_s": 0.3,
                        "latest_published_train_version": 4,
                        "min_weight_version": 3,
                        "max_weight_version": 4,
                        "stale_worker_count": 1,
                        "workers": {},
                    },
                    "cluster": {},
                },
            )
        raise AssertionError(request.full_url)

    with patch.object(
        openforge_cli.active_state,
        "load_active_gateway_target",
        lambda: ("127.0.0.1", 8000),
    ):
        with patch.object(openforge_cli.urllib_request, "urlopen", fake_urlopen):
            with patch.object(openforge_cli.sys.stdout, "isatty", lambda: False):
                with patch("builtins.print") as print_mock:
                    assert openforge_cli.main(["watch", "--once"]) == 0

    printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
    assert "session sess-123" in printed
    assert "min_weight_version=3" in printed
    assert requests == [
        ("GET", "http://127.0.0.1:8000/health"),
        ("GET", "http://127.0.0.1:8000/status"),
    ]


def test_openforge_watch_prints_error_state_once_on_status_failure() -> None:
    requests: list[tuple[str, str]] = []

    def fake_urlopen(request, timeout: float):
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/health"):
            return _FakeHTTPResponse(status=200, payload={"ok": True})
        if request.full_url.endswith("/status"):
            raise openforge_cli.urllib_error.URLError("timed out")
        raise AssertionError(request.full_url)

    with patch.object(
        openforge_cli.active_state,
        "load_active_gateway_target",
        lambda: ("127.0.0.1", 8000),
    ):
        with patch.object(openforge_cli.urllib_request, "urlopen", fake_urlopen):
            with patch.object(openforge_cli.sys.stdout, "isatty", lambda: False):
                with patch("builtins.print") as print_mock:
                    assert openforge_cli.main(["watch", "--once"]) == 0

    printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
    assert "watch state=errored" in printed
    assert "waiting for next refresh attempt..." in printed
    assert requests == [
        ("GET", "http://127.0.0.1:8000/health"),
        ("GET", "http://127.0.0.1:8000/status"),
    ]


def main() -> int:
    return run_tests(
        [
            test_openforge_gateway_start_runs_server,
            test_openforge_session_start_rejects_existing_session,
            test_openforge_session_stop_ends_active_session,
            test_openforge_gateway_stop_terminates_recorded_process,
            test_openforge_gateway_stop_escalates_to_sigkill,
            test_openforge_watch_prints_status_once,
            test_openforge_watch_prints_error_state_once_on_status_failure,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
