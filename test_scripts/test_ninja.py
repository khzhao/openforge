# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from threading import Lock
from typing import Iterator
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import openforge.ninja as ninja
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.types import RuntimeConfig
from openforge.ninja import Gateway, Session, register


def _gateway_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


def _runtime_config(model_name: str = "model-a") -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "algo": {"kl_coef": 0.0},
            "model": {
                "model_name_or_path": model_name,
                "tokenizer_name_or_path": f"{model_name}-tokenizer",
                "attn_implementation": "sdpa",
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
                        "warmup_steps": 1,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                    },
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints": "/tmp/openforge-test-checkpoints",
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
                    "max_new_tokens": 8,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                    "spaces_between_words": True,
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
    )


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(str(self._payload))


class _FakeGateway:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active_session_id: str | None = None
        self.active_trajectory_ids: set[str] = set()
        self.next_trajectory_index = 0
        self.generate_calls: list[dict[str, object]] = []
        self.finished: list[tuple[str, float]] = []
        self.failed: list[str] = []
        self.active_generates = 0
        self.max_active_generates = 0
        self.http_client_count = 0

    def handle(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> _FakeResponse:
        if method == "GET" and path == "/health":
            return _FakeResponse(200, {"ok": True})

        if method == "POST" and path == "/start_session":
            with self._lock:
                assert self.active_session_id is None
                self.active_session_id = "sess_1"
            return _FakeResponse(200, {"session_id": "sess_1", "model": "model-a"})

        if method == "POST" and path == "/start_trajectory":
            with self._lock:
                assert self.active_session_id is not None
                self.next_trajectory_index += 1
                trajectory_id = f"traj_{self.next_trajectory_index}"
                self.active_trajectory_ids.add(trajectory_id)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "parent_trajectory_id": None,
                },
            )

        if method == "POST" and path == "/generate":
            assert payload is not None
            messages = [dict(message) for message in payload["messages"]]
            trajectory_id = str(payload["trajectory_id"])
            with self._lock:
                assert trajectory_id in self.active_trajectory_ids
                turn_index = sum(
                    1
                    for call in self.generate_calls
                    if call["trajectory_id"] == trajectory_id
                )
                self.generate_calls.append(
                    {
                        "trajectory_id": trajectory_id,
                        "messages": messages,
                        "sampling_params": dict(payload["sampling_params"]),
                    }
                )
                self.active_generates += 1
                self.max_active_generates = max(
                    self.max_active_generates,
                    self.active_generates,
                )

            time.sleep(0.02)
            text = f"reply to {messages[-1]['content']}"

            with self._lock:
                self.active_generates -= 1

            return _FakeResponse(
                200,
                {
                    "id": f"chatcmpl_{trajectory_id}_{turn_index}",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "created": 123,
                    "model": "model-a",
                    "object": "chat.completion",
                    "usage": {
                        "completion_tokens": 2,
                        "prompt_tokens": 1,
                        "total_tokens": 3,
                    },
                    "metadata": {
                        "session_id": self.active_session_id,
                        "trajectory_id": trajectory_id,
                        "token_ids": [1, 2],
                        "rollout_model_version": "v1",
                    },
                },
            )

        if method == "POST" and path == "/end_trajectory":
            assert payload is not None
            trajectory_id = str(payload["trajectory_id"])
            reward = float(payload["final_reward"])
            with self._lock:
                self.active_trajectory_ids.remove(trajectory_id)
                self.finished.append((trajectory_id, reward))
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "status": "completed",
                },
            )

        if method == "POST" and path == "/error_trajectory":
            assert payload is not None
            trajectory_id = str(payload["trajectory_id"])
            with self._lock:
                self.active_trajectory_ids.remove(trajectory_id)
                self.failed.append(trajectory_id)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "status": "failed",
                },
            )

        if method == "POST" and path == "/end_session":
            with self._lock:
                if self.active_trajectory_ids:
                    return _FakeResponse(409, {"detail": "active trajectories remain"})
                self.active_session_id = None
            return _FakeResponse(200, {"session_id": "sess_1", "status": "completed"})

        raise AssertionError(f"unexpected request: {method} {path}")


class _FakeProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False
        self.terminated = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


@contextmanager
def _patched_ninja() -> Iterator[_FakeGateway]:
    gateway = _FakeGateway()
    process_contexts: list[str] = []

    class _FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            gateway.http_client_count += 1

        def get(self, path: str) -> _FakeResponse:
            return gateway.handle("GET", path)

        def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
            return gateway.handle("POST", path, json)

        def close(self) -> None:
            return None

    class _FakeContext:
        def Process(self, *args, **kwargs) -> _FakeProcess:
            return _FakeProcess(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ninja.mp,
                "get_context",
                lambda name: process_contexts.append(name) or _FakeContext(),
            )
        )
        stack.enter_context(patch.object(ninja.httpx, "Client", _FakeHttpClient))
        yield gateway
    assert process_contexts == [] or process_contexts == ["spawn"]


def test_register_requires_active_scopes() -> None:
    @register(_gateway_config(), _runtime_config())
    def agent(client, prompt: str) -> float:
        return 1.0

    with _patched_ninja():
        with expect_raises(AssertionError, "no active gateway"):
            agent(prompt="hello")

        with agent.gateway():
            with expect_raises(AssertionError, "no active session"):
                agent(prompt="hello")


def test_gateway_and_session_work_with_registered_function() -> None:
    gateway_config = _gateway_config()
    runtime_config = _runtime_config()

    @register(gateway_config, runtime_config)
    def agent(client, prompt: str) -> float:
        response = client.generate([{"role": "user", "content": prompt}])
        message = response["choices"][0]["message"]
        return 1.0 if message["content"] else -1.0

    with _patched_ninja() as fake_gateway:
        with Gateway(gateway_config).start():
            with Session(runtime_config).start():
                reward = agent(prompt="hello")

    assert reward == 1.0
    assert fake_gateway.finished == [("traj_1", 1.0)]


def test_register_routes_explicit_messages() -> None:
    @register(_gateway_config(), _runtime_config())
    def agent(client, prompt: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        first = client.generate(messages, sampling_params={"temperature": 0.7})
        messages.append(first["choices"][0]["message"])
        messages.append({"role": "user", "content": "follow up"})
        second = client.generate(messages)
        assert first["id"] == "chatcmpl_traj_1_0"
        assert first["object"] == "chat.completion"
        assert first["model"] == "model-a"
        assert first["usage"] == {
            "completion_tokens": 2,
            "prompt_tokens": 1,
            "total_tokens": 3,
        }
        assert first["metadata"] == {
            "session_id": "sess_1",
            "trajectory_id": "traj_1",
            "token_ids": [1, 2],
            "rollout_model_version": "v1",
        }
        assert first["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "reply to hello"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ]
        assert second["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "reply to follow up"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ]
        assert second["id"] == "chatcmpl_traj_1_1"
        return 1.0

    with _patched_ninja() as fake_gateway:
        with agent.gateway():
            with agent.session():
                reward = agent(prompt="hello")

    assert reward == 1.0
    assert fake_gateway.generate_calls == [
        {
            "trajectory_id": "traj_1",
            "messages": [{"role": "user", "content": "hello"}],
            "sampling_params": {"temperature": 0.7},
        },
        {
            "trajectory_id": "traj_1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "reply to hello"},
                {"role": "user", "content": "follow up"},
            ],
            "sampling_params": {},
        },
    ]
    assert fake_gateway.finished == [("traj_1", 1.0)]


def test_register_marks_failed_trajectory_on_error() -> None:
    @register(_gateway_config(), _runtime_config())
    def agent(client, prompt: str) -> float:
        client.generate([{"role": "user", "content": prompt}])
        raise RuntimeError("boom")

    with _patched_ninja() as fake_gateway:
        with agent.gateway():
            with agent.session():
                with expect_raises(RuntimeError, "boom"):
                    agent(prompt="hello")

    assert fake_gateway.failed == ["traj_1"]
    assert fake_gateway.finished == []


def test_register_handles_concurrent_calls() -> None:
    @register(_gateway_config(), _runtime_config())
    def agent(client, prompt: str) -> float:
        response = client.generate([{"role": "user", "content": prompt}])
        message = response["choices"][0]["message"]
        return 1.0 if message["content"] else -1.0

    with _patched_ninja() as fake_gateway:
        with agent.gateway():
            with agent.session():
                with ThreadPoolExecutor(max_workers=4) as executor:
                    rewards = list(
                        executor.map(
                            lambda index: agent(prompt=f"hello {index}"),
                            range(4),
                        )
                    )

    assert rewards == [1.0, 1.0, 1.0, 1.0]
    assert len(fake_gateway.finished) == 4
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 4
    assert fake_gateway.max_active_generates > 1
    assert fake_gateway.http_client_count == 2


def test_ninja_runner_reuses_one_scope_for_many_episodes() -> None:
    call_count = 0

    @register(_gateway_config(), _runtime_config())
    def agent(client, prompt: str) -> float:
        nonlocal call_count
        call_count += 1
        response = client.generate([{"role": "user", "content": prompt}])
        assert response["choices"][0]["message"]["content"] == f"reply to {prompt}"
        return 1.0

    with _patched_ninja() as fake_gateway:
        ninja.NinjaRunner.run(
            agent,
            episodes=3,
            kwargs={"prompt": "hello"},
        )

    assert call_count == 3
    assert fake_gateway.finished == [
        ("traj_1", 1.0),
        ("traj_2", 1.0),
        ("traj_3", 1.0),
    ]
    assert fake_gateway.http_client_count == 2


def main() -> int:
    return run_tests(
        [
            test_register_requires_active_scopes,
            test_gateway_and_session_work_with_registered_function,
            test_register_routes_explicit_messages,
            test_register_marks_failed_trajectory_on_error,
            test_register_handles_concurrent_calls,
            test_ninja_runner_reuses_one_scope_for_many_episodes,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
