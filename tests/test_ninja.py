# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import json
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator
from unittest.mock import patch

import httpx
from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import openforge.ninja as ninja
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.types import RuntimeConfig, StartSessionRequest


def _gateway_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


def _runtime_config(*, global_batch_size: int) -> RuntimeConfig:
    return StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "model-a",
                    "tokenizer_name_or_path": "model-a-tokenizer",
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
                            "warmup_steps": 1,
                            "min_lr": 0.0,
                            "num_cycles": 0.5,
                        },
                    },
                    "global_batch_size": global_batch_size,
                    "mini_batch_size": 1,
                    "micro_batch_size": 1,
                    "max_rollout_policy_lag": 0,
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
            },
        }
    ).runtime


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
    def __init__(self, *, active_session_id: str | None = "sess_1") -> None:
        self._lock = Lock()
        self.active_session_id = active_session_id
        self.active_trajectory_ids: set[str] = set()
        self.trajectory_statuses: dict[str, str] = {}
        self.next_trajectory_index = 0
        self.generate_calls: list[dict[str, object]] = []
        self.finished: list[tuple[str, float]] = []
        self.failed: list[str] = []
        self.policy_version = 0
        self.active_generates = 0
        self.max_active_generates = 0
        self.http_client_count = 0
        self.status_calls = 0
        self.validation_updates: list[dict[str, object]] = []
        self.rollout_min_weight_versions: list[int] = []
        self._transport_errors: dict[
            tuple[str, str], tuple[int, type[httpx.TransportError]]
        ] = {}

    def inject_transport_error(
        self,
        method: str,
        path: str,
        *,
        count: int = 1,
        error_type: type[httpx.TransportError] = httpx.ReadError,
    ) -> None:
        self._transport_errors[(method, path)] = (count, error_type)

    def inject_read_error(self, method: str, path: str, *, count: int = 1) -> None:
        self.inject_transport_error(method, path, count=count)

    def handle(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> _FakeResponse:
        read_error_key = (method, path)
        with self._lock:
            remaining, error_type = self._transport_errors.get(
                read_error_key, (0, httpx.ReadError)
            )
            if remaining > 0:
                self._transport_errors[read_error_key] = (remaining - 1, error_type)
                raise error_type(f"transient transport error for {method} {path}")

        if method == "GET" and path == "/current_session":
            if self.active_session_id is None:
                return _FakeResponse(404, {"detail": "no active session"})
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "model": "model-a",
                    "policy_version": self.policy_version,
                },
            )

        if method == "GET" and path == "/status":
            with self._lock:
                self.status_calls += 1
                min_weight_version = (
                    self.rollout_min_weight_versions.pop(0)
                    if self.rollout_min_weight_versions
                    else self.policy_version
                )
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "wall_time_s": 0.0,
                    "gateway": {
                        "heartbeat_age_s": 0.0,
                        "pending_generate_count": 0,
                    },
                    "train": {
                        "policy_version": self.policy_version,
                        "latest_update": None,
                    },
                    "rollout": {
                        "latest_published_train_version": self.policy_version,
                        "min_weight_version": min_weight_version,
                        "max_weight_version": min_weight_version,
                        "stale_worker_count": int(
                            min_weight_version < self.policy_version
                        ),
                        "heartbeat_age_s": 0.0,
                        "max_version_skew": max(
                            0,
                            self.policy_version - min_weight_version,
                        ),
                    },
                    "cluster": {},
                },
            )

        if method == "POST" and path == "/start_trajectory":
            with self._lock:
                assert self.active_session_id is not None
                self.next_trajectory_index += 1
                trajectory_id = f"traj_{self.next_trajectory_index}"
                self.active_trajectory_ids.add(trajectory_id)
                self.trajectory_statuses[trajectory_id] = "active"
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "group_id": None,
                },
            )

        if method == "POST" and path == "/start_trajectory_groups":
            assert payload is not None
            counts = [int(count) for count in payload["counts"]]
            trajectory_ids: list[list[str]] = []
            with self._lock:
                assert self.active_session_id is not None
                for count in counts:
                    group: list[str] = []
                    for _ in range(count):
                        self.next_trajectory_index += 1
                        trajectory_id = f"traj_{self.next_trajectory_index}"
                        self.active_trajectory_ids.add(trajectory_id)
                        self.trajectory_statuses[trajectory_id] = "active"
                        group.append(trajectory_id)
                    trajectory_ids.append(group)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_ids": trajectory_ids,
                },
            )

        if method == "POST" and path == "/v1/chat/completions":
            assert payload is not None
            messages = [dict(message) for message in payload["messages"]]
            openforge = dict(payload["_openforge"])
            trajectory_id = str(openforge["trajectory_id"])
            purpose = str(openforge.get("purpose", "train"))
            sampling_params: dict[str, object] = {}
            if "temperature" in payload:
                sampling_params["temperature"] = payload["temperature"]
            if "top_p" in payload:
                sampling_params["top_p"] = payload["top_p"]
            if "top_k" in payload:
                sampling_params["top_k"] = payload["top_k"]
            if "repetition_penalty" in payload:
                sampling_params["repetition_penalty"] = payload["repetition_penalty"]
            if "max_completion_tokens" in payload:
                sampling_params["max_new_tokens"] = payload["max_completion_tokens"]
            with self._lock:
                self.active_trajectory_ids.add(trajectory_id)
                self.trajectory_statuses[trajectory_id] = "active"
                turn_index = sum(
                    1
                    for call in self.generate_calls
                    if call["trajectory_id"] == trajectory_id
                )
                self.generate_calls.append(
                    {
                        "trajectory_id": trajectory_id,
                        "purpose": purpose,
                        "messages": messages,
                        "sampling_params": sampling_params,
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
                        "rollout_model_version": 1,
                    },
                },
            )

        if method == "POST" and path == "/end_trajectories":
            assert payload is not None
            trajectory_ids = [
                str(trajectory_id) for trajectory_id in payload["trajectory_ids"]
            ]
            final_rewards = [float(reward) for reward in payload["final_rewards"]]
            with self._lock:
                for trajectory_id, reward in zip(
                    trajectory_ids,
                    final_rewards,
                    strict=True,
                ):
                    self.active_trajectory_ids.remove(trajectory_id)
                    self.trajectory_statuses[trajectory_id] = "completed"
                    self.finished.append((trajectory_id, reward))
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_ids": trajectory_ids,
                    "status": "completed",
                },
            )

        if method == "POST" and path == "/end_trajectory":
            assert payload is not None
            trajectory_id = str(payload["trajectory_id"])
            reward = float(payload["final_reward"])
            with self._lock:
                self.active_trajectory_ids.remove(trajectory_id)
                self.trajectory_statuses[trajectory_id] = "completed"
                self.finished.append((trajectory_id, reward))
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "status": "completed",
                },
            )

        if method == "POST" and path == "/error_trajectories":
            assert payload is not None
            trajectory_ids = [
                str(trajectory_id) for trajectory_id in payload["trajectory_ids"]
            ]
            with self._lock:
                for trajectory_id in trajectory_ids:
                    self.active_trajectory_ids.remove(trajectory_id)
                    self.trajectory_statuses[trajectory_id] = "failed"
                    self.failed.append(trajectory_id)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_ids": trajectory_ids,
                    "status": "failed",
                },
            )

        if method == "POST" and path == "/error_trajectory":
            assert payload is not None
            trajectory_id = str(payload["trajectory_id"])
            with self._lock:
                self.active_trajectory_ids.remove(trajectory_id)
                self.trajectory_statuses[trajectory_id] = "failed"
                self.failed.append(trajectory_id)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "status": "failed",
                },
            )

        if method == "POST" and path == "/trajectory_statuses":
            assert payload is not None
            trajectory_ids = [
                str(trajectory_id) for trajectory_id in payload["trajectory_ids"]
            ]
            with self._lock:
                trained_any = False
                trajectories: list[dict[str, str]] = []
                for trajectory_id in trajectory_ids:
                    status = self.trajectory_statuses[trajectory_id]
                    if status == "completed":
                        status = "trained"
                        self.trajectory_statuses[trajectory_id] = status
                        trained_any = True
                    trajectories.append(
                        {
                            "trajectory_id": trajectory_id,
                            "status": status,
                        }
                    )
                if trained_any:
                    self.policy_version += 1
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectories": trajectories,
                },
            )

        if method == "POST" and path == "/export_checkpoint":
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "policy_version": self.policy_version,
                    "checkpoint_path": "/tmp/openforge-test-checkpoint",
                },
            )

        if method == "POST" and path == "/log_validation":
            assert payload is not None
            logged_payload = dict(payload["payload"])
            self.validation_updates.append(logged_payload)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "status": "logged",
                },
            )

        raise AssertionError(f"unexpected request: {method} {path}")


@contextmanager
def _patched_ninja(
    *,
    active_session_id: str | None = "sess_1",
) -> Iterator[_FakeGateway]:
    gateway = _FakeGateway(active_session_id=active_session_id)

    class _FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            gateway.http_client_count += 1

        def get(self, path: str) -> _FakeResponse:
            return gateway.handle("GET", path)

        def post(
            self,
            path: str,
            json: dict[str, object],
            *,
            timeout: float | None = None,
        ) -> _FakeResponse:
            return gateway.handle("POST", path, json)

        def close(self) -> None:
            return None

    with ExitStack() as stack:
        stack.enter_context(patch.object(ninja.httpx, "Client", _FakeHttpClient))
        yield gateway


def test_register_requires_active_session() -> None:
    @ninja.agent(_gateway_config())
    def agent(prompt: str) -> float:
        return 1.0

    with _patched_ninja(active_session_id=None):
        with expect_raises(
            AssertionError,
            "python -m openforge.cli.main session start",
        ):
            agent(prompt="hello")


def test_register_requires_active_gateway_for_implicit_discovery() -> None:
    @ninja.agent()
    def agent(prompt: str) -> float:
        return 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        with patch.object(ninja.active_state, "active_state_path", lambda: state_path):
            with expect_raises(
                AssertionError,
                "python -m openforge.cli.main gateway start",
            ):
                agent(prompt="hello")


def test_register_rejects_async_functions() -> None:
    with expect_raises(TypeError, "does not support async functions"):

        @ninja.agent(_gateway_config())
        async def agent(prompt: str) -> float:
            _ = prompt
            return 1.0


def test_register_discovers_active_gateway_from_shared_state() -> None:
    @ninja.agent()
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

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
        with patch.object(ninja.active_state, "active_state_path", lambda: state_path):
            with _patched_ninja() as fake_gateway:
                reward = agent(prompt="hello")

    assert reward == 1.0
    assert len(fake_gateway.generate_calls) == 1


def test_register_validates_call_signature_before_connecting() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, *, prompt: str, ground_truth: str) -> float:
        _ = (client, prompt, ground_truth)
        return 1.0

    with _patched_ninja() as fake_gateway:
        with expect_raises(TypeError, "ground_truth"):
            agent(prompt="hello")

    assert fake_gateway.http_client_count == 0


def test_register_routes_explicit_messages() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        first = client.chat.completions.create(
            model="model-a",
            messages=messages,
            temperature=0.7,
            top_k=-1,
            repetition_penalty=1.1,
        )
        messages.append(
            {"role": "assistant", "content": first.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "follow up"})
        second = client.chat.completions.create(
            model="model-a",
            messages=messages,
        )
        trajectory_id = str(first.metadata["trajectory_id"])
        assert first.id == f"chatcmpl_{trajectory_id}_0"
        assert first.object == "chat.completion"
        assert first.model == "model-a"
        assert first.usage.model_dump(exclude_none=True) == {
            "completion_tokens": 2,
            "prompt_tokens": 1,
            "total_tokens": 3,
        }
        assert first.metadata == {
            "session_id": "sess_1",
            "trajectory_id": trajectory_id,
            "token_ids": [1, 2],
            "rollout_model_version": 1,
        }
        assert first.model_dump(exclude_none=True)["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "reply to hello"},
            }
        ]
        assert second.model_dump(exclude_none=True)["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "reply to follow up"},
            }
        ]
        assert second.id == f"chatcmpl_{trajectory_id}_1"
        return 1.0

    with _patched_ninja() as fake_gateway:
        reward = agent(prompt="hello")

    trajectory_id = str(fake_gateway.finished[0][0])
    assert reward == 1.0
    assert fake_gateway.generate_calls == [
        {
            "trajectory_id": trajectory_id,
            "purpose": "train",
            "messages": [{"role": "user", "content": "hello"}],
            "sampling_params": {
                "temperature": 0.7,
                "top_k": -1,
                "repetition_penalty": 1.1,
            },
        },
        {
            "trajectory_id": trajectory_id,
            "purpose": "train",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "reply to hello"},
                {"role": "user", "content": "follow up"},
            ],
            "sampling_params": {},
        },
    ]
    assert fake_gateway.finished == [(trajectory_id, 1.0)]


def test_register_marks_failed_trajectory_on_error() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        raise RuntimeError("boom")

    with _patched_ninja() as fake_gateway:
        with expect_raises(RuntimeError, "boom"):
            agent(prompt="hello")

    assert len(fake_gateway.failed) == 1
    assert fake_gateway.finished == []


def test_register_rejects_non_finite_reward_and_marks_trajectory_failed() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return float("nan")

    with _patched_ninja() as fake_gateway:
        with expect_raises(ValueError, "must be finite"):
            agent(prompt="hello")

    assert len(fake_gateway.failed) == 1
    assert fake_gateway.finished == []


def test_execute_runs_many_requests_by_default() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.sample(
            requests=[{"prompt": f"hello {index}"} for index in range(4)]
        )

    assert rewards == [1.0, 1.0, 1.0, 1.0]
    assert len(fake_gateway.finished) == 4
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 4
    assert fake_gateway.max_active_generates > 1


def test_execute_uses_group_size_for_grouped_rollouts() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.sample(
            requests=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            group_size=3,
            concurrency=6,
        )

    assert rewards == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert len(fake_gateway.finished) == 6
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 6
    assert fake_gateway.failed == []
    assert fake_gateway.max_active_generates > 1


def test_execute_accepts_num_rollouts_alias() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

    with _patched_ninja():
        rewards = agent.sample(
            requests=[{"prompt": "hello"}],
            num_rollouts=2,
            concurrency=2,
        )

    assert rewards == [[1.0, 1.0]]


def test_execute_grouped_retries_only_failed_group() -> None:
    prompt_counts: dict[str, int] = {}
    lock = Lock()

    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content
        with lock:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            prompt_call_index = prompt_counts[prompt]
        if prompt == "hello" and prompt_call_index == 2:
            raise RuntimeError("boom")
        return 1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.sample(
            requests=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            group_size=3,
            concurrency=6,
            retries=1,
        )

    assert rewards == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert len(fake_gateway.finished) == 6
    assert len(fake_gateway.failed) == 3
    assert len(fake_gateway.generate_calls) == 9
    failed_prompts = {
        call["messages"][0]["content"]
        for call in fake_gateway.generate_calls
        if call["trajectory_id"] in set(fake_gateway.failed)
    }
    assert failed_prompts == {"hello"}


def test_execute_grouped_results_can_return_failures_without_raising() -> None:
    prompt_counts: dict[str, int] = {}
    lock = Lock()

    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content
        with lock:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            prompt_call_index = prompt_counts[prompt]
        if prompt == "hello" and prompt_call_index == 1:
            raise RuntimeError("boom")
        return 1.0

    with _patched_ninja() as fake_gateway:
        with agent._session() as session:
            call_specs, _single_request = ninja._normalize_requests(
                args=(),
                kwargs={},
                requests=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            )
            grouped_results, failures = ninja._execute_grouped_results(
                agent,
                session,
                call_specs,
                group_size=2,
                concurrency=4,
                retries=0,
                raise_on_failure=False,
            )

    assert len(grouped_results) == 1
    assert grouped_results[0].request_index == 1
    assert len(failures) == 1
    assert failures[0].request_index == 0
    assert "RuntimeError: boom" in failures[0].error
    assert len(fake_gateway.failed) == 2


def test_grouped_fail_fallback_marks_started_clients_failed() -> None:
    with _patched_ninja() as fake_gateway:
        session = ninja._ActiveSession(("127.0.0.1", 8000))
        with session:
            clients = session.trajectory_groups(
                counts=[3],
                group_ids=["group_1"],
            )[0]
            session.fail_clients = lambda _clients: (_ for _ in ()).throw(  # type: ignore[method-assign]
                RuntimeError("boom")
            )
            ninja._fail_clients_best_effort(session, clients)

    assert len(fake_gateway.failed) == 3
    assert fake_gateway.active_trajectory_ids == set()


def test_active_session_configures_http_client_for_high_concurrency() -> None:
    captured: dict[str, object] = {}

    class _CapturingHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def get(self, path: str) -> _FakeResponse:
            assert path == "/current_session"
            return _FakeResponse(200, {"session_id": "sess_1"})

        def close(self) -> None:
            return None

    with patch.object(ninja.httpx, "Client", _CapturingHttpClient):
        session = ninja._ActiveSession(("127.0.0.1", 8000))
        with session:
            pass

    timeout = captured["timeout"]
    limits = captured["limits"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 10.0
    assert timeout.read == ninja._ActiveSession.REQUEST_TIMEOUT_SECONDS
    assert timeout.write == ninja._ActiveSession.REQUEST_TIMEOUT_SECONDS
    assert timeout.pool == ninja._ActiveSession.REQUEST_TIMEOUT_SECONDS
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == ninja._ActiveSession.MAX_CONNECTIONS
    assert (
        limits.max_keepalive_connections
        == ninja._ActiveSession.MAX_CONNECTIONS
    )


def test_sample_validates_requests_before_connecting() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, *, prompt: str, ground_truth: str) -> float:
        _ = (client, prompt, ground_truth)
        return 1.0

    with _patched_ninja() as fake_gateway:
        with expect_raises(TypeError, r"request\[0\].*ground_truth"):
            agent.sample(requests=[{"prompt": "hello"}])

    assert fake_gateway.http_client_count == 0


def test_register_retries_single_finish_on_read_error() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0

    with _patched_ninja() as fake_gateway:
        fake_gateway.inject_read_error("POST", "/end_trajectory")
        reward = agent(prompt="hello")

    assert reward == 1.0
    assert len(fake_gateway.finished) == 1
    assert fake_gateway.failed == []


def test_register_retries_single_fail_on_read_error() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        raise RuntimeError("boom")

    with _patched_ninja() as fake_gateway:
        fake_gateway.inject_read_error("POST", "/error_trajectory")
        with expect_raises(RuntimeError, "boom"):
            agent(prompt="hello")

    assert len(fake_gateway.failed) == 1
    assert fake_gateway.finished == []


def test_register_retries_single_finish_on_remote_protocol_error() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0

    with _patched_ninja() as fake_gateway:
        fake_gateway.inject_transport_error(
            "POST",
            "/end_trajectory",
            error_type=httpx.RemoteProtocolError,
        )
        reward = agent(prompt="hello")

    assert reward == 1.0
    assert len(fake_gateway.finished) == 1
    assert fake_gateway.failed == []


def test_agent_checkpoint_accessor() -> None:
    @ninja.agent(_gateway_config())
    def agent(prompt: str) -> float:
        _ = prompt
        return 1.0

    with _patched_ninja():
        assert agent.save() == {
            "session_id": "sess_1",
            "policy_version": 0,
            "checkpoint_path": "/tmp/openforge-test-checkpoint",
        }


def test_module_train_runs_grouped_rollouts_and_waits_for_training() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

    with _patched_ninja() as fake_gateway:
        summary = ninja.train(
            agent,
            inputs=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            group_size=3,
            concurrency=6,
        )

    assert summary == {
        "group_size": 3,
        "prompt_groups": 2,
        "samples": 6,
        "initial_policy_version": 0,
        "final_policy_version": 1,
        "max_group_reward": 1.0,
        "mean_group_reward": 1.0,
        "sample_mean_reward": 1.0,
    }
    assert len(fake_gateway.finished) == 6
    assert all(
        fake_gateway.trajectory_statuses[trajectory_id] == "trained"
        for trajectory_id, _reward in fake_gateway.finished
    )


def test_module_train_async_returns_after_rollout_completes() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        return 1.0 if response.choices[0].message.content else -1.0

    with _patched_ninja() as fake_gateway:
        summary = ninja.train_async(
            agent,
            inputs=[{"prompt": "hello"}],
            group_size=2,
            concurrency=2,
        )

    assert summary == {
        "group_size": 2,
        "prompt_groups": 1,
        "samples": 2,
        "initial_policy_version": 0,
        "final_policy_version": 0,
        "max_group_reward": 1.0,
        "mean_group_reward": 1.0,
        "sample_mean_reward": 1.0,
        "trajectory_ids": ["traj_1", "traj_2"],
    }
    assert fake_gateway.policy_version == 0
    assert all(
        fake_gateway.trajectory_statuses[trajectory_id] == "completed"
        for trajectory_id in summary["trajectory_ids"]
    )


def test_module_train_validates_requests_before_connecting() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, *, prompt: str, ground_truth: str) -> float:
        _ = (client, prompt, ground_truth)
        return 1.0

    with _patched_ninja() as fake_gateway:
        with expect_raises(TypeError, r"request\[0\].*ground_truth"):
            ninja.train(
                agent,
                inputs=[{"prompt": "hello"}],
                group_size=2,
            )

    assert fake_gateway.http_client_count == 0


def test_module_train_preflights_against_active_runtime_global_batch_size() -> None:
    @ninja.agent()
    def agent(client, prompt: str) -> float:
        _ = (client, prompt)
        return 1.0

    runtime_config = _runtime_config(global_batch_size=2)

    with _patched_ninja() as fake_gateway:
        with patch.object(
            ninja.active_state,
            "load_active_gateway_target",
            lambda: ("127.0.0.1", 8000),
        ):
            with patch.object(
                ninja.active_state,
                "load_active_runtime_config",
                lambda: runtime_config,
            ):
                with expect_raises(ValueError, "active runtime global_batch_size"):
                    ninja.train(
                        agent,
                        inputs=[{"prompt": "hello"}],
                        group_size=3,
                    )

    assert fake_gateway.http_client_count == 0


def test_module_train_skips_active_runtime_preflight_for_explicit_gateway() -> None:
    explicit_gateway_config = GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8001),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )

    @ninja.agent(explicit_gateway_config)
    def agent(client, prompt: str) -> float:
        _ = (client, prompt)
        return 1.0

    runtime_config = _runtime_config(global_batch_size=2)

    with _patched_ninja() as fake_gateway:
        with patch.object(
            ninja.active_state,
            "load_active_runtime_config",
            lambda: runtime_config,
        ):
            summary = ninja.train(
                agent,
                inputs=[{"prompt": "hello"}],
                group_size=3,
            )

    assert summary["group_size"] == 3
    assert fake_gateway.http_client_count == 1


def test_validate_runs_file_backed_requests_and_logs_validation_update() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, *, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content
        return float(len(prompt))

    with tempfile.TemporaryDirectory() as tmpdir:
        validation_dir = Path(tmpdir)
        (validation_dir / "validation.jsonl").write_text(
            json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "abcd"}) + "\n",
            encoding="utf-8",
        )
        with _patched_ninja() as fake_gateway:
            summary = ninja.validate(
                agent,
                file_path=tmpdir,
            )

    assert summary["policy_version"] == 0
    assert summary["sample_count"] == 2
    assert summary["reward_mean"] == 2.5
    assert summary["reward_min"] == 1.0
    assert summary["reward_max"] == 4.0
    assert summary["file_path"] == str(validation_dir / "validation.jsonl")
    assert fake_gateway.validation_updates == [
        {
            "policy_version": 0,
            "sample_count": 2,
            "reward_mean": 2.5,
            "reward_std": 1.5,
            "reward_min": 1.0,
            "reward_max": 4.0,
            "validation_time_s": fake_gateway.validation_updates[0]["validation_time_s"],
            "samples_per_second": fake_gateway.validation_updates[0][
                "samples_per_second"
            ],
        }
    ]
    assert all(call["purpose"] == "validation" for call in fake_gateway.generate_calls)


def test_validate_waits_for_rollout_to_load_latest_policy_version() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, *, prompt: str) -> float:
        response = client.chat.completions.create(
            model="model-a",
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content
        return 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        validation_dir = Path(tmpdir)
        (validation_dir / "validation.jsonl").write_text(
            json.dumps({"prompt": "check"}) + "\n",
            encoding="utf-8",
        )
        with _patched_ninja() as fake_gateway:
            fake_gateway.policy_version = 2
            fake_gateway.rollout_min_weight_versions = [0, 1, 2]
            with patch.object(
                ninja._ActiveSession,
                "STATUS_POLL_INTERVAL_SECONDS",
                0.001,
            ):
                summary = ninja.validate(
                    agent,
                    file_path=tmpdir,
                    wait_timeout=1.0,
                )

    assert summary["policy_version"] == 2
    assert fake_gateway.status_calls == 3
    assert all(call["purpose"] == "validation" for call in fake_gateway.generate_calls)


def main() -> int:
    return run_tests(
        [
            test_register_requires_active_session,
            test_register_requires_active_gateway_for_implicit_discovery,
            test_register_rejects_async_functions,
            test_register_discovers_active_gateway_from_shared_state,
            test_register_validates_call_signature_before_connecting,
            test_register_routes_explicit_messages,
            test_register_marks_failed_trajectory_on_error,
            test_register_rejects_non_finite_reward_and_marks_trajectory_failed,
            test_execute_runs_many_requests_by_default,
            test_execute_uses_group_size_for_grouped_rollouts,
            test_execute_accepts_num_rollouts_alias,
            test_execute_grouped_retries_only_failed_group,
            test_grouped_fail_fallback_marks_started_clients_failed,
            test_sample_validates_requests_before_connecting,
            test_register_retries_single_finish_on_read_error,
            test_register_retries_single_fail_on_read_error,
            test_agent_checkpoint_accessor,
            test_module_train_runs_grouped_rollouts_and_waits_for_training,
            test_module_train_async_returns_after_rollout_completes,
            test_module_train_validates_requests_before_connecting,
            test_module_train_preflights_against_active_runtime_global_batch_size,
            test_module_train_skips_active_runtime_preflight_for_explicit_gateway,
            test_validate_runs_file_backed_requests_and_logs_validation_update,
            test_validate_waits_for_rollout_to_load_latest_policy_version,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
