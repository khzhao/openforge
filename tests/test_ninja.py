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

from openai.types.chat import ChatCompletion

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import openforge.ninja as ninja
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig


def _gateway_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
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
    def __init__(self, *, active_session_id: str | None = "sess_1") -> None:
        self._lock = Lock()
        self.active_session_id = active_session_id
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
        if method == "GET" and path == "/current_session":
            if self.active_session_id is None:
                return _FakeResponse(404, {"detail": "no active session"})
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "model": "model-a",
                    "policy_version": 0,
                },
            )

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
            sampling_params: dict[str, object] = {}
            if "temperature" in payload:
                sampling_params["temperature"] = payload["temperature"]
            if "top_p" in payload:
                sampling_params["top_p"] = payload["top_p"]
            if "max_completion_tokens" in payload:
                sampling_params["max_new_tokens"] = payload["max_completion_tokens"]
            with self._lock:
                self.active_trajectory_ids.add(trajectory_id)
                turn_index = sum(
                    1
                    for call in self.generate_calls
                    if call["trajectory_id"] == trajectory_id
                )
                self.generate_calls.append(
                    {
                        "trajectory_id": trajectory_id,
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
                        "rollout_model_version": "v1",
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
                self.failed.append(trajectory_id)
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "trajectory_id": trajectory_id,
                    "status": "failed",
                },
            )

        if method == "POST" and path == "/export_checkpoint":
            return _FakeResponse(
                200,
                {
                    "session_id": self.active_session_id,
                    "policy_version": 0,
                    "checkpoint_path": "/tmp/openforge-test-checkpoint",
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

        def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
            return gateway.handle("POST", path, json)

        def close(self) -> None:
            return None

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.chat = self._Chat()
            self.models = object()

        class _Chat:
            def __init__(self) -> None:
                self.completions = _FakeOpenAI._Completions()

        class _Completions:
            def create(self, **kwargs):
                extra_body = dict(kwargs.pop("extra_body", {}))
                payload = dict(kwargs)
                payload.update(extra_body)
                response = gateway.handle(
                    "POST",
                    "/v1/chat/completions",
                    payload,
                )
                response.raise_for_status()
                return ChatCompletion.model_validate(response.json())

        def close(self) -> None:
            return None

    with ExitStack() as stack:
        stack.enter_context(patch.object(ninja.httpx, "Client", _FakeHttpClient))
        stack.enter_context(patch.object(ninja, "OpenAI", _FakeOpenAI))
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


def test_register_routes_explicit_messages() -> None:
    @ninja.agent(_gateway_config())
    def agent(client, prompt: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        first = client.chat.completions.create(
            model="model-a",
            messages=messages,
            temperature=0.7,
        )
        messages.append({"role": "assistant", "content": first.choices[0].message.content})
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
            "rollout_model_version": "v1",
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
            "messages": [{"role": "user", "content": "hello"}],
            "sampling_params": {"temperature": 0.7},
        },
        {
            "trajectory_id": trajectory_id,
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
            num_rollouts=3,
            concurrency=6,
        )

    assert rewards == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert len(fake_gateway.finished) == 6
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 6
    assert fake_gateway.failed == []
    assert fake_gateway.max_active_generates > 1


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
            num_rollouts=3,
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


def test_agent_checkpoint_and_policy_accessors() -> None:
    @ninja.agent(_gateway_config())
    def agent(prompt: str) -> float:
        _ = prompt
        return 1.0

    with _patched_ninja():
        assert agent.policy_version() == 0
        assert agent.save() == {
            "session_id": "sess_1",
            "policy_version": 0,
            "checkpoint_path": "/tmp/openforge-test-checkpoint",
        }


def main() -> int:
    return run_tests(
        [
            test_register_requires_active_session,
            test_register_requires_active_gateway_for_implicit_discovery,
            test_register_discovers_active_gateway_from_shared_state,
            test_register_routes_explicit_messages,
            test_register_marks_failed_trajectory_on_error,
            test_execute_runs_many_requests_by_default,
            test_execute_uses_group_size_for_grouped_rollouts,
            test_execute_grouped_retries_only_failed_group,
            test_agent_checkpoint_and_policy_accessors,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
