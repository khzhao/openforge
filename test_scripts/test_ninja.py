# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from threading import Lock
from typing import Iterator
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import openforge.ninja as ninja
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.ninja import register


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
                    "parent_trajectory_id": None,
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

        if method == "POST" and path == "/generate":
            assert payload is not None
            messages = [dict(message) for message in payload["messages"]]
            trajectory_id = str(payload["trajectory_id"])
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

    with ExitStack() as stack:
        stack.enter_context(patch.object(ninja.httpx, "Client", _FakeHttpClient))
        yield gateway


def test_register_requires_active_session() -> None:
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        return 1.0

    with _patched_ninja(active_session_id=None):
        with expect_raises(AssertionError, "no active session"):
            agent(prompt="hello")


def test_register_routes_explicit_messages() -> None:
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        first = client.generate(messages, sampling_params={"temperature": 0.7})
        messages.append(first["choices"][0]["message"])
        messages.append({"role": "user", "content": "follow up"})
        second = client.generate(messages)
        trajectory_id = str(first["metadata"]["trajectory_id"])
        assert first["id"] == f"chatcmpl_{trajectory_id}_0"
        assert first["object"] == "chat.completion"
        assert first["model"] == "model-a"
        assert first["usage"] == {
            "completion_tokens": 2,
            "prompt_tokens": 1,
            "total_tokens": 3,
        }
        assert first["metadata"] == {
            "session_id": "sess_1",
            "trajectory_id": trajectory_id,
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
        assert second["id"] == f"chatcmpl_{trajectory_id}_1"
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
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        client.generate([{"role": "user", "content": prompt}])
        raise RuntimeError("boom")

    with _patched_ninja() as fake_gateway:
        with expect_raises(RuntimeError, "boom"):
            agent(prompt="hello")

    assert len(fake_gateway.failed) == 1
    assert fake_gateway.finished == []


def test_execute_runs_many_requests_by_default() -> None:
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.generate([{"role": "user", "content": prompt}])
        message = response["choices"][0]["message"]
        return 1.0 if message["content"] else -1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.execute(requests=[{"prompt": f"hello {index}"} for index in range(4)])

    assert rewards == [1.0, 1.0, 1.0, 1.0]
    assert len(fake_gateway.finished) == 4
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 4
    assert fake_gateway.max_active_generates > 1


def test_execute_uses_group_size_for_grouped_rollouts() -> None:
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.generate([{"role": "user", "content": prompt}])
        message = response["choices"][0]["message"]
        return 1.0 if message["content"] else -1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.execute(
            requests=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            group_size=3,
            max_parallelism=6,
        )

    assert rewards == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert len(fake_gateway.finished) == 6
    assert len({trajectory_id for trajectory_id, _ in fake_gateway.finished}) == 6
    assert fake_gateway.failed == []
    assert fake_gateway.max_active_generates > 1


def test_execute_grouped_retries_only_failed_group() -> None:
    prompt_counts: dict[str, int] = {}
    lock = Lock()

    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        response = client.generate([{"role": "user", "content": prompt}])
        assert response["choices"][0]["message"]["content"]
        with lock:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            prompt_call_index = prompt_counts[prompt]
        if prompt == "hello" and prompt_call_index == 2:
            raise RuntimeError("boom")
        return 1.0

    with _patched_ninja() as fake_gateway:
        rewards = agent.execute(
            requests=[{"prompt": "hello"}, {"prompt": "goodbye"}],
            group_size=3,
            max_parallelism=6,
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
    @register(_gateway_config())
    def agent(client, prompt: str) -> float:
        _ = prompt
        return 1.0

    with _patched_ninja():
        assert agent.current_policy_version() == 0
        assert agent.export_checkpoint() == {
            "session_id": "sess_1",
            "policy_version": 0,
            "checkpoint_path": "/tmp/openforge-test-checkpoint",
        }


def main() -> int:
    return run_tests(
        [
            test_register_requires_active_session,
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
