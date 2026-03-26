# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import json
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from _script_test_utils import install_test_stubs, run_tests
from fastapi.testclient import TestClient

install_test_stubs()

import openforge.gateway.server as gateway_server
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import Generation
from openforge.gateway.types import StartSessionRequest


class _FakeTrainLoop:
    instances: list["_FakeTrainLoop"] = []

    def __init__(self, *, session_id: str, store, train_manager) -> None:
        self.session_id = session_id
        self.store = store
        self.train_manager = train_manager
        self.started = False
        self.stopped = False
        self.train_once_calls = 0
        self.policy_version = 0
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def train_once(self) -> bool:
        self.train_once_calls += 1
        return False


class _FakeTrainRuntime:
    def __init__(self) -> None:
        _FakeTrainLoop.instances.clear()
        self._train_loop: _FakeTrainLoop | None = None
        self.policy_version = 0

    def start_session(self, *, session_id: str, store) -> None:
        assert self._train_loop is None
        train_loop = _FakeTrainLoop(
            session_id=session_id,
            store=store,
            train_manager=object(),
        )
        train_loop.start()
        self._train_loop = train_loop

    async def end_session(self) -> None:
        train_loop = self._train_loop
        if train_loop is None:
            return
        await train_loop.stop()
        self._train_loop = None
        self.policy_version = 0

    def export_checkpoint(self) -> tuple[int, str]:
        train_loop = self._train_loop
        assert train_loop is not None
        return train_loop.policy_version, f"/tmp/checkpoint-{train_loop.policy_version}"

    async def shutdown(self) -> None:
        await self.end_session()


def _start_session_payload(model_name: str = "model-a") -> dict[str, object]:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
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
    )
    return request.model_dump(mode="json")


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=0),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


class _FakeRuntime:
    def __init__(self, supported_models: tuple[str, ...] = ("model-a",)) -> None:
        self._supported_models = supported_models
        self._current_model: str | None = None
        self._train = _FakeTrainRuntime()
        self.last_sampling_params: dict[str, object] | None = None
        self.last_trajectory_ids: list[str] | None = None
        self.released_trajectory_ids: list[list[str]] = []
        self.shutdown_count = 0

    def list_models(self) -> list[str]:
        return list(self._supported_models)

    def current_model(self) -> str | None:
        return self._current_model

    def start(self, *, runtime_config) -> str:
        model_name = str(runtime_config.model.model_name_or_path)
        if model_name not in self._supported_models:
            raise Exception(f"unsupported model: {model_name}")
        if self._current_model is None:
            self._current_model = model_name
            return model_name
        if self._current_model != model_name:
            raise Exception(
                f"gateway already has active model {self._current_model!r}; "
                f"cannot switch to {model_name!r}"
            )
        return model_name

    def tokenize_messages(
        self,
        messages: list[Any],
        *,
        tools: list[Any] | None = None,
    ) -> list[int]:
        token_count = sum(len(message.content.split()) for message in messages)
        return list(range(1, token_count + 2))

    def tokenize_messages_batch(
        self,
        message_batches: list[list[Any]],
        *,
        tools: list[Any] | None = None,
    ) -> list[list[int]]:
        return [self.tokenize_messages(messages) for messages in message_batches]

    def generate(
        self,
        *,
        input_ids: list[int],
        sampling_params: dict[str, object] | None = None,
    ) -> Generation:
        self.last_sampling_params = sampling_params
        prompt_tail = int(input_ids[-1]) if input_ids else 0
        return Generation(
            text=f"reply-{prompt_tail}",
            token_ids=[10 + prompt_tail, 20 + prompt_tail],
            rollout_model_version=4,
            rollout_log_probs=[-0.1, -0.2],
        )

    def generate_batch(
        self,
        *,
        trajectory_ids: list[str] | None = None,
        input_ids: list[list[int]],
        sampling_params: dict[str, object] | None = None,
    ) -> list[Generation]:
        self.last_trajectory_ids = trajectory_ids
        return [
            self.generate(input_ids=item, sampling_params=sampling_params)
            for item in input_ids
        ]

    def train(self) -> _FakeTrainRuntime:
        return self._train

    def release_trajectories(self, trajectory_ids: list[str]) -> None:
        self.released_trajectory_ids.append(list(trajectory_ids))

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        self._current_model = None
        await self._train.shutdown()


class _FailingTokenizeRuntime(_FakeRuntime):
    def tokenize_messages(
        self,
        messages: list[Any],
        *,
        tools: list[Any] | None = None,
    ) -> list[int]:
        raise RuntimeError("template boom")

    def tokenize_messages_batch(
        self,
        message_batches: list[list[Any]],
        *,
        tools: list[Any] | None = None,
    ) -> list[list[int]]:
        raise RuntimeError("template boom")


def _chat_payload(
    *,
    session_id: str,
    trajectory_id: str,
    content: str,
    model: str = "model-a",
    temperature: float | None = None,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "_openforge": {
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "group_id": None,
        },
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    return payload


@contextmanager
def _patched_app(
    *,
    store: SQLiteOpenForgeStore,
    runtime: object,
) -> Iterator[object]:
    with ExitStack() as stack:
        state_tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
        state_path = Path(state_tmpdir) / "active_gateway.json"
        stack.enter_context(
            patch.object(gateway_server, "_build_store", lambda cfg: store)
        )
        stack.enter_context(
            patch.object(gateway_server, "Runtime", lambda cfg: runtime)
        )
        stack.enter_context(
            patch.object(
                gateway_server.active_state, "active_state_path", lambda: state_path
            )
        )
        yield gateway_server.create_app(_server_config())


def test_gateway_http_flow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime()
        with _patched_app(store=store, runtime=runtime) as app:
            with TestClient(app) as client:
                models = client.get("/v1/models")
                assert models.status_code == 200
                assert models.json() == {
                    "object": "list",
                    "data": [
                        {
                            "id": "model-a",
                            "object": "model",
                            "created": 0,
                            "owned_by": "openforge",
                        }
                    ],
                }

                started = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                )
                assert started.status_code == 200
                session_id = started.json()["session_id"]

                trajectory = client.post(
                    "/start_trajectory", json={"session_id": session_id}
                )
                assert trajectory.status_code == 200
                trajectory_id = trajectory.json()["trajectory_id"]

                generated = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id=trajectory_id,
                        content="hello",
                        temperature=0.5,
                        top_p=0.9,
                    ),
                )
                assert generated.status_code == 200
                payload = generated.json()
                assert payload["id"] == f"chatcmpl_{trajectory_id}_0"
                assert payload["object"] == "chat.completion"
                assert payload["model"] == "model-a"
                assert payload["choices"] == [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "reply-2",
                            "tool_calls": None,
                        },
                        "logprobs": None,
                    }
                ]
                assert payload["usage"] == {
                    "completion_tokens": 2,
                    "prompt_tokens": 2,
                    "total_tokens": 4,
                }
                assert payload["metadata"] == {
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "token_ids": [12, 22],
                    "rollout_model_version": 4,
                }
                assert runtime.last_sampling_params == {
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
                assert runtime.last_trajectory_ids == [trajectory_id]

                ended_trajectory = client.post(
                    "/end_trajectory",
                    json={
                        "session_id": session_id,
                        "trajectory_id": trajectory_id,
                        "final_reward": 1.0,
                    },
                )
                assert ended_trajectory.status_code == 200
                assert ended_trajectory.json()["status"] == "completed"

                ended_session = client.post(
                    "/end_session", json={"session_id": session_id}
                )
                assert ended_session.status_code == 200
                assert ended_session.json()["status"] == "completed"


def test_gateway_http_health_and_models_reflect_loaded_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FakeRuntime()) as app:
            with TestClient(app) as client:
                assert client.get("/health").json() == {"ok": True}
                assert client.get("/info").json() == {
                    "data": {"path": None},
                    "gateway": {"host": "127.0.0.1", "port": 0},
                    "cluster": {
                        "num_nodes": 1,
                        "gpus_per_node": 1,
                        "cpus_per_node": 1,
                    },
                }
                assert client.get("/current_session").status_code == 404
                assert client.get("/v1/models").json()["data"] == [
                    {
                        "id": "model-a",
                        "object": "model",
                        "created": 0,
                        "owned_by": "openforge",
                    }
                ]

                started = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                )
                assert started.status_code == 200
                assert client.get("/current_session").json() == started.json()


def test_gateway_http_updates_shared_state_for_gateway_and_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(gateway_server, "_build_store", lambda cfg: store)
            )
            stack.enter_context(
                patch.object(gateway_server, "Runtime", lambda cfg: runtime)
            )
            stack.enter_context(
                patch.object(
                    gateway_server.active_state,
                    "active_state_path",
                    lambda: state_path,
                )
            )
            app = gateway_server.create_app(_server_config())
            with TestClient(app) as client:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                assert state["gateway"] == {
                    "host": "127.0.0.1",
                    "pid": state["gateway"]["pid"],
                    "port": 0,
                }
                assert isinstance(state["gateway"]["pid"], int)
                assert state["session"] is None
                assert state["version"] == 1

                started = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                )
                assert started.status_code == 200
                session_state = json.loads(state_path.read_text(encoding="utf-8"))
                assert session_state["session"] == {
                    "session_id": started.json()["session_id"],
                    "runtime": _start_session_payload("model-a")["runtime"],
                }

                ended = client.post(
                    "/end_session",
                    json={"session_id": started.json()["session_id"]},
                )
                assert ended.status_code == 200
                ended_state = json.loads(state_path.read_text(encoding="utf-8"))
                assert ended_state["session"] is None

        assert state_path.exists() is False


def test_gateway_http_start_session_reports_errors() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime(("model-a", "model-b"))
        with _patched_app(store=store, runtime=runtime) as app:
            with TestClient(app) as client:
                unsupported = client.post(
                    "/start_session", json=_start_session_payload("model-c")
                )
                assert unsupported.status_code == 400

                created = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                )
                assert created.status_code == 200
                reused = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                )
                assert reused.status_code == 400

                busy = client.post(
                    "/start_session", json=_start_session_payload("model-b")
                )
                assert busy.status_code == 400


def test_gateway_http_releases_model_after_last_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime(("model-a", "model-b"))
        with _patched_app(store=store, runtime=runtime) as app:
            with TestClient(app) as client:
                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                trajectory_id = client.post(
                    "/start_trajectory",
                    json={"session_id": session_id},
                ).json()["trajectory_id"]
                generated = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id=trajectory_id,
                        content="hello",
                    ),
                )
                assert generated.status_code == 200
                ended_trajectory = client.post(
                    "/end_trajectory",
                    json={
                        "session_id": session_id,
                        "trajectory_id": trajectory_id,
                        "final_reward": 1.0,
                    },
                )
                assert ended_trajectory.status_code == 200
                ended_session = client.post(
                    "/end_session", json={"session_id": session_id}
                )

                assert ended_session.status_code == 200
                assert runtime.shutdown_count == 1
                assert client.get("/current_session").status_code == 404

                switched = client.post(
                    "/start_session", json=_start_session_payload("model-b")
                )
                assert switched.status_code == 200
                assert switched.json()["model"] == "model-b"


def test_gateway_http_generate_validates_request_and_unknown_records() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FakeRuntime()) as app:
            with TestClient(app) as client:
                invalid = client.post(
                    "/v1/chat/completions",
                    json={"model": "model-a", "messages": [{"role": "user", "content": "hello"}]},
                )
                assert invalid.status_code == 422

                missing_session = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id="missing",
                        trajectory_id="traj_missing",
                        content="hello",
                    ),
                )
                assert missing_session.status_code == 400

                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                missing_trajectory = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id="traj_missing",
                        content="hello",
                    ),
                )
                assert missing_trajectory.status_code == 200
                assert (
                    missing_trajectory.json()["metadata"]["trajectory_id"]
                    == "traj_missing"
                )


def test_gateway_http_generate_returns_bad_request_when_tokenization_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FailingTokenizeRuntime()) as app:
            with TestClient(app) as client:
                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                trajectory_id = client.post(
                    "/start_trajectory",
                    json={"session_id": session_id},
                ).json()["trajectory_id"]

                generated = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id=trajectory_id,
                        content="hello",
                    ),
                )
                assert generated.status_code == 400
                assert generated.json() == {"detail": "template boom"}


def test_gateway_http_generate_after_end_trajectory_returns_conflict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FakeRuntime()) as app:
            with TestClient(app) as client:
                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                trajectory_id = client.post(
                    "/start_trajectory",
                    json={"session_id": session_id},
                ).json()["trajectory_id"]
                generated = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id=trajectory_id,
                        content="hello",
                    ),
                )
                assert generated.status_code == 200
                ended_trajectory = client.post(
                    "/end_trajectory",
                    json={
                        "session_id": session_id,
                        "trajectory_id": trajectory_id,
                        "final_reward": 1.0,
                    },
                )
                assert ended_trajectory.status_code == 200

                again = client.post(
                    "/v1/chat/completions",
                    json=_chat_payload(
                        session_id=session_id,
                        trajectory_id=trajectory_id,
                        content="hello again",
                    ),
                )
                assert again.status_code == 400
                assert again.json() == {
                    "detail": f"trajectory {trajectory_id} is not active"
                }


def test_gateway_http_end_session_requires_completed_trajectories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FakeRuntime()) as app:
            with TestClient(app) as client:
                missing = client.post("/end_session", json={"session_id": "missing"})
                assert missing.status_code == 400

                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                trajectory_id = client.post(
                    "/start_trajectory",
                    json={"session_id": session_id},
                ).json()["trajectory_id"]
                incomplete = client.post(
                    "/end_session", json={"session_id": session_id}
                )
                assert incomplete.status_code == 400

                completed = client.post(
                    "/end_trajectory",
                    json={
                        "session_id": session_id,
                        "trajectory_id": trajectory_id,
                        "final_reward": 1.0,
                    },
                )
                assert completed.status_code == 200
                ended = client.post("/end_session", json={"session_id": session_id})
                assert ended.status_code == 200


def test_gateway_http_start_trajectory_accepts_group_id() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        with _patched_app(store=store, runtime=_FakeRuntime()) as app:
            with TestClient(app) as client:
                session_id = client.post(
                    "/start_session", json=_start_session_payload("model-a")
                ).json()["session_id"]
                trajectory = client.post(
                    "/start_trajectory",
                    json={
                        "session_id": session_id,
                        "group_id": "g0",
                    },
                )
                assert trajectory.status_code == 200
                assert trajectory.json()["group_id"] == "g0"


def test_create_app_accepts_server_config_without_injected_dependencies() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime = _FakeRuntime()
        cfg = GatewayServerConfig(
            data=DataConfig(path=str(Path(tmpdir) / "gateway.sqlite3")),
            gateway=GatewayConfig(host="127.0.0.1", port=0),
            cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
        )
        with patch.object(gateway_server, "Runtime", lambda cfg: runtime):
            app = gateway_server.create_app(cfg)
        assert app.title == "OpenForge Gateway"


def main() -> int:
    return run_tests(
        [
            test_gateway_http_flow,
            test_gateway_http_health_and_models_reflect_loaded_model,
            test_gateway_http_updates_shared_state_for_gateway_and_session,
            test_gateway_http_start_session_reports_errors,
            test_gateway_http_releases_model_after_last_session,
            test_gateway_http_generate_validates_request_and_unknown_records,
            test_gateway_http_generate_returns_bad_request_when_tokenization_fails,
            test_gateway_http_generate_after_end_trajectory_returns_conflict,
            test_gateway_http_end_session_requires_completed_trajectories,
            test_gateway_http_start_trajectory_accepts_group_id,
            test_create_app_accepts_server_config_without_injected_dependencies,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
