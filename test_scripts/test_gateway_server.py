# Copyright 2026 openforge

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from openforge.configs.models import GatewayConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.server import (
    GatewayGeneration,
    ModelBusyError,
    UnsupportedModelError,
    create_app,
)


class _FakeController:
    def __init__(self, supported_models: tuple[str, ...] = ("model-a",)) -> None:
        self._supported_models = supported_models
        self._current_model: str | None = None
        self.last_sampling_params: dict[str, object] | None = None
        self.shutdown_count = 0

    def list_models(self) -> list[dict[str, str]]:
        return [
            {"id": model_name, "tokenizer": f"{model_name}-tokenizer"}
            for model_name in self._supported_models
        ]

    def current_model(self) -> str | None:
        return self._current_model

    def ensure_model(self, model_name: str) -> None:
        if model_name not in self._supported_models:
            raise UnsupportedModelError(model_name)
        if self._current_model is None:
            self._current_model = model_name
            return
        if self._current_model != model_name:
            raise ModelBusyError(model_name)

    def tokenize_messages(self, model_name: str, messages: list[dict[str, str]]) -> list[int]:
        self.ensure_model(model_name)
        token_count = sum(len(message["content"].split()) for message in messages)
        return list(range(1, token_count + 2))

    def generate(
        self,
        model_name: str,
        *,
        prompt_token_ids: list[int],
        sampling_params: dict[str, object] | None = None,
    ) -> GatewayGeneration:
        self.ensure_model(model_name)
        self.last_sampling_params = sampling_params
        prompt_tail = int(prompt_token_ids[-1]) if prompt_token_ids else 0
        return GatewayGeneration(
            token_ids=[10 + prompt_tail, 20 + prompt_tail],
            logprobs=[-0.1, -0.2],
            rollout_model_version=4,
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self._current_model = None


def test_gateway_http_flow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        controller = _FakeController()
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=controller,
        )
        with TestClient(app) as client:
            models = client.get("/models")
            assert models.status_code == 200
            assert models.json() == {
                "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
                "active_model": None,
            }

            started = client.post("/start_session", json={"model": "model-a"})
            assert started.status_code == 200
            session_id = started.json()["session_id"]

            trajectory = client.post("/start_trajectory", json={"session_id": session_id})
            assert trajectory.status_code == 200
            trajectory_id = trajectory.json()["trajectory_id"]

            generated = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "messages": [{"role": "user", "content": "hello"}],
                    "sampling_params": {"temperature": 0.5, "top_p": 0.9},
                },
            )
            assert generated.status_code == 200
            assert generated.json() == {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "token_ids": [12, 22],
                "logprobs": [-0.1, -0.2],
                "finish_reason": "stop",
                "rollout_model_version": 4,
            }
            assert controller.last_sampling_params == {"temperature": 0.5, "top_p": 0.9}

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

            ended_session = client.post("/end_session", json={"session_id": session_id})
            assert ended_session.status_code == 200
            assert ended_session.json()["status"] == "completed"


def test_gateway_http_health_and_models_reflect_loaded_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        with TestClient(app) as client:
            assert client.get("/health").json() == {"ok": True}
            assert client.get("/models").json()["active_model"] is None
            assert client.get("/models").json()["models"] == [
                {"id": "model-a", "tokenizer": "model-a-tokenizer"}
            ]

            started = client.post("/start_session", json={"model": "model-a"})
            assert started.status_code == 200
            assert client.get("/models").json()["active_model"] == "model-a"


def test_gateway_http_start_session_reports_unsupported_and_busy_models() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(("model-a", "model-b")),
        )
        with TestClient(app) as client:
            unsupported = client.post("/start_session", json={"model": "model-c"})
            assert unsupported.status_code == 404

            created = client.post("/start_session", json={"model": "model-a"})
            assert created.status_code == 200

            busy = client.post("/start_session", json={"model": "model-b"})
            assert busy.status_code == 409


def test_gateway_http_releases_model_after_last_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        controller = _FakeController(("model-a", "model-b"))
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=controller,
        )
        with TestClient(app) as client:
            session_id = client.post("/start_session", json={"model": "model-a"}).json()["session_id"]
            trajectory_id = client.post(
                "/start_trajectory",
                json={"session_id": session_id},
            ).json()["trajectory_id"]
            generated = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "messages": [{"role": "user", "content": "hello"}],
                },
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
            ended_session = client.post("/end_session", json={"session_id": session_id})

            assert ended_session.status_code == 200
            assert controller.shutdown_count == 1
            assert client.get("/models").json()["active_model"] is None

            switched = client.post("/start_session", json={"model": "model-b"})
            assert switched.status_code == 200
            assert switched.json()["model"] == "model-b"


def test_gateway_http_generate_validates_request_and_unknown_records() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        with TestClient(app) as client:
            invalid = client.post(
                "/generate",
                json={
                    "session_id": "missing",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert invalid.status_code == 422

            missing_session = client.post(
                "/generate",
                json={
                    "session_id": "missing",
                    "trajectory_id": "traj_missing",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert missing_session.status_code == 404

            session_id = client.post("/start_session", json={"model": "model-a"}).json()["session_id"]
            missing_trajectory = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": "traj_missing",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert missing_trajectory.status_code == 404


def test_gateway_http_generate_after_end_trajectory_returns_conflict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        with TestClient(app) as client:
            session_id = client.post("/start_session", json={"model": "model-a"}).json()["session_id"]
            trajectory_id = client.post(
                "/start_trajectory",
                json={"session_id": session_id},
            ).json()["trajectory_id"]
            generated = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "messages": [{"role": "user", "content": "hello"}],
                },
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
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "messages": [{"role": "user", "content": "hello again"}],
                },
            )
            assert again.status_code == 409


def test_gateway_http_end_session_requires_completed_trajectories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        with TestClient(app) as client:
            missing = client.post("/end_session", json={"session_id": "missing"})
            assert missing.status_code == 404

            session_id = client.post("/start_session", json={"model": "model-a"}).json()["session_id"]
            trajectory_id = client.post(
                "/start_trajectory",
                json={"session_id": session_id},
            ).json()["trajectory_id"]
            incomplete = client.post("/end_session", json={"session_id": session_id})
            assert incomplete.status_code == 409

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


def test_gateway_http_start_trajectory_can_fork_from_parent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        with TestClient(app) as client:
            session_id = client.post("/start_session", json={"model": "model-a"}).json()["session_id"]
            root = client.post("/start_trajectory", json={"session_id": session_id})
            assert root.status_code == 200
            root_id = root.json()["trajectory_id"]

            generated = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": root_id,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert generated.status_code == 200

            child = client.post(
                "/start_trajectory",
                json={
                    "session_id": session_id,
                    "parent_trajectory_id": root_id,
                },
            )
            assert child.status_code == 200
            assert child.json()["parent_trajectory_id"] == root_id


def test_create_app_requires_dependencies_with_gateway_config() -> None:
    try:
        create_app(GatewayConfig(host="127.0.0.1", port=0))
    except ValueError as exc:
        assert "store and controller must be provided" in str(exc)
    else:
        raise AssertionError("expected create_app to reject missing dependencies")
