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

    def list_models(self) -> list[str]:
        return list(self._supported_models)

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
        return [1, 2, 3]

    def generate(
        self,
        model_name: str,
        *,
        prompt_token_ids: list[int],
        n: int,
        sampling_params: dict[str, object] | None = None,
    ) -> list[GatewayGeneration]:
        self.ensure_model(model_name)
        self.last_sampling_params = sampling_params
        return [
            GatewayGeneration(
                token_ids=[10 + choice_index, 20 + choice_index],
                logprobs=[-0.1, -0.2],
                rollout_model_version=4,
            )
            for choice_index in range(n)
        ]


def test_gateway_http_flow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        controller = _FakeController()
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=controller,
        )
        client = TestClient(app)

        models = client.get("/models")
        assert models.status_code == 200
        assert models.json() == {"models": ["model-a"], "current_model": None}

        created = client.post("/create_session", json={"model": "model-a"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        generated = client.post(
            "/generate",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
                "sampling_params": {"temperature": 0.5, "top_p": 0.9},
            },
        )
        assert generated.status_code == 200
        choices = generated.json()["choices"]
        assert len(choices) == 2
        assert controller.last_sampling_params == {"temperature": 0.5, "top_p": 0.9}

        ended = client.post(
            "/end_session",
            json={
                "session_id": session_id,
                "rewards": [
                    {"rollout_id": choices[0]["rollout_id"], "reward": 1.0},
                    {"rollout_id": choices[1]["rollout_id"], "reward": 0.0},
                ],
            },
        )
        assert ended.status_code == 200
        assert ended.json()["status"] == "completed"


def test_gateway_http_health_and_models_reflect_loaded_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        client = TestClient(app)

        assert client.get("/health").json() == {"ok": True}
        assert client.get("/models").json()["current_model"] is None

        created = client.post("/create_session", json={"model": "model-a"})
        assert created.status_code == 200
        assert client.get("/models").json()["current_model"] == "model-a"


def test_gateway_http_create_session_reports_unsupported_and_busy_models() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(("model-a", "model-b")),
        )
        client = TestClient(app)

        unsupported = client.post("/create_session", json={"model": "model-c"})
        assert unsupported.status_code == 404

        created = client.post("/create_session", json={"model": "model-a"})
        assert created.status_code == 200

        busy = client.post("/create_session", json={"model": "model-b"})
        assert busy.status_code == 409


def test_gateway_http_generate_validates_request_and_unknown_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        client = TestClient(app)

        invalid = client.post(
            "/generate",
            json={
                "session_id": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 0,
            },
        )
        assert invalid.status_code == 422

        missing = client.post(
            "/generate",
            json={
                "session_id": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 1,
            },
        )
        assert missing.status_code == 404


def test_gateway_http_generate_after_end_session_returns_conflict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        client = TestClient(app)

        session_id = client.post("/create_session", json={"model": "model-a"}).json()["session_id"]
        generated = client.post(
            "/generate",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "hello"}],
                "n": 1,
            },
        )
        rollout_id = generated.json()["choices"][0]["rollout_id"]
        ended = client.post(
            "/end_session",
            json={
                "session_id": session_id,
                "rewards": [{"rollout_id": rollout_id, "reward": 1.0}],
            },
        )
        assert ended.status_code == 200

        again = client.post(
            "/generate",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "hello again"}],
                "n": 1,
            },
        )
        assert again.status_code == 409


def test_gateway_http_end_session_reports_missing_or_incomplete_rewards() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = create_app(
            GatewayConfig(host="127.0.0.1", port=0),
            store=store,
            controller=_FakeController(),
        )
        client = TestClient(app)

        missing = client.post(
            "/end_session",
            json={"session_id": "missing", "rewards": []},
        )
        assert missing.status_code == 404

        session_id = client.post("/create_session", json={"model": "model-a"}).json()["session_id"]
        generated = client.post(
            "/generate",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            },
        )
        choices = generated.json()["choices"]

        incomplete = client.post(
            "/end_session",
            json={
                "session_id": session_id,
                "rewards": [{"rollout_id": choices[0]["rollout_id"], "reward": 1.0}],
            },
        )
        assert incomplete.status_code == 400


def test_create_app_requires_dependencies_with_gateway_config() -> None:
    try:
        create_app(GatewayConfig(host="127.0.0.1", port=0))
    except ValueError as exc:
        assert "store and controller must be provided" in str(exc)
    else:
        raise AssertionError("expected create_app to reject missing dependencies")
