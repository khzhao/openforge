# Copyright 2026 openforge

from __future__ import annotations

import tempfile
from pathlib import Path

import openforge.gateway.server as gateway_server
from fastapi.testclient import TestClient

from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import Generation, ModelBusyError, UnsupportedModelError
from openforge.gateway.types import StartSessionRequest


def _start_session_payload(model_name: str = "model-a") -> dict[str, object]:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
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


def _create_test_app(monkeypatch, *, store: SQLiteOpenForgeStore, runtime: object):
    monkeypatch.setattr(gateway_server, "_build_store", lambda cfg: store)
    monkeypatch.setattr(gateway_server, "Runtime", lambda cfg: runtime)
    return gateway_server.create_app(_server_config())


class _FakeRuntime:
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

    def start(self, *, runtime_config) -> str:
        model_name = str(runtime_config.model.model_name_or_path)
        if model_name not in self._supported_models:
            raise UnsupportedModelError(model_name)
        if self._current_model is None:
            self._current_model = model_name
            return model_name
        if self._current_model != model_name:
            raise ModelBusyError(model_name)
        return model_name

    def tokenize_messages(
        self,
        model_name: str,
        messages: list[dict[str, str]],
    ) -> list[int]:
        if self._current_model != model_name:
            raise ModelBusyError(model_name)
        token_count = sum(len(message["content"].split()) for message in messages)
        return list(range(1, token_count + 2))

    def generate(
        self,
        model_name: str,
        *,
        prompt_token_ids: list[int],
        sampling_params: dict[str, object] | None = None,
    ) -> Generation:
        if self._current_model != model_name:
            raise ModelBusyError(model_name)
        self.last_sampling_params = sampling_params
        prompt_tail = int(prompt_token_ids[-1]) if prompt_token_ids else 0
        return Generation(
            token_ids=[10 + prompt_tail, 20 + prompt_tail],
            logprobs=[-0.1, -0.2],
            rollout_model_version="v4",
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self._current_model = None


def test_gateway_http_flow(monkeypatch) -> None:
    """Exercise the HTTP flow across session, trajectory, generate, and teardown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime()
        app = _create_test_app(monkeypatch, store=store, runtime=runtime)
        with TestClient(app) as client:
            models = client.get("/models")
            assert models.status_code == 200
            assert models.json() == {
                "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
                "active_model": None,
            }

            started = client.post("/start_session", json=_start_session_payload("model-a"))
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
                "rollout_model_version": "v4",
            }
            assert runtime.last_sampling_params == {"temperature": 0.5, "top_p": 0.9}

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


def test_gateway_http_health_and_models_reflect_loaded_model(monkeypatch) -> None:
    """Expose health and active-model state through the HTTP surface."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(monkeypatch, store=store, runtime=_FakeRuntime())
        with TestClient(app) as client:
            assert client.get("/health").json() == {"ok": True}
            assert client.get("/current_session").status_code == 404
            assert client.get("/models").json()["active_model"] is None
            assert client.get("/models").json()["models"] == [
                {"id": "model-a", "tokenizer": "model-a-tokenizer"}
            ]

            started = client.post("/start_session", json=_start_session_payload("model-a"))
            assert started.status_code == 200
            assert client.get("/models").json()["active_model"] == "model-a"
            assert client.get("/current_session").json() == started.json()


def test_gateway_http_start_session_reports_errors(
    monkeypatch,
) -> None:
    """Return the correct HTTP errors for unsupported or busy model starts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(
            monkeypatch,
            store=store,
            runtime=_FakeRuntime(("model-a", "model-b")),
        )
        with TestClient(app) as client:
            unsupported = client.post("/start_session", json=_start_session_payload("model-c"))
            assert unsupported.status_code == 404

            created = client.post("/start_session", json=_start_session_payload("model-a"))
            assert created.status_code == 200
            reused = client.post("/start_session", json=_start_session_payload("model-a"))
            assert reused.status_code == 409

            busy = client.post("/start_session", json=_start_session_payload("model-b"))
            assert busy.status_code == 409


def test_gateway_http_releases_model_after_last_session(monkeypatch) -> None:
    """Release the model after the last session and allow the next one to start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        runtime = _FakeRuntime(("model-a", "model-b"))
        app = _create_test_app(monkeypatch, store=store, runtime=runtime)
        with TestClient(app) as client:
            session_id = client.post("/start_session", json=_start_session_payload("model-a")).json()["session_id"]
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
            assert runtime.shutdown_count == 1
            assert client.get("/models").json()["active_model"] is None

            switched = client.post("/start_session", json=_start_session_payload("model-b"))
            assert switched.status_code == 200
            assert switched.json()["model"] == "model-b"


def test_gateway_http_generate_validates_request_and_unknown_records(
    monkeypatch,
) -> None:
    """Validate generate requests and return not-found errors for missing records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(monkeypatch, store=store, runtime=_FakeRuntime())
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

            session_id = client.post("/start_session", json=_start_session_payload("model-a")).json()["session_id"]
            missing_trajectory = client.post(
                "/generate",
                json={
                    "session_id": session_id,
                    "trajectory_id": "traj_missing",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert missing_trajectory.status_code == 404


def test_gateway_http_generate_after_end_trajectory_returns_conflict(
    monkeypatch,
) -> None:
    """Reject generation attempts after a trajectory has already completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(monkeypatch, store=store, runtime=_FakeRuntime())
        with TestClient(app) as client:
            session_id = client.post("/start_session", json=_start_session_payload("model-a")).json()["session_id"]
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


def test_gateway_http_end_session_requires_completed_trajectories(monkeypatch) -> None:
    """Reject end_session while there are still active trajectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(monkeypatch, store=store, runtime=_FakeRuntime())
        with TestClient(app) as client:
            missing = client.post("/end_session", json={"session_id": "missing"})
            assert missing.status_code == 404

            session_id = client.post("/start_session", json=_start_session_payload("model-a")).json()["session_id"]
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


def test_gateway_http_start_trajectory_can_fork_from_parent(monkeypatch) -> None:
    """Allow a new trajectory to copy the parent turn history via HTTP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
        app = _create_test_app(monkeypatch, store=store, runtime=_FakeRuntime())
        with TestClient(app) as client:
            session_id = client.post("/start_session", json=_start_session_payload("model-a")).json()["session_id"]
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


def test_create_app_accepts_server_config_without_injected_dependencies(
    monkeypatch,
) -> None:
    """Allow production-style app creation from the server-owned config alone."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime = _FakeRuntime()
        cfg = GatewayServerConfig(
            data=DataConfig(path=str(Path(tmpdir) / "gateway.sqlite3")),
            gateway=GatewayConfig(host="127.0.0.1", port=0),
            cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
        )
        monkeypatch.setattr(gateway_server, "Runtime", lambda cfg: runtime)
        app = gateway_server.create_app(cfg)
        assert app.title == "OpenForge Gateway"
