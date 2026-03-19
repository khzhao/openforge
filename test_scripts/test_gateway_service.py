# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.server import (
    ActiveSessionError,
    ActiveTrajectoriesRemainError,
    ConfiguredGatewayRuntimeController,
    GatewayGeneration,
    GatewayService,
    ModelBusyError,
    SessionClosedError,
    SessionNotFoundError,
    TrajectoryNotActiveError,
    TrajectoryNotFoundError,
    UnsupportedModelError,
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

    def tokenize_messages(
        self,
        model_name: str,
        messages: list[dict[str, str]],
    ) -> list[int]:
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
            token_ids=[100 + prompt_tail, 200 + prompt_tail],
            logprobs=[-0.1, -0.2],
            rollout_model_version=5,
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self._current_model = None


def test_gateway_service_start_generate_fork_and_end() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            controller = _FakeController()
            service = GatewayService(store=store, controller=controller)

            session = await service.start_session("model-a")
            root = await service.start_trajectory(session_id=session.session_id)

            generated = await service.generate(
                session_id=session.session_id,
                trajectory_id=root.trajectory_id,
                messages=[{"role": "user", "content": "hello world"}],
                sampling_params={"temperature": 0.7, "max_new_tokens": 32},
            )
            assert generated.session_id == session.session_id
            assert generated.trajectory_id == root.trajectory_id
            assert generated.token_ids == [103, 203]
            assert controller.last_sampling_params == {
                "temperature": 0.7,
                "max_new_tokens": 32,
            }

            child = await service.start_trajectory(
                session_id=session.session_id,
                parent_trajectory_id=root.trajectory_id,
            )
            child_turns = await store.list_turns(child.trajectory_id)
            assert len(child_turns) == 1
            assert child_turns[0].turn_index == 0
            assert child_turns[0].input_ids == [1, 2, 3, 103, 203]

            child_generated = await service.generate(
                session_id=session.session_id,
                trajectory_id=child.trajectory_id,
                messages=[{"role": "user", "content": "continue child"}],
            )
            assert child_generated.trajectory_id == child.trajectory_id

            await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=root.trajectory_id,
                final_reward=1.0,
            )
            await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=child.trajectory_id,
                final_reward=0.5,
            )

            ended = await service.end_session(session_id=session.session_id)
            assert ended == {
                "session_id": session.session_id,
                "status": "completed",
            }

            completed = await store.list_completed_trajectories(model_name="model-a")
            assert sorted(
                (trajectory.trajectory_id, trajectory.final_reward)
                for trajectory in completed
            ) == sorted(
                [
                    (root.trajectory_id, 1.0),
                    (child.trajectory_id, 0.5),
                ]
            )
            await store.close()

    asyncio.run(run())


def test_gateway_service_list_models_and_start_session_tracks_active_model() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        controller = _FakeController()
        service = GatewayService(store=store, controller=controller)

        assert await service.list_models() == {
            "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
            "active_model": None,
        }

        created = await service.start_session("model-a")

        assert await service.list_models() == {
            "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
            "active_model": "model-a",
        }
        assert created.model == "model-a"

        await store.close()

    asyncio.run(run())


def test_gateway_service_start_session_rejects_unsupported_busy_and_second_session() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(
            store=store,
            controller=_FakeController(("model-a", "model-b")),
        )

        with pytest.raises(UnsupportedModelError):
            await service.start_session("model-c")

        await service.start_session("model-a")

        with pytest.raises(ActiveSessionError):
            await service.start_session("model-a")

        with pytest.raises(ActiveSessionError):
            await service.start_session("model-b")

        await store.close()

    asyncio.run(run())


def test_gateway_service_releases_runtime_after_last_session_and_allows_switch() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        controller = _FakeController(("model-a", "model-b"))
        service = GatewayService(store=store, controller=controller)

        session = await service.start_session("model-a")
        trajectory = await service.start_trajectory(session_id=session.session_id)
        await service.generate(
            session_id=session.session_id,
            trajectory_id=trajectory.trajectory_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=trajectory.trajectory_id,
            final_reward=1.0,
        )
        ended = await service.end_session(session_id=session.session_id)

        assert ended["status"] == "completed"
        assert controller.shutdown_count == 1
        assert controller.current_model() is None

        created_again = await service.start_session("model-b")
        assert created_again.model == "model-b"
        assert controller.current_model() == "model-b"

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_unknown_session_raises() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())

        with pytest.raises(SessionNotFoundError, match="unknown session_id"):
            await service.generate(
                session_id="missing",
                trajectory_id="traj_missing",
                messages=[{"role": "user", "content": "hello"}],
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_trajectory_lifecycle_errors() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())
        session = await service.start_session("model-a")
        root = await service.start_trajectory(session_id=session.session_id)

        with pytest.raises(TrajectoryNotFoundError, match="unknown trajectory_id"):
            await service.generate(
                session_id=session.session_id,
                trajectory_id="traj_missing",
                messages=[{"role": "user", "content": "hello"}],
            )

        await service.generate(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            final_reward=1.0,
        )

        with pytest.raises(TrajectoryNotActiveError, match="is not active"):
            await service.generate(
                session_id=session.session_id,
                trajectory_id=root.trajectory_id,
                messages=[{"role": "user", "content": "again"}],
            )

        with pytest.raises(TrajectoryNotActiveError, match="is not active"):
            await service.start_trajectory(
                session_id=session.session_id,
                parent_trajectory_id=root.trajectory_id,
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_end_session_requires_all_trajectories_completed() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())
        session = await service.start_session("model-a")
        root = await service.start_trajectory(session_id=session.session_id)
        child = await service.start_trajectory(session_id=session.session_id)

        await service.generate(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            final_reward=1.0,
        )

        with pytest.raises(
            ActiveTrajectoriesRemainError,
            match="all trajectories must be ended",
        ):
            await service.end_session(session_id=session.session_id)

        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=child.trajectory_id,
            final_reward=0.0,
        )
        ended = await service.end_session(session_id=session.session_id)
        assert ended["status"] == "completed"

        with pytest.raises(SessionClosedError, match="is not active"):
            await service.start_trajectory(session_id=session.session_id)

        await store.close()

    asyncio.run(run())


def test_parse_generation_payload_prefers_weight_version() -> None:
    generation = ConfiguredGatewayRuntimeController._parse_generation_payload(
        {
            "output_ids": [10, 11],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10, None], [-0.2, 11, None]],
                "finish_reason": "stop",
                "weight_version": "37",
            },
        },
        fallback_policy_version=5,
    )

    assert generation.rollout_model_version == 37


def test_gateway_controller_parses_router_payload() -> None:
    generation = ConfiguredGatewayRuntimeController._parse_generation_payload(
        {
            "output_ids": [11, 12],
            "meta_info": {
                "output_token_logprobs": [
                    [-0.3, 11, "foo"],
                    [-0.4, 12, "bar"],
                ],
                "finish_reason": {"type": "stop"},
                "token_steps": [7, 7],
            },
        },
        fallback_policy_version=3,
    )

    assert generation.token_ids == [11, 12]
    assert generation.logprobs == [-0.3, -0.4]
    assert generation.finish_reason == "stop"
    assert generation.rollout_model_version == 7


def test_gateway_controller_parses_router_payload_without_output_ids() -> None:
    generation = ConfiguredGatewayRuntimeController._parse_generation_payload(
        {
            "meta_info": {
                "output_token_logprobs": [
                    [-0.25, 101],
                    [-0.5, 102],
                ],
                "finish_reason": "length",
            },
        },
        fallback_policy_version=9,
    )

    assert generation.token_ids == [101, 102]
    assert generation.logprobs == [-0.25, -0.5]
    assert generation.finish_reason == "length"
    assert generation.rollout_model_version == 9
