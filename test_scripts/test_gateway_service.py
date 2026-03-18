# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.server import (
    ConfiguredGatewayRuntimeController,
    GatewayGeneration,
    GatewayService,
    InvalidRolloutRewardError,
    ModelBusyError,
    NoActiveTrajectoriesError,
    SessionNotFoundError,
    UnsupportedModelError,
)
from openforge.gateway.types import RolloutReward


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
        n: int,
        sampling_params: dict[str, object] | None = None,
    ) -> list[GatewayGeneration]:
        self.ensure_model(model_name)
        self.last_sampling_params = sampling_params
        return [
            GatewayGeneration(
                token_ids=[100 + choice_index, 200 + choice_index],
                logprobs=[-0.1, -0.2],
                rollout_model_version=5,
            )
            for choice_index in range(n)
        ]


def test_gateway_service_create_generate_fork_and_end_session() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            controller = _FakeController()
            service = GatewayService(store=store, controller=controller)

            created = await service.create_session("model-a")
            generated = await service.generate(
                session_id=created.session_id,
                messages=[{"role": "user", "content": "hello world"}],
                n=1,
                sampling_params={"temperature": 0.7, "max_new_tokens": 32},
            )
            assert len(generated.choices) == 1
            root_rollout_id = generated.choices[0].rollout_id
            assert controller.last_sampling_params == {
                "temperature": 0.7,
                "max_new_tokens": 32,
            }

            forked = await service.generate(
                session_id=created.session_id,
                messages=[{"role": "user", "content": "fork now"}],
                n=2,
            )
            assert len(forked.choices) == 2
            child_rollout_ids = [choice.rollout_id for choice in forked.choices]
            assert root_rollout_id not in child_rollout_ids

            trajectories = await store.list_trajectories(created.session_id)
            turns_per_child = {
                trajectory.trajectory_id: len(await store.list_turns(trajectory.trajectory_id))
                for trajectory in trajectories
                if trajectory.status == "active"
            }
            assert sorted(turns_per_child.values()) == [2, 2]

            ended = await service.end_session(
                session_id=created.session_id,
                rewards=[
                    RolloutReward(rollout_id=child_rollout_ids[0], reward=1.0),
                    RolloutReward(rollout_id=child_rollout_ids[1], reward=0.5),
                ],
            )
            assert ended["status"] == "completed"

            completed = await store.list_completed_trajectories(model_name="model-a")
            assert sorted(trajectory.final_reward for trajectory in completed) == [0.5, 1.0]
            await store.close()

    asyncio.run(run())


def test_gateway_service_list_models_and_create_session_tracks_current_model() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        controller = _FakeController()
        service = GatewayService(store=store, controller=controller)

        assert await service.list_models() == {
            "models": ["model-a"],
            "current_model": None,
        }

        created = await service.create_session("model-a")

        assert await service.list_models() == {
            "models": ["model-a"],
            "current_model": "model-a",
        }

        trajectories = await store.list_trajectories(created.session_id, status="active")
        assert len(trajectories) == 1
        assert trajectories[0].parent_trajectory_id is None

        await store.close()

    asyncio.run(run())


def test_gateway_service_create_session_rejects_unsupported_and_busy_models() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(
            store=store,
            controller=_FakeController(("model-a", "model-b")),
        )

        with pytest.raises(UnsupportedModelError):
            await service.create_session("model-c")

        await service.create_session("model-a")

        with pytest.raises(ModelBusyError):
            await service.create_session("model-b")

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_unknown_session_raises() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())

        with pytest.raises(SessionNotFoundError, match="unknown session_id"):
            await service.generate(
                session_id="missing",
                messages=[{"role": "user", "content": "hello"}],
                n=1,
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_continue_after_fork_advances_all_active_rollouts() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())

        created = await service.create_session("model-a")
        forked = await service.generate(
            session_id=created.session_id,
            messages=[{"role": "user", "content": "fork"}],
            n=2,
        )
        active_rollout_ids = [choice.rollout_id for choice in forked.choices]

        continued = await service.generate(
            session_id=created.session_id,
            messages=[{"role": "user", "content": "continue"}],
            n=1,
        )

        assert len(continued.choices) == 2
        assert {choice.rollout_id for choice in continued.choices} == set(active_rollout_ids)

        for rollout_id in active_rollout_ids:
            turns = await store.list_turns(rollout_id)
            assert [turn.turn_index for turn in turns] == [0, 1]

        await store.close()

    asyncio.run(run())


def test_gateway_service_requires_exact_active_rollout_rewards() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            service = GatewayService(store=store, controller=_FakeController())
            created = await service.create_session("model-a")
            generated = await service.generate(
                session_id=created.session_id,
                messages=[{"role": "user", "content": "hello"}],
                n=2,
            )

            with pytest.raises(
                InvalidRolloutRewardError,
                match="exactly the active rollout ids",
            ):
                await service.end_session(
                    session_id=created.session_id,
                    rewards=[
                        RolloutReward(
                            rollout_id=generated.choices[0].rollout_id,
                            reward=1.0,
                        )
                    ],
                )
            await store.close()

    asyncio.run(run())


def test_gateway_service_end_session_unknown_session_raises() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())

        with pytest.raises(SessionNotFoundError, match="unknown session_id"):
            await service.end_session(
                session_id="missing",
                rewards=[],
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_and_end_session_require_active_rollouts() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = GatewayService(store=store, controller=_FakeController())
        created = await service.create_session("model-a")

        generated = await service.generate(
            session_id=created.session_id,
            messages=[{"role": "user", "content": "hello"}],
            n=1,
        )
        await service.end_session(
            session_id=created.session_id,
            rewards=[
                RolloutReward(
                    rollout_id=generated.choices[0].rollout_id,
                    reward=1.0,
                )
            ],
        )

        with pytest.raises(NoActiveTrajectoriesError, match="no active trajectories"):
            await service.generate(
                session_id=created.session_id,
                messages=[{"role": "user", "content": "again"}],
                n=1,
            )

        with pytest.raises(NoActiveTrajectoriesError, match="no active trajectories"):
            await service.end_session(
                session_id=created.session_id,
                rewards=[
                    RolloutReward(
                        rollout_id=generated.choices[0].rollout_id,
                        reward=1.0,
                    )
                ],
            )

        await store.close()

    asyncio.run(run())


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
