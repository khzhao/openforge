# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import (
    Generation,
    ModelBusyError,
    Runtime,
    UnsupportedModelError,
)
from openforge.gateway.service import (
    ActiveSessionError,
    ActiveTrajectoriesRemainError,
    Service,
    SessionClosedError,
    SessionNotFoundError,
    TrajectoryNotActiveError,
    TrajectoryNotFoundError,
)
from openforge.gateway.types import StartSessionRequest


def _start_session_kwargs(model_name: str = "model-a") -> dict[str, object]:
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
    return {"runtime_config": request.runtime}


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
        messages: list[dict[str, str]],
    ) -> list[int]:
        token_count = sum(len(message["content"].split()) for message in messages)
        return list(range(1, token_count + 2))

    def generate(
        self,
        *,
        prompt_token_ids: list[int],
        sampling_params: dict[str, object] | None = None,
    ) -> Generation:
        self.last_sampling_params = sampling_params
        prompt_tail = int(prompt_token_ids[-1]) if prompt_token_ids else 0
        return Generation(
            token_ids=[100 + prompt_tail, 200 + prompt_tail],
            rollout_model_version="v5",
        )

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self._current_model = None


class _FailingTokenizeRuntime(_FakeRuntime):
    def tokenize_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[int]:
        raise RuntimeError("template boom")


def test_gateway_service_start_generate_fork_and_end() -> None:
    """Exercise the main service lifecycle across session, trajectory, and turn writes."""

    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
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
            assert generated.rollout_model_version == "v5"
            assert runtime.last_sampling_params == {
                "temperature": 0.7,
                "max_new_tokens": 32,
            }
            root_turns = await store.list_turns(root.trajectory_id)

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
            assert ended.session_id == session.session_id
            assert ended.status == "completed"

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
    """Verify model listing and active-model tracking through session start."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime()
        service = Service(store=store, runtime=runtime)

        assert await service.list_models() == {
            "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
            "active_model": None,
        }
        assert await service.current_session() is None

        created = await service.start_session(**_start_session_kwargs("model-a"))

        assert await service.list_models() == {
            "models": [{"id": "model-a", "tokenizer": "model-a-tokenizer"}],
            "active_model": "model-a",
        }
        assert created.model == "model-a"
        assert await service.current_session() == created

        await store.close()

    asyncio.run(run())


def test_gateway_service_start_session_rejects_second_active_session() -> None:
    """Reject a second start_session call while another session is active."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(
            store=store,
            runtime=_FakeRuntime(("model-a", "model-b")),
        )

        with pytest.raises(UnsupportedModelError):
            await service.start_session(**_start_session_kwargs("model-c"))

        await service.start_session(**_start_session_kwargs("model-a"))

        with pytest.raises(ActiveSessionError):
            await service.start_session(**_start_session_kwargs("model-a"))

        with pytest.raises(ActiveSessionError):
            await service.start_session(**_start_session_kwargs("model-b"))

        await store.close()

    asyncio.run(run())


def test_gateway_service_releases_runtime_after_last_session_and_allows_switch() -> (
    None
):
    """Release the runtime after the last session and allow a model switch."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime(("model-a", "model-b"))
        service = Service(store=store, runtime=runtime)

        session = await service.start_session(**_start_session_kwargs("model-a"))
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

        assert ended.status == "completed"
        assert runtime.shutdown_count == 1
        assert runtime.current_model() is None

        created_again = await service.start_session(**_start_session_kwargs("model-b"))
        assert created_again.model == "model-b"
        assert runtime.current_model() == "model-b"

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_unknown_session_raises() -> None:
    """Raise a not-found error when generation targets an unknown session."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())

        with pytest.raises(SessionNotFoundError, match="unknown session_id"):
            await service.generate(
                session_id="missing",
                trajectory_id="traj_missing",
                messages=[{"role": "user", "content": "hello"}],
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_wraps_tokenization_failure() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FailingTokenizeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
        trajectory = await service.start_trajectory(session_id=session.session_id)

        with pytest.raises(
            ValueError,
            match="failed to tokenize messages with chat template: template boom",
        ):
            await service.generate(
                session_id=session.session_id,
                trajectory_id=trajectory.trajectory_id,
                messages=[{"role": "user", "content": "hello"}],
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_trajectory_lifecycle_errors() -> None:
    """Raise the expected errors across invalid trajectory lifecycle operations."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
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
    """Require all trajectories to complete before ending the session."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
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
        assert ended.status == "completed"

        with pytest.raises(SessionClosedError, match="is not active"):
            await service.start_trajectory(session_id=session.session_id)

        await store.close()

    asyncio.run(run())


def test_parse_generation_payload_prefers_weight_version() -> None:
    """Prefer the explicit weight version over token-step fallbacks."""
    generation = Runtime._parse_generation_payload(
        {
            "output_ids": [10, 11],
            "meta_info": {
                "finish_reason": "stop",
                "weight_version": "default",
            },
        },
    )

    assert generation.rollout_model_version == "default"


def test_gateway_controller_parses_router_payload() -> None:
    """Parse the standard router payload shape into a gateway generation."""
    generation = Runtime._parse_generation_payload(
        {
            "output_ids": [11, 12],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "weight_version": "policy-7",
            },
        },
    )

    assert generation.token_ids == [11, 12]
    assert generation.finish_reason == "stop"
    assert generation.rollout_model_version == "policy-7"


def test_gateway_controller_parses_router_payload_with_meta_token_ids() -> None:
    generation = Runtime._parse_generation_payload(
        {
            "meta_info": {
                "token_ids": [101, 102],
                "finish_reason": "length",
                "weight_version": "default",
            },
        },
    )

    assert generation.token_ids == [101, 102]
    assert generation.finish_reason == "length"
    assert generation.rollout_model_version == "default"


def test_gateway_controller_requires_output_token_ids() -> None:
    with pytest.raises(ValueError, match="output_ids or token_ids"):
        Runtime._parse_generation_payload(
            {
                "meta_info": {
                    "finish_reason": "stop",
                    "weight_version": "default",
                },
            }
        )


def test_gateway_controller_parses_router_payload_without_extra_fields() -> None:
    generation = Runtime._parse_generation_payload(
        {
            "output_ids": [11, 12],
            "meta_info": {
                "finish_reason": "stop",
                "weight_version": "default",
            },
        }
    )

    assert generation.token_ids == [11, 12]
    assert generation.finish_reason == "stop"
    assert generation.rollout_model_version == "default"


def test_parse_generation_payload_requires_weight_version() -> None:
    """Reject generate payloads that do not report a weight version."""
    with pytest.raises(ValueError, match="weight_version"):
        Runtime._parse_generation_payload(
            {
                "output_ids": [10],
                "meta_info": {
                    "finish_reason": "stop",
                },
            }
        )
