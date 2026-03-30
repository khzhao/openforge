# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

from openforge.data import SQLiteOpenForgeStore, Trajectory
from openforge.gateway.runtime import (
    Generation,
    Runtime,
)
from openforge.gateway.service import Service, _PendingGenerate
from openforge.gateway.types import (
    ChatCompletionCreateRequest,
    ChatMessage,
    StartSessionRequest,
)


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
        self._update_callback = None

    def set_update_callback(self, callback) -> None:
        self._update_callback = callback

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

    def status(self) -> dict[str, object]:
        return {
            "active": self._train_loop is not None,
            "policy_version": self.policy_version,
            "heartbeat_age_s": 0.0,
            "last_update_age_s": None,
        }

    async def shutdown(self) -> None:
        await self.end_session()


def _start_session_kwargs(model_name: str = "model-a") -> dict[str, object]:
    return _start_session_kwargs_with_replicas(model_name=model_name, replicas=1)


def _start_session_kwargs_with_replicas(
    *,
    model_name: str = "model-a",
    replicas: int,
) -> dict[str, object]:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": model_name,
                    "tokenizer_name_or_path": f"{model_name}-tokenizer",
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
                            "replicas": replicas,
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
        self._train = _FakeTrainRuntime()
        self.last_sampling_params: dict[str, object] | None = None
        self.last_trajectory_ids: list[str] | None = None
        self.released_trajectory_ids: list[list[str]] = []
        self.shutdown_count = 0
        self.rollout_min_weight_versions: list[int] = []
        self.rollout_status_calls = 0

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
            token_ids=[100 + prompt_tail, 200 + prompt_tail],
            rollout_model_version=5,
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

    def rollout_status(self) -> dict[str, object]:
        self.rollout_status_calls += 1
        min_weight_version = (
            self.rollout_min_weight_versions.pop(0)
            if self.rollout_min_weight_versions
            else 0
        )
        return {
            "heartbeat_age_s": 0.0,
            "latest_published_train_version": 0,
            "min_weight_version": min_weight_version,
            "max_weight_version": min_weight_version,
            "stale_worker_count": 0,
            "workers": {},
        }

    def status(self) -> dict[str, object]:
        return {
            "current_model": self._current_model,
            "train": self._train.status(),
            "rollout": self.rollout_status(),
            "cluster": {},
        }

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


def _chat_request(
    *,
    session_id: str,
    trajectory_id: str,
    content: str,
    model: str = "model-a",
    purpose: str = "train",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    max_completion_tokens: int | None = None,
) -> ChatCompletionCreateRequest:
    payload: dict[str, object] = {
        "_openforge": {
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "group_id": None,
            "purpose": purpose,
        },
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    return ChatCompletionCreateRequest.model_validate(payload)


def test_gateway_service_start_generate_and_end() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            assert len(_FakeTrainLoop.instances) == 1
            assert _FakeTrainLoop.instances[0].session_id == session.session_id
            assert _FakeTrainLoop.instances[0].started is True
            root = await service.start_trajectory(session_id=session.session_id)

            generated = await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=root.trajectory_id,
                    content="hello world",
                    temperature=0.7,
                    top_k=-1,
                    repetition_penalty=1.1,
                    max_completion_tokens=32,
                ),
            )
            payload = generated.model_dump(mode="json", exclude_none=True)
            assert payload["id"] == f"chatcmpl_{root.trajectory_id}_0"
            assert payload["object"] == "chat.completion"
            assert payload["model"] == "model-a"
            assert payload["choices"] == [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": "reply-3"},
                }
            ]
            assert payload["usage"] == {
                "completion_tokens": 2,
                "prompt_tokens": 3,
                "total_tokens": 5,
            }
            assert payload["metadata"] == {
                "session_id": session.session_id,
                "trajectory_id": root.trajectory_id,
                "token_ids": [103, 203],
                "rollout_model_version": 5,
            }
            assert runtime.last_sampling_params == {
                "temperature": 0.7,
                "top_k": -1,
                "repetition_penalty": 1.1,
                "max_new_tokens": 32,
            }
            assert runtime.last_trajectory_ids == [root.trajectory_id]

            child = await service.start_trajectory(session_id=session.session_id)
            child_turns = await store.list_turns(child.trajectory_id)
            assert child_turns == []

            child_generated = await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=child.trajectory_id,
                    content="continue child",
                ),
            )
            child_payload = child_generated.model_dump(mode="json", exclude_none=True)
            assert child_payload["id"] == f"chatcmpl_{child.trajectory_id}_0"
            assert child_payload["choices"][0]["message"]["role"] == "assistant"

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
            assert _FakeTrainLoop.instances[0].stopped is True

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


def test_gateway_service_caps_generate_batch_per_rollout_replica() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime()
        service = Service(store=store, runtime=runtime)

        session = await service.start_session(
            **_start_session_kwargs_with_replicas(model_name="model-a", replicas=2)
        )
        assert service._generate_batch_max_size == 128

        loop = asyncio.get_running_loop()
        service._pending_generates = [
            _PendingGenerate(
                session_id=session.session_id,
                trajectory_id=f"traj_{i}",
                group_id=None,
                messages=[],
                tools=None,
                sampling_params={},
                future=loop.create_future(),
                batch_key=("batch", "tools", "params"),
            )
            for i in range(200)
        ]
        batch = service._take_generate_batch_locked()
        assert len(batch) == 128

        ended = await service.end_session(session_id=session.session_id)
        assert ended.status == "completed"
        assert service._generate_batch_max_size == Service.GENERATE_BATCH_MAX_SIZE
        await store.close()

    asyncio.run(run())


def test_gateway_service_current_session_tracks_active_model() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime()
        service = Service(store=store, runtime=runtime)

        assert runtime.current_model() is None
        assert await service.current_session() is None

        created = await service.start_session(**_start_session_kwargs("model-a"))

        assert runtime.current_model() == "model-a"
        assert created.model == "model-a"
        assert await service.current_session() == created

        await store.close()

    asyncio.run(run())


def test_gateway_service_validation_trajectories_are_not_persisted() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            started = await service.start_trajectory(
                session_id=session.session_id,
                purpose="validation",
            )
            response = await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=started.trajectory_id,
                    content="hello validation",
                    purpose="validation",
                )
            )
            assert response.metadata["trajectory_id"] == started.trajectory_id

            ended = await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=started.trajectory_id,
                final_reward=0.75,
            )
            assert ended.status == "completed"
            assert runtime.last_trajectory_ids == [started.trajectory_id]
            assert runtime.released_trajectory_ids == [[started.trajectory_id]]
            assert await store.list_trajectories(session.session_id) == []
            await store.close()

    asyncio.run(run())


def test_gateway_service_duplicate_validation_end_is_idempotent() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            started = await service.start_trajectory(
                session_id=session.session_id,
                purpose="validation",
            )
            await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=started.trajectory_id,
                    content="hello validation",
                    purpose="validation",
                )
            )

            first = await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=started.trajectory_id,
                final_reward=0.75,
            )
            second = await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=started.trajectory_id,
                final_reward=0.75,
            )

            assert first.status == "completed"
            assert second.status == "completed"
            assert runtime.released_trajectory_ids == [[started.trajectory_id]]
            assert await store.list_trajectories(session.session_id) == []
            await store.close()

    asyncio.run(run())


def test_gateway_service_waits_for_rollout_policy_version() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            runtime.rollout_min_weight_versions = [0, 1, 2]
            service = Service(store=store, runtime=runtime)
            session = await service.start_session(**_start_session_kwargs("model-a"))
            original_poll = service.ROLLOUT_VERSION_POLL_SECONDS
            service.ROLLOUT_VERSION_POLL_SECONDS = 0.001
            try:
                response = await service.wait_for_rollout_policy_version(
                    session_id=session.session_id,
                    policy_version=2,
                    timeout_s=1.0,
                )
            finally:
                service.ROLLOUT_VERSION_POLL_SECONDS = original_poll

            assert response.session_id == session.session_id
            assert response.policy_version == 2
            assert response.min_weight_version == 2
            assert runtime.rollout_status_calls == 3

            await service.shutdown()

    asyncio.run(run())


def test_gateway_service_blocks_train_group_admission_until_capacity_frees() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            first = await service.start_trajectory_groups(
                session_id=session.session_id,
                counts=[1],
                group_ids=[None],
            )
            first_trajectory_id = first.trajectory_ids[0][0]

            second_task = asyncio.create_task(
                service.start_trajectory_groups(
                    session_id=session.session_id,
                    counts=[1],
                    group_ids=[None],
                )
            )
            await asyncio.sleep(service.TRAIN_ADMISSION_POLL_INITIAL_SECONDS * 2)
            assert second_task.done() is False

            await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=first_trajectory_id,
                final_reward=1.0,
            )
            await asyncio.sleep(service.TRAIN_ADMISSION_POLL_INITIAL_SECONDS * 2)
            assert second_task.done() is False

            await store.update_trajectory(
                Trajectory(
                    trajectory_id=first_trajectory_id,
                    session_id=session.session_id,
                    group_id=None,
                    status="trained",
                    final_reward=1.0,
                )
            )
            second = await asyncio.wait_for(second_task, timeout=3.0)
            assert second.trajectory_ids[0][0] != first_trajectory_id
            await store.close()

    asyncio.run(run())


def test_gateway_service_reuses_explicit_group_trajectory_ids() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            trajectory_ids = [["traj_explicit_1", "traj_explicit_2"]]
            first = await service.start_trajectory_groups(
                session_id=session.session_id,
                counts=[2],
                trajectory_ids=trajectory_ids,
                group_ids=["group_a"],
                purpose="validation",
            )
            second = await service.start_trajectory_groups(
                session_id=session.session_id,
                counts=[2],
                trajectory_ids=trajectory_ids,
                group_ids=["group_a"],
                purpose="validation",
            )

            assert first.trajectory_ids == trajectory_ids
            assert second.trajectory_ids == trajectory_ids
            assert sorted(service._active_trajectories) == sorted(trajectory_ids[0])
            await store.close()

    asyncio.run(run())


def test_gateway_service_concurrent_duplicate_train_start_is_idempotent() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            started = await asyncio.wait_for(
                asyncio.gather(
                    service.start_trajectory(
                        session_id=session.session_id,
                        trajectory_id="traj_explicit_single",
                    ),
                    service.start_trajectory(
                        session_id=session.session_id,
                        trajectory_id="traj_explicit_single",
                    ),
                ),
                timeout=1.0,
            )

            assert [item.trajectory_id for item in started] == [
                "traj_explicit_single",
                "traj_explicit_single",
            ]
            stored = await store.list_trajectories(session.session_id)
            assert [trajectory.trajectory_id for trajectory in stored] == [
                "traj_explicit_single"
            ]
            await store.close()

    asyncio.run(run())


def test_gateway_service_concurrent_duplicate_train_group_start_is_idempotent() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            started = await asyncio.wait_for(
                asyncio.gather(
                    service.start_trajectory_groups(
                        session_id=session.session_id,
                        counts=[1],
                        trajectory_ids=[["traj_explicit_group"]],
                        group_ids=["group_a"],
                    ),
                    service.start_trajectory_groups(
                        session_id=session.session_id,
                        counts=[1],
                        trajectory_ids=[["traj_explicit_group"]],
                        group_ids=["group_a"],
                    ),
                ),
                timeout=1.0,
            )

            assert [item.trajectory_ids for item in started] == [
                [["traj_explicit_group"]],
                [["traj_explicit_group"]],
            ]
            stored = await store.list_trajectories(session.session_id)
            assert [trajectory.trajectory_id for trajectory in stored] == [
                "traj_explicit_group"
            ]
            await store.close()

    asyncio.run(run())


def test_gateway_service_reports_trajectory_statuses() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs("model-a"))
            trajectory = await service.start_trajectory(session_id=session.session_id)

            active = await service.trajectory_statuses(
                session_id=session.session_id,
                trajectory_ids=[trajectory.trajectory_id],
            )
            assert active.trajectories[0].status == "active"

            await service.end_trajectory(
                session_id=session.session_id,
                trajectory_id=trajectory.trajectory_id,
                final_reward=1.0,
            )
            completed = await service.trajectory_statuses(
                session_id=session.session_id,
                trajectory_ids=[trajectory.trajectory_id],
            )
            assert completed.trajectories[0].status == "completed"
            await store.close()

    asyncio.run(run())


def test_gateway_service_status_includes_gateway_train_and_rollout() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime()
        service = Service(store=store, runtime=runtime)

        session = await service.start_session(**_start_session_kwargs("model-a"))
        status = await service.status()

        assert status["session_id"] == session.session_id
        assert status["gateway"]["pending_generate_count"] == 0
        assert status["train"]["active"] is True
        assert status["rollout"]["max_weight_version"] == 0
        assert status["rollout"]["max_version_skew"] == 0.0
        await store.close()

    asyncio.run(run())


def test_gateway_service_start_session_rejects_second_active_session() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(
            store=store,
            runtime=_FakeRuntime(("model-a", "model-b")),
        )

        with expect_raises(Exception, "unsupported model"):
            await service.start_session(**_start_session_kwargs("model-c"))

        await service.start_session(**_start_session_kwargs("model-a"))

        with expect_raises(Exception, "another session is already active"):
            await service.start_session(**_start_session_kwargs("model-a"))

        with expect_raises(Exception, "another session is already active"):
            await service.start_session(**_start_session_kwargs("model-b"))

        await store.close()

    asyncio.run(run())


def test_gateway_service_releases_runtime_after_last_session_and_allows_switch() -> (
    None
):
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        runtime = _FakeRuntime(("model-a", "model-b"))
        service = Service(store=store, runtime=runtime)

        session = await service.start_session(**_start_session_kwargs("model-a"))
        trajectory = await service.start_trajectory(session_id=session.session_id)
        await service.generate(
            request=_chat_request(
                session_id=session.session_id,
                trajectory_id=trajectory.trajectory_id,
                content="hello",
            ),
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
        assert _FakeTrainLoop.instances[0].stopped is True

        created_again = await service.start_session(**_start_session_kwargs("model-b"))
        assert created_again.model == "model-b"
        assert runtime.current_model() == "model-b"

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_unknown_session_raises() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())

        with expect_raises(Exception, "unknown session_id"):
            await service.generate(
                request=_chat_request(
                    session_id="missing",
                    trajectory_id="traj_missing",
                    content="hello",
                ),
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_generate_wraps_tokenization_failure() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FailingTokenizeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
        trajectory = await service.start_trajectory(session_id=session.session_id)

        with expect_raises(
            Exception,
            "template boom",
        ):
            await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=trajectory.trajectory_id,
                    content="hello",
                ),
            )

        await store.close()

    asyncio.run(run())


def test_gateway_service_trajectory_lifecycle_errors() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
        root = await service.start_trajectory(session_id=session.session_id)

        generated = await service.generate(
            request=_chat_request(
                session_id=session.session_id,
                trajectory_id="traj_missing",
                content="hello",
            ),
        )
        assert generated.metadata["trajectory_id"] == "traj_missing"

        await service.generate(
            request=_chat_request(
                session_id=session.session_id,
                trajectory_id=root.trajectory_id,
                content="hello",
            ),
        )
        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            final_reward=1.0,
        )

        generated_again = await service.generate(
            request=_chat_request(
                session_id=session.session_id,
                trajectory_id="traj_missing",
                content="again",
            ),
        )
        assert generated_again.metadata["trajectory_id"] == "traj_missing"

        with expect_raises(Exception, f"trajectory {root.trajectory_id} is not active"):
            await service.generate(
                request=_chat_request(
                    session_id=session.session_id,
                    trajectory_id=root.trajectory_id,
                    content="again",
                ),
            )
        await store.close()

    asyncio.run(run())


def test_gateway_service_end_session_requires_all_trajectories_completed() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        service = Service(store=store, runtime=_FakeRuntime())
        session = await service.start_session(**_start_session_kwargs("model-a"))
        root = await service.start_trajectory(session_id=session.session_id)
        child = await service.start_trajectory(session_id=session.session_id)

        await service.generate(
            request=_chat_request(
                session_id=session.session_id,
                trajectory_id=root.trajectory_id,
                content="hello",
            ),
        )
        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=root.trajectory_id,
            final_reward=1.0,
        )

        with expect_raises(
            Exception,
            "all trajectories must be ended",
        ):
            await service.end_session(session_id=session.session_id)

        await service.end_trajectory(
            session_id=session.session_id,
            trajectory_id=child.trajectory_id,
            final_reward=0.0,
        )
        ended = await service.end_session(session_id=session.session_id)
        assert ended.status == "completed"

        with expect_raises(Exception, "is not active"):
            await service.start_trajectory(session_id=session.session_id)

        await store.close()

    asyncio.run(run())


def test_gateway_service_ignores_late_turn_count_updates_for_finished_trajectories() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "gateway.sqlite3")
            runtime = _FakeRuntime()
            service = Service(store=store, runtime=runtime)

            session = await service.start_session(**_start_session_kwargs())
            started = await service.start_trajectory(session_id=session.session_id)

            original_append_turns = store.append_turns

            async def append_turns_and_finish(turns):
                await original_append_turns(turns)
                trajectory_id = turns[0].trajectory_id
                service._active_turn_counts.pop(trajectory_id, None)
                service._active_trajectories.pop(trajectory_id, None)

            store.append_turns = append_turns_and_finish  # type: ignore[method-assign]

            outputs = await service._execute_generate_batch(
                session_id=session.session_id,
                trajectory_ids=[started.trajectory_id],
                group_ids=[None],
                messages_per_item=[
                    [
                        ChatMessage(role="user", content="hello world")
                    ]
                ],
                tools=None,
                sampling_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": 8,
                },
            )

            assert len(outputs) == 1
            stored_turns = await store.list_turns(started.trajectory_id)
            assert len(stored_turns) == 1
            assert started.trajectory_id not in service._active_turn_counts
            await store.close()

    asyncio.run(run())


def test_parse_generation_payload_parses_numeric_weight_version() -> None:
    generation = Runtime._parse_generation_info(
        {
            "text": "hello",
            "output_ids": [10, 11],
            "meta_info": {
                "finish_reason": "stop",
                "weight_version": 0,
                "output_token_logprobs": [[-0.1, 10, "a"], [-0.2, 11, "b"]],
            },
        },
    )
    assert generation["rollout_model_version"] == 0
    assert generation["rollout_log_probs"] == [-0.1, -0.2]


def test_gateway_controller_parses_router_payload() -> None:
    generation = Runtime._parse_generation_info(
        {
            "text": "hello",
            "output_ids": [11, 12],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 11, "a"], [-0.2, 12, "b"]],
                "finish_reason": {"type": "stop"},
                "weight_version": 7,
            },
        },
    )
    assert generation["text"] == "hello"
    assert generation["token_ids"] == [11, 12]
    assert generation["finish_reason"] == "stop"
    assert generation["rollout_model_version"] == 7
    assert generation["rollout_log_probs"] == [-0.1, -0.2]


def test_gateway_controller_parses_router_payload_with_meta_token_ids() -> None:
    generation = Runtime._parse_generation_info(
        {
            "text": "hello",
            "meta_info": {
                "token_ids": [101, 102],
                "output_token_logprobs": [[-0.1, 101, "a"], [-0.2, 102, "b"]],
                "finish_reason": "length",
                "weight_version": 0,
            },
        },
    )
    assert generation["token_ids"] == [101, 102]
    assert generation["finish_reason"] == "length"
    assert generation["rollout_model_version"] == 0
    assert generation["rollout_log_probs"] == [-0.1, -0.2]


def test_gateway_controller_requires_output_token_ids() -> None:
    with expect_raises(ValueError, "output_ids or token_ids"):
        Runtime._parse_generation_info(
            {
                "text": "hello",
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 10, "a"]],
                    "finish_reason": "stop",
                    "weight_version": 0,
                },
            }
        )


def test_gateway_controller_parses_router_payload_without_extra_fields() -> None:
    generation = Runtime._parse_generation_info(
        {
            "text": "hello",
            "output_ids": [11, 12],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 11, "a"], [-0.2, 12, "b"]],
                "finish_reason": "stop",
                "weight_version": 0,
            },
        }
    )
    assert generation["token_ids"] == [11, 12]
    assert generation["finish_reason"] == "stop"
    assert generation["rollout_model_version"] == 0
    assert generation["rollout_log_probs"] == [-0.1, -0.2]


def test_parse_generation_payload_requires_weight_version() -> None:
    with expect_raises(ValueError, "weight_version"):
        Runtime._parse_generation_info(
            {
                "text": "hello",
                "output_ids": [10],
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 10, "a"]],
                    "finish_reason": "stop",
                },
            }
        )


def main() -> int:
    return run_tests(
        [
            test_gateway_service_start_generate_and_end,
            test_gateway_service_current_session_tracks_active_model,
            test_gateway_service_validation_trajectories_are_not_persisted,
            test_gateway_service_blocks_train_group_admission_until_capacity_frees,
            test_gateway_service_reports_trajectory_statuses,
            test_gateway_service_status_includes_gateway_train_and_rollout,
            test_gateway_service_start_session_rejects_second_active_session,
            test_gateway_service_releases_runtime_after_last_session_and_allows_switch,
            test_gateway_service_generate_unknown_session_raises,
            test_gateway_service_generate_wraps_tokenization_failure,
            test_gateway_service_trajectory_lifecycle_errors,
            test_gateway_service_end_session_requires_all_trajectories_completed,
            test_gateway_service_ignores_late_turn_count_updates_for_finished_trajectories,
            test_parse_generation_payload_parses_numeric_weight_version,
            test_gateway_controller_parses_router_payload,
            test_gateway_controller_parses_router_payload_with_meta_token_ids,
            test_gateway_controller_requires_output_token_ids,
            test_gateway_controller_parses_router_payload_without_extra_fields,
            test_parse_generation_payload_requires_weight_version,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
