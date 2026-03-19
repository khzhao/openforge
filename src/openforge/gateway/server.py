# Copyright 2026 openforge

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from openforge.configs.models import GatewayConfig, OpenForgeConfig
from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn
from openforge.gateway.types import (
    EndSessionRequest,
    EndSessionResponse,
    EndTrajectoryRequest,
    EndTrajectoryResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateResult,
    GetPolicyVersionRequest,
    GetPolicyVersionResponse,
    ModelsResponse,
    StartSessionRequest,
    StartSessionResponse,
    StartSessionResult,
    StartTrajectoryRequest,
    StartTrajectoryResponse,
    StartTrajectoryResult,
)

__all__ = [
    "ConfiguredGatewayRuntimeController",
    "GatewayGeneration",
    "GatewayService",
    "ModelBusyError",
    "UnsupportedModelError",
    "create_app",
]


class UnsupportedModelError(ValueError):
    """Raised when the caller requests a model this gateway does not support."""


class ModelBusyError(RuntimeError):
    """Raised when a different model is already active in the single-model slot."""


class ActiveSessionError(RuntimeError):
    """Raised when another session is already active."""


class SessionNotFoundError(RuntimeError):
    """Raised when a session id does not exist."""


class SessionClosedError(RuntimeError):
    """Raised when a session is no longer active."""


class TrajectoryNotFoundError(RuntimeError):
    """Raised when a trajectory id does not exist."""


class TrajectoryNotActiveError(RuntimeError):
    """Raised when a trajectory is not active."""


class ActiveTrajectoriesRemainError(RuntimeError):
    """Raised when a session still has active trajectories."""


@dataclass(slots=True)
class GatewayGeneration:
    """One generated token sequence returned by the runtime controller."""

    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str = "stop"
    rollout_model_version: int = 1

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have the same length")


@dataclass(slots=True)
class ManagedRuntimeSlot:
    placement_groups: dict[str, Any]
    rollout_manager: Any
    policy_version: int
    started_ray: bool = False

    def shutdown(self) -> None:
        import ray

        try:
            self.rollout_manager.shutdown()
        finally:
            if self.started_ray and ray.is_initialized():
                ray.shutdown()


class ConfiguredGatewayRuntimeController:
    """Single-model runtime controller for the gateway."""

    def __init__(
        self,
        *,
        cfg: OpenForgeConfig,
    ) -> None:
        self._cfg = cfg
        self._supported_model = cfg.model.model_name_or_path
        self._tokenizer_name = cfg.model.tokenizer_name_or_path
        self._loaded_model: str | None = None
        self._tokenizer = None
        self._runtime: ManagedRuntimeSlot | None = None

    def list_models(self) -> list[dict[str, str]]:
        return [
            {
                "id": self._supported_model,
                "tokenizer": self._tokenizer_name,
            }
        ]

    def current_model(self) -> str | None:
        return self._loaded_model

    def ensure_model(self, model_name: str) -> None:
        if model_name != self._supported_model:
            raise UnsupportedModelError(f"unsupported model: {model_name}")
        if self._loaded_model is None:
            self._loaded_model = model_name
            self._runtime = self._start_runtime()
            return
        if self._loaded_model != model_name:
            raise ModelBusyError(
                f"gateway already has active model {self._loaded_model!r}; "
                f"cannot switch to {model_name!r}"
            )

    def tokenize_messages(
        self,
        model_name: str,
        messages: Sequence[dict[str, str]],
    ) -> list[int]:
        self.ensure_model(model_name)
        tokenizer = self._get_tokenizer()
        try:
            token_ids = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in messages
            )
            token_ids = tokenizer.encode(
                f"{prompt_text}\nassistant:",
                add_special_tokens=True,
            )
        return [int(token_id) for token_id in token_ids]

    def generate(
        self,
        model_name: str,
        *,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewayGeneration:
        self.ensure_model(model_name)
        assert self._runtime is not None
        payload_sampling_params = self._build_sampling_params(sampling_params)
        payload = self._runtime.rollout_manager.generate(
            payload_sampling_params,
            input_ids=[int(token_id) for token_id in prompt_token_ids],
            return_logprob=True,
        )
        return self._parse_generation_payload(
            payload,
            fallback_policy_version=self._runtime.policy_version,
        )

    def get_policy_version(self, model_name: str) -> int:
        self.ensure_model(model_name)
        assert self._runtime is not None
        policy_version = self._resolve_policy_version(self._runtime.rollout_manager)
        self._runtime.policy_version = policy_version
        return policy_version

    def shutdown(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._loaded_model = None
        if runtime is not None:
            runtime.shutdown()

    def _start_runtime(self) -> ManagedRuntimeSlot:
        import ray

        from openforge.runtime import create_rollout_manager
        from openforge.utils.ray import create_placement_groups

        started_ray = False
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
            started_ray = True

        placement_groups = create_placement_groups(self._cfg)
        rollout_manager = create_rollout_manager(self._cfg, placement_groups)
        return ManagedRuntimeSlot(
            placement_groups=placement_groups,
            rollout_manager=rollout_manager,
            policy_version=self._resolve_policy_version(rollout_manager),
            started_ray=started_ray,
        )

    def _build_sampling_params(
        self,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = self._cfg.rollout.request.model_dump()
        if overrides:
            payload.update(overrides)
        return payload

    @staticmethod
    def _resolve_policy_version(rollout_manager: Any) -> int:
        import ray

        versions = ray.get(
            [
                worker.get_weight_version.remote()
                for worker in rollout_manager.engine_workers
            ]
        )
        numeric_versions = [
            int(version)
            for version in versions
            if version is not None and str(version).isdigit()
        ]
        return max(numeric_versions, default=0)

    @staticmethod
    def _parse_generation_payload(
        payload: dict[str, Any],
        *,
        fallback_policy_version: int,
    ) -> GatewayGeneration:
        meta_info = payload.get("meta_info", {})
        token_logprobs = meta_info.get("output_token_logprobs", [])
        token_ids = ConfiguredGatewayRuntimeController._extract_token_ids(
            payload,
            token_logprobs=token_logprobs,
        )
        logprobs = ConfiguredGatewayRuntimeController._extract_logprobs(token_logprobs)
        if not logprobs:
            logprobs = [0.0] * len(token_ids)
        finish_reason = ConfiguredGatewayRuntimeController._extract_finish_reason(
            meta_info
        )
        rollout_model_version = (
            ConfiguredGatewayRuntimeController._extract_policy_version(
                meta_info,
                fallback=fallback_policy_version,
            )
        )
        return GatewayGeneration(
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            rollout_model_version=rollout_model_version,
        )

    @staticmethod
    def _extract_token_ids(
        payload: dict[str, Any],
        *,
        token_logprobs: Sequence[Any],
    ) -> list[int]:
        for source in (payload, payload.get("meta_info", {})):
            if not isinstance(source, dict):
                continue
            for key in ("output_ids", "token_ids"):
                token_ids = source.get(key)
                if isinstance(token_ids, list):
                    return [int(token_id) for token_id in token_ids]

        extracted_ids: list[int] = []
        for item in token_logprobs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                extracted_ids.append(int(item[1]))
                continue
            if isinstance(item, dict) and "token_id" in item:
                extracted_ids.append(int(item["token_id"]))
        return extracted_ids

    @staticmethod
    def _extract_logprobs(token_logprobs: Sequence[Any]) -> list[float]:
        extracted: list[float] = []
        for item in token_logprobs:
            if isinstance(item, (list, tuple)) and item:
                extracted.append(float(item[0] or 0.0))
                continue
            if isinstance(item, dict):
                value = item.get("logprob", item.get("token_logprob", 0.0))
                extracted.append(float(value or 0.0))
        return extracted

    @staticmethod
    def _extract_finish_reason(meta_info: dict[str, Any]) -> str:
        finish_reason = meta_info.get("finish_reason", "stop")
        if isinstance(finish_reason, str):
            return finish_reason
        if isinstance(finish_reason, dict):
            return str(finish_reason.get("type", "stop"))
        return "stop"

    @staticmethod
    def _extract_policy_version(meta_info: dict[str, Any], *, fallback: int) -> int:
        weight_version = meta_info.get("weight_version")
        if weight_version is not None and str(weight_version).isdigit():
            return int(weight_version)
        token_steps = meta_info.get("token_steps")
        if isinstance(token_steps, list) and token_steps:
            if isinstance(token_steps[0], list):
                flattened = [int(step) for group in token_steps for step in group]
                if flattened:
                    return flattened[-1]
            return int(token_steps[-1])
        token_step = meta_info.get("token_step")
        if isinstance(token_step, int):
            return token_step
        return fallback

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer


class GatewayService:
    """Owns the session, trajectory, and turn-recording workflow."""

    def __init__(
        self,
        *,
        store: SQLiteOpenForgeStore,
        controller: ConfiguredGatewayRuntimeController,
    ) -> None:
        self.store = store
        self.controller = controller
        self._active_session_ids: set[str] = set()

    async def list_models(self) -> dict[str, object]:
        return {
            "models": self.controller.list_models(),
            "active_model": self.controller.current_model(),
        }

    async def start_session(self, model_name: str) -> StartSessionResult:
        if self._active_session_ids:
            raise ActiveSessionError("another session is already active")
        if self.controller.current_model() not in (None, model_name):
            await self._shutdown_controller()
        self.controller.ensure_model(model_name)

        session_id = self._new_id("sess")
        await self.store.create_session(
            Session(session_id=session_id, model_name=model_name)
        )
        self._active_session_ids.add(session_id)
        return StartSessionResult(session_id=session_id, model=model_name)

    async def start_trajectory(
        self,
        *,
        session_id: str,
        parent_trajectory_id: str | None = None,
    ) -> StartTrajectoryResult:
        await self._require_active_session(session_id)

        trajectory_id = self._new_id("traj")
        parent_turns: list[Turn] = []
        if parent_trajectory_id is not None:
            parent = await self._require_active_trajectory(
                session_id=session_id,
                trajectory_id=parent_trajectory_id,
            )
            parent_turns = await self.store.list_turns(parent.trajectory_id)

        await self.store.create_trajectory(
            Trajectory(
                trajectory_id=trajectory_id,
                session_id=session_id,
                parent_trajectory_id=parent_trajectory_id,
                status="active",
            )
        )

        for turn in parent_turns:
            await self.store.append_turn(
                Turn(
                    trajectory_id=trajectory_id,
                    turn_index=turn.turn_index,
                    rollout_model_version=turn.rollout_model_version,
                    prompt_length=turn.prompt_length,
                    input_ids=list(turn.input_ids),
                    position_ids=list(turn.position_ids),
                    loss_mask=list(turn.loss_mask),
                    old_logprobs=list(turn.old_logprobs),
                )
            )

        return StartTrajectoryResult(
            session_id=session_id,
            trajectory_id=trajectory_id,
            parent_trajectory_id=parent_trajectory_id,
        )

    async def generate(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any] | None = None,
    ) -> GenerateResult:
        session = await self._require_active_session(session_id)
        await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        self.controller.ensure_model(session.model_name)
        prompt_token_ids = self.controller.tokenize_messages(
            session.model_name, messages
        )
        turn_index = len(await self.store.list_turns(trajectory_id))
        generation = self.controller.generate(
            session.model_name,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        await self.store.append_turn(
            self._build_turn(
                trajectory_id=trajectory_id,
                turn_index=turn_index,
                prompt_token_ids=prompt_token_ids,
                generation=generation,
            )
        )
        return GenerateResult(
            session_id=session_id,
            trajectory_id=trajectory_id,
            token_ids=list(generation.token_ids),
            logprobs=list(generation.logprobs),
            finish_reason=generation.finish_reason,
            rollout_model_version=generation.rollout_model_version,
        )

    async def get_policy_version(
        self,
        *,
        session_id: str,
    ) -> dict[str, int | str]:
        session = await self._require_active_session(session_id)
        return {
            "session_id": session_id,
            "policy_version": self.controller.get_policy_version(session.model_name),
        }

    async def end_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        final_reward: float,
    ) -> dict[str, str]:
        trajectory = await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        await self.store.update_trajectory(
            Trajectory(
                trajectory_id=trajectory.trajectory_id,
                session_id=trajectory.session_id,
                parent_trajectory_id=trajectory.parent_trajectory_id,
                status="completed",
                final_reward=float(final_reward),
            )
        )
        return {
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "status": "completed",
        }

    async def end_session(
        self,
        *,
        session_id: str,
    ) -> dict[str, str]:
        await self._require_active_session(session_id)
        active_trajectories = await self.store.list_trajectories(
            session_id,
            status="active",
        )
        if active_trajectories:
            raise ActiveTrajectoriesRemainError(
                "all trajectories must be ended before ending the session"
            )

        self._active_session_ids.discard(session_id)
        if not self._active_session_ids:
            await self._shutdown_controller()

        return {"session_id": session_id, "status": "completed"}

    async def _require_active_session(self, session_id: str) -> Session:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")
        if session_id not in self._active_session_ids:
            raise SessionClosedError(f"session {session_id} is not active")
        return session

    async def _require_active_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
    ) -> Trajectory:
        trajectory = await self.store.get_trajectory(trajectory_id)
        if trajectory is None or trajectory.session_id != session_id:
            raise TrajectoryNotFoundError(f"unknown trajectory_id: {trajectory_id}")
        if trajectory.status != "active":
            raise TrajectoryNotActiveError(f"trajectory {trajectory_id} is not active")
        return trajectory

    async def _shutdown_controller(self) -> None:
        shutdown = getattr(self.controller, "shutdown", None)
        if shutdown is None:
            return
        await asyncio.to_thread(shutdown)

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _build_turn(
        *,
        trajectory_id: str,
        turn_index: int,
        prompt_token_ids: list[int],
        generation: GatewayGeneration,
    ) -> Turn:
        prompt_length = len(prompt_token_ids)
        completion_length = len(generation.token_ids)
        prompt_prediction_count = max(prompt_length - 1, 0)
        return Turn(
            trajectory_id=trajectory_id,
            turn_index=turn_index,
            rollout_model_version=generation.rollout_model_version,
            prompt_length=prompt_length,
            input_ids=[*prompt_token_ids, *generation.token_ids],
            position_ids=list(range(prompt_length + completion_length)),
            loss_mask=[False] * prompt_prediction_count + [True] * completion_length,
            old_logprobs=[0.0] * prompt_prediction_count + list(generation.logprobs),
        )


def _build_store(cfg: OpenForgeConfig) -> SQLiteOpenForgeStore:
    if cfg.data.path is None:
        return SQLiteOpenForgeStore(":memory:")
    return SQLiteOpenForgeStore(Path(cfg.data.path))


def create_app(
    config: OpenForgeConfig | GatewayConfig,
    *,
    store: SQLiteOpenForgeStore | None = None,
    controller: ConfiguredGatewayRuntimeController | None = None,
) -> FastAPI:
    """Create the OpenForge gateway app."""
    runtime_enabled = isinstance(config, OpenForgeConfig)
    if isinstance(config, OpenForgeConfig):
        gateway_config = config.gateway
        store = store or _build_store(config)
        controller = controller or ConfiguredGatewayRuntimeController(cfg=config)
    else:
        gateway_config = config
        if store is None or controller is None:
            raise ValueError(
                "store and controller must be provided when create_app receives GatewayConfig"
            )

    service = GatewayService(store=store, controller=controller)
    app = FastAPI(title="OpenForge Gateway")
    app.state.config = gateway_config
    app.state.controller = controller
    app.state.store = store
    app.state.service = service

    if runtime_enabled:

        @app.on_event("shutdown")
        async def shutdown_runtime() -> None:
            await asyncio.to_thread(controller.shutdown)
            await store.close()

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        return ModelsResponse.model_validate(await service.list_models())

    @app.post("/start_session", response_model=StartSessionResponse)
    async def start_session(payload: StartSessionRequest) -> StartSessionResponse:
        try:
            result = await service.start_session(payload.model)
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ActiveSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StartSessionResponse(session_id=result.session_id, model=result.model)

    @app.post("/start_trajectory", response_model=StartTrajectoryResponse)
    async def start_trajectory(
        payload: StartTrajectoryRequest,
    ) -> StartTrajectoryResponse:
        try:
            result = await service.start_trajectory(
                session_id=payload.session_id,
                parent_trajectory_id=payload.parent_trajectory_id,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StartTrajectoryResponse(
            session_id=result.session_id,
            trajectory_id=result.trajectory_id,
            parent_trajectory_id=result.parent_trajectory_id,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            result = await service.generate(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                messages=[message.model_dump() for message in payload.messages],
                sampling_params=payload.sampling_params,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return GenerateResponse(
            session_id=result.session_id,
            trajectory_id=result.trajectory_id,
            token_ids=result.token_ids,
            logprobs=result.logprobs,
            finish_reason=result.finish_reason,
            rollout_model_version=result.rollout_model_version,
        )

    @app.post("/get_policy_version", response_model=GetPolicyVersionResponse)
    async def get_policy_version(
        payload: GetPolicyVersionRequest,
    ) -> GetPolicyVersionResponse:
        try:
            result = await service.get_policy_version(session_id=payload.session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return GetPolicyVersionResponse(**result)

    @app.post("/end_trajectory", response_model=EndTrajectoryResponse)
    async def end_trajectory(
        payload: EndTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        try:
            result = await service.end_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                final_reward=payload.final_reward,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return EndTrajectoryResponse(**result)

    @app.post("/end_session", response_model=EndSessionResponse)
    async def end_session(payload: EndSessionRequest) -> EndSessionResponse:
        try:
            result = await service.end_session(session_id=payload.session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ActiveTrajectoriesRemainError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return EndSessionResponse(**result)

    return app
