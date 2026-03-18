# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from openforge.configs.models import GatewayConfig, OpenForgeConfig
from openforge.data import SQLiteOpenForgeStore, Session, Trajectory, Turn
from openforge.gateway.types import (
    CreateSessionRequest,
    CreateSessionResponse,
    CreateSessionResult,
    EndSessionRequest,
    EndSessionResponse,
    GenerateChoice,
    GenerateChoiceResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateResult,
    RolloutReward,
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


class SessionNotFoundError(RuntimeError):
    """Raised when a session id does not exist."""


class NoActiveTrajectoriesError(RuntimeError):
    """Raised when a session has no active trajectories to continue."""


class InvalidRolloutRewardError(RuntimeError):
    """Raised when end_session reward input does not match the active frontier."""


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


class ConfiguredGatewayRuntimeController:
    """Single-model placeholder runtime controller for gateway development."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer_name: str,
    ) -> None:
        self._supported_model = model_name
        self._tokenizer_name = tokenizer_name
        self._loaded_model: str | None = None
        self._tokenizer = None
        self._rollout_model_version = 1

    def list_models(self) -> list[str]:
        return [self._supported_model]

    def current_model(self) -> str | None:
        return self._loaded_model

    def ensure_model(self, model_name: str) -> None:
        if model_name != self._supported_model:
            raise UnsupportedModelError(f"unsupported model: {model_name}")
        if self._loaded_model is None:
            self._loaded_model = model_name
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
        n: int,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[GatewayGeneration]:
        self.ensure_model(model_name)
        del sampling_params
        tail = [int(token_id) for token_id in prompt_token_ids[-min(4, len(prompt_token_ids)) :]]
        if not tail:
            tail = [0]

        generations: list[GatewayGeneration] = []
        for choice_index in range(n):
            shift = choice_index % len(tail)
            rotated = tail[shift:] + tail[:shift]
            token_ids = rotated[: min(3, len(rotated))]
            if not token_ids:
                token_ids = [tail[0]]
            generations.append(
                GatewayGeneration(
                    token_ids=token_ids,
                    logprobs=[-0.1 * (index + 1) for index in range(len(token_ids))],
                    finish_reason="stop",
                    rollout_model_version=self._rollout_model_version,
                )
            )
        return generations

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

    async def list_models(self) -> dict[str, object]:
        return {
            "models": self.controller.list_models(),
            "current_model": self.controller.current_model(),
        }

    async def create_session(self, model_name: str) -> CreateSessionResult:
        self.controller.ensure_model(model_name)

        session_id = self._new_id("sess")
        trajectory_id = self._new_id("traj")
        await self.store.create_session(
            Session(session_id=session_id, model_name=model_name)
        )
        await self.store.create_trajectory(
            Trajectory(
                trajectory_id=trajectory_id,
                session_id=session_id,
                parent_trajectory_id=None,
                status="active",
            )
        )
        return CreateSessionResult(session_id=session_id, model=model_name)

    async def generate(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        n: int,
        sampling_params: dict[str, Any] | None = None,
    ) -> GenerateResult:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")

        self.controller.ensure_model(session.model_name)
        prompt_token_ids = self.controller.tokenize_messages(session.model_name, messages)

        active_trajectories = await self.store.list_trajectories(
            session_id,
            status="active",
        )
        if not active_trajectories:
            raise NoActiveTrajectoriesError(
                f"session {session_id} has no active trajectories"
            )

        choices: list[GenerateChoice] = []
        for trajectory in active_trajectories:
            parent_turns = await self.store.list_turns(trajectory.trajectory_id)
            parent_turn_count = len(parent_turns)
            generations = self.controller.generate(
                session.model_name,
                prompt_token_ids=prompt_token_ids,
                n=n,
                sampling_params=sampling_params,
            )

            if n == 1:
                generation = generations[0]
                await self.store.append_turn(
                    self._build_turn(
                        trajectory_id=trajectory.trajectory_id,
                        turn_index=parent_turn_count,
                        prompt_token_ids=prompt_token_ids,
                        generation=generation,
                    )
                )
                choices.append(
                    GenerateChoice(
                        rollout_id=trajectory.trajectory_id,
                        token_ids=list(generation.token_ids),
                        logprobs=list(generation.logprobs),
                        finish_reason=generation.finish_reason,
                        rollout_model_version=generation.rollout_model_version,
                    )
                )
                continue

            await self.store.update_trajectory(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    parent_trajectory_id=trajectory.parent_trajectory_id,
                    status="forked",
                    final_reward=trajectory.final_reward,
                )
            )
            for generation in generations:
                child_trajectory_id = self._new_id("traj")
                await self.store.create_trajectory(
                    Trajectory(
                        trajectory_id=child_trajectory_id,
                        session_id=session.session_id,
                        parent_trajectory_id=trajectory.trajectory_id,
                        status="active",
                    )
                )
                for turn in parent_turns:
                    await self.store.append_turn(
                        Turn(
                            trajectory_id=child_trajectory_id,
                            turn_index=turn.turn_index,
                            rollout_model_version=turn.rollout_model_version,
                            prompt_length=turn.prompt_length,
                            input_ids=list(turn.input_ids),
                            position_ids=list(turn.position_ids),
                            loss_mask=list(turn.loss_mask),
                            old_logprobs=list(turn.old_logprobs),
                        )
                    )

                await self.store.append_turn(
                    self._build_turn(
                        trajectory_id=child_trajectory_id,
                        turn_index=parent_turn_count,
                        prompt_token_ids=prompt_token_ids,
                        generation=generation,
                    )
                )
                choices.append(
                    GenerateChoice(
                        rollout_id=child_trajectory_id,
                        token_ids=list(generation.token_ids),
                        logprobs=list(generation.logprobs),
                        finish_reason=generation.finish_reason,
                        rollout_model_version=generation.rollout_model_version,
                    )
                )

        return GenerateResult(session_id=session_id, choices=choices)

    async def end_session(
        self,
        *,
        session_id: str,
        rewards: list[RolloutReward],
    ) -> dict[str, str]:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")

        active_trajectories = await self.store.list_trajectories(
            session_id,
            status="active",
        )
        if not active_trajectories:
            raise NoActiveTrajectoriesError(
                f"session {session_id} has no active trajectories"
            )

        reward_by_rollout = {reward.rollout_id: reward.reward for reward in rewards}
        active_rollout_ids = {
            trajectory.trajectory_id for trajectory in active_trajectories
        }
        if set(reward_by_rollout) != active_rollout_ids:
            raise InvalidRolloutRewardError(
                "rewards must be provided for exactly the active rollout ids"
            )

        for trajectory in active_trajectories:
            await self.store.update_trajectory(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    parent_trajectory_id=trajectory.parent_trajectory_id,
                    status="completed",
                    final_reward=float(reward_by_rollout[trajectory.trajectory_id]),
                )
            )

        return {"session_id": session_id, "status": "completed"}

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
    if isinstance(config, OpenForgeConfig):
        gateway_config = config.gateway
        store = store or _build_store(config)
        controller = controller or ConfiguredGatewayRuntimeController(
            model_name=config.model.model_name_or_path,
            tokenizer_name=config.model.tokenizer_name_or_path,
        )
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

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/models")
    async def list_models() -> dict[str, object]:
        return await service.list_models()

    @app.post("/create_session", response_model=CreateSessionResponse)
    async def create_session(payload: CreateSessionRequest) -> CreateSessionResponse:
        try:
            result = await service.create_session(payload.model)
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CreateSessionResponse(session_id=result.session_id, model=result.model)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            result = await service.generate(
                session_id=payload.session_id,
                messages=[message.model_dump() for message in payload.messages],
                n=payload.n,
                sampling_params=payload.sampling_params,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NoActiveTrajectoriesError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return GenerateResponse(
            session_id=result.session_id,
            choices=[
                GenerateChoiceResponse(
                    rollout_id=choice.rollout_id,
                    token_ids=choice.token_ids,
                    logprobs=choice.logprobs,
                    finish_reason=choice.finish_reason,
                    rollout_model_version=choice.rollout_model_version,
                )
                for choice in result.choices
            ],
        )

    @app.post("/end_session", response_model=EndSessionResponse)
    async def end_session(payload: EndSessionRequest) -> EndSessionResponse:
        try:
            result = await service.end_session(
                session_id=payload.session_id,
                rewards=[
                    RolloutReward(
                        rollout_id=reward.rollout_id,
                        reward=reward.reward,
                    )
                    for reward in payload.rewards
                ],
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NoActiveTrajectoriesError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidRolloutRewardError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return EndSessionResponse(**result)

    return app
