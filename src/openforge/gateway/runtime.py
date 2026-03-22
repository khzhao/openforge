# Copyright 2026 openforge

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from openforge.configs.models import GatewayServerConfig, OpenForgeConfig
from openforge.gateway.types import ChatMessage, RuntimeConfig

__all__ = [
    "Generation",
    "Runtime",
    "RuntimeSlot",
]


SUPPORTED_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

SUPPORTED_TOKENIZERS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


@dataclass(slots=True)
class Generation:
    """One generated token sequence returned by the runtime."""

    text: str
    token_ids: list[int]
    rollout_model_version: str
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if not self.rollout_model_version:
            raise ValueError("rollout_model_version must be non-empty")


@dataclass(slots=True)
class RuntimeSlot:
    """Active train+rollout runtime resources for the loaded model."""

    train_manager: Any
    rollout_manager: Any

    def shutdown(self) -> None:
        try:
            self.rollout_manager.shutdown()
        finally:
            self.train_manager.shutdown()


class Runtime:
    """Single-model runtime owner for the gateway."""

    def __init__(self, *, cfg: GatewayServerConfig) -> None:
        self._cfg = cfg
        self._loaded_model: str | None = None
        self._tokenizer_name: str | None = None
        self._tokenizer = None
        self._runtime_cfg: OpenForgeConfig | None = None
        self._slot: RuntimeSlot | None = None

    def list_models(self) -> list[dict[str, str]]:
        return [
            {"id": model_id, "tokenizer": tokenizer}
            for model_id, tokenizer in zip(
                SUPPORTED_MODELS,
                SUPPORTED_TOKENIZERS,
                strict=True,
            )
        ]

    def current_model(self) -> str | None:
        return self._loaded_model

    def start(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> str:
        model_name = runtime_config.model.model_name_or_path
        if not self._is_supported_model(model_name):
            raise Exception(f"unsupported model: {model_name}")

        runtime_cfg = self._build_config(runtime_config=runtime_config)
        if self._loaded_model is None:
            slot = self._start_slot(runtime_cfg)
            self._runtime_cfg = runtime_cfg
            self._loaded_model = runtime_cfg.model.model_name_or_path
            self._tokenizer_name = runtime_cfg.model.tokenizer_name_or_path
            self._tokenizer = None
            self._slot = slot
            return self._loaded_model

        if self._loaded_model != runtime_cfg.model.model_name_or_path:
            raise Exception(
                f"gateway already has active model {self._loaded_model!r}; "
                f"cannot switch to {runtime_cfg.model.model_name_or_path!r}"
            )
        return self._loaded_model

    def tokenize_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[int]:
        tokenizer = self._get_tokenizer()
        try:
            token_ids = tokenizer.apply_chat_template(
                [message.model_dump(mode="json") for message in messages],
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise Exception(
                f"failed to tokenize messages with chat template: {exc}"
            ) from exc
        return [int(token_id) for token_id in token_ids]

    def tokenize_messages_batch(
        self,
        message_batches: Sequence[Sequence[ChatMessage]],
    ) -> list[list[int]]:
        if not message_batches:
            return []
        tokenizer = self._get_tokenizer()
        conversations = [
            [message.model_dump(mode="json") for message in messages]
            for messages in message_batches
        ]
        try:
            token_batches = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise Exception(
                f"failed to tokenize messages with chat template: {exc}"
            ) from exc
        return [
            [int(token_id) for token_id in token_ids] for token_ids in token_batches
        ]

    def generate(
        self,
        *,
        input_ids: Sequence[int],
        sampling_params: dict[str, Any] | None = None,
    ) -> Generation:
        payload = self._slot.rollout_manager.router.generate(
            self._build_sampling_params(sampling_params),
            return_logprob=False,
            input_ids=[int(token_id) for token_id in input_ids],
        )
        return Generation(**self._parse_generation_info(payload))

    def generate_batch(
        self,
        *,
        input_ids: Sequence[Sequence[int]],
        sampling_params: dict[str, Any] | None = None,
    ) -> list[Generation]:
        requests = [[int(token_id) for token_id in ids] for ids in input_ids]
        if not requests:
            return []

        sampling_payload = self._build_sampling_params(sampling_params)
        payload = self._slot.rollout_manager.router.generate(
            sampling_payload,
            return_logprob=False,
            input_ids=requests,
        )
        if isinstance(payload, dict):
            payloads = [payload]
        else:
            payloads = list(payload)
        if len(payloads) != len(requests):
            raise ValueError(
                "batched generate payload size mismatch: "
                f"{len(payloads)} != {len(requests)}"
            )
        return [Generation(**self._parse_generation_info(item)) for item in payloads]

    def train_manager(self):
        slot = self._slot
        assert slot is not None
        return slot.train_manager

    def shutdown(self) -> None:
        import ray

        from openforge.rollout.sglang.utils import stop_spawn_resource_tracker

        slot = self._slot
        self._slot = None
        self._runtime_cfg = None
        self._loaded_model = None
        self._tokenizer_name = None
        self._tokenizer = None
        try:
            if slot is not None:
                slot.shutdown()
        finally:
            if ray.is_initialized():
                ray.shutdown()
            stop_spawn_resource_tracker()

    def _start_slot(self, cfg: OpenForgeConfig) -> RuntimeSlot:
        import ray

        from openforge.runtime import (
            create_rollout_manager,
            create_train_manager,
            register_rollout,
        )
        from openforge.utils.networking import get_free_port, get_host_ip
        from openforge.utils.ray import create_placement_groups

        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

        started_ray = False
        if not ray.is_initialized():
            ray.init(
                address=os.environ.get("RAY_ADDRESS", "local"),
                log_to_driver=False,
            )
            started_ray = True

        placement_groups = create_placement_groups(cfg)
        train_manager = None
        rollout_manager = None
        try:
            train_manager = create_train_manager(
                cfg,
                master_addr=get_host_ip(),
                master_port=get_free_port(start=20000),
                placement_groups=placement_groups,
            )
            rollout_manager = create_rollout_manager(
                cfg,
                placement_groups,
                log_level="warn",
            )
            register_rollout(train_manager, rollout_manager)
        except Exception:
            try:
                if rollout_manager is not None:
                    rollout_manager.shutdown()
                if train_manager is not None:
                    train_manager.shutdown()
                else:
                    ray.util.remove_placement_group(placement_groups["actor"][0])
            finally:
                if started_ray and ray.is_initialized():
                    ray.shutdown()
            raise

        return RuntimeSlot(
            train_manager=train_manager,
            rollout_manager=rollout_manager,
        )

    def _build_config(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> OpenForgeConfig:
        return OpenForgeConfig(
            data=self._cfg.data,
            gateway=self._cfg.gateway,
            cluster=self._cfg.cluster,
            algo=runtime_config.algo,
            model=runtime_config.model,
            train=runtime_config.train,
            rollout=runtime_config.rollout,
        )

    def _build_sampling_params(
        self,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = self._runtime_cfg.rollout.request.model_dump()
        if overrides:
            payload.update(overrides)
        return payload

    @staticmethod
    def _parse_generation_info(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        meta_info = payload.get("meta_info", {})
        if not isinstance(meta_info, dict):
            meta_info = {}

        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError("generate payload missing text")

        token_ids: list[int] | None = None
        for source in (payload, meta_info):
            for key in ("output_ids", "token_ids"):
                value = source.get(key)
                if isinstance(value, list):
                    token_ids = [int(token_id) for token_id in value]
                    break
            if token_ids is not None:
                break
        if token_ids is None:
            raise ValueError("generate payload missing output_ids or token_ids")

        finish_reason = meta_info.get("finish_reason", "stop")
        if isinstance(finish_reason, str):
            finish_reason_text = finish_reason
        elif isinstance(finish_reason, dict):
            finish_reason_text = str(finish_reason.get("type", "stop"))
        else:
            finish_reason_text = "stop"

        weight_version = meta_info.get("weight_version")
        if weight_version is None:
            raise ValueError("generate payload missing meta_info.weight_version")
        rollout_model_version = str(weight_version)
        if not rollout_model_version:
            raise ValueError("generate payload has empty meta_info.weight_version")

        return {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": finish_reason_text,
            "rollout_model_version": rollout_model_version,
        }

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer

    @staticmethod
    def _is_supported_model(model_name: str) -> bool:
        if model_name in SUPPORTED_MODELS:
            return True
        return Path(model_name).exists()
