# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from openforge.configs.models import GatewayServerConfig, OpenForgeConfig
from openforge.gateway.types import RuntimeConfig

__all__ = [
    "Generation",
    "ModelBusyError",
    "Runtime",
    "RuntimeSlot",
    "UnsupportedModelError",
]


SUPPORTED_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

SUPPORTED_TOKENIZERS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


class UnsupportedModelError(ValueError):
    """Raised when the caller requests a model this gateway does not support."""


class ModelBusyError(RuntimeError):
    """Raised when a different model is already active in the single-model slot."""


@dataclass(slots=True)
class Generation:
    """One generated token sequence returned by the runtime."""

    token_ids: list[int]
    logprobs: list[float]
    rollout_model_version: str
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have the same length")
        if not self.rollout_model_version:
            raise ValueError("rollout_model_version must be non-empty")


@dataclass(slots=True)
class RuntimeSlot:
    """Active train+rollout runtime resources for the loaded model."""

    placement_groups: dict[str, Any]
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
            raise UnsupportedModelError(f"unsupported model: {model_name}")

        runtime_cfg = self._build_config(runtime_config=runtime_config)
        if self._loaded_model is None:
            self._runtime_cfg = runtime_cfg
            self._loaded_model = runtime_cfg.model.model_name_or_path
            self._tokenizer_name = runtime_cfg.model.tokenizer_name_or_path
            self._tokenizer = None
            self._slot = self._start_slot(runtime_cfg)
            return self._loaded_model

        if self._loaded_model != runtime_cfg.model.model_name_or_path:
            raise ModelBusyError(
                f"gateway already has active model {self._loaded_model!r}; "
                f"cannot switch to {runtime_cfg.model.model_name_or_path!r}"
            )
        return self._loaded_model

    def tokenize_messages(
        self,
        model_name: str,
        messages: Sequence[dict[str, str]],
    ) -> list[int]:
        if self._loaded_model != model_name:
            raise ModelBusyError(f"runtime is not loaded for model {model_name!r}")
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
    ) -> Generation:
        if self._loaded_model != model_name:
            raise ModelBusyError(f"runtime is not loaded for model {model_name!r}")
        assert self._slot is not None
        payload = self._slot.rollout_manager.generate(
            self._build_sampling_params(sampling_params),
            input_ids=[int(token_id) for token_id in prompt_token_ids],
            return_logprob=True,
        )
        return self._parse_generation_payload(payload)

    def shutdown(self) -> None:
        slot = self._slot
        self._slot = None
        self._runtime_cfg = None
        self._loaded_model = None
        self._tokenizer_name = None
        self._tokenizer = None
        if slot is not None:
            slot.shutdown()

    def _start_slot(self, cfg: OpenForgeConfig) -> RuntimeSlot:
        import ray

        from openforge.runtime import (
            create_rollout_manager,
            create_train_manager,
            register_rollout,
        )
        from openforge.utils.networking import get_free_port, get_host_ip
        from openforge.utils.ray import create_placement_groups

        if not ray.is_initialized():
            ray.init(log_to_driver=False)

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
            rollout_manager = create_rollout_manager(cfg, placement_groups)
            register_rollout(train_manager, rollout_manager)
        except Exception:
            if rollout_manager is not None:
                rollout_manager.shutdown()
            if train_manager is not None:
                train_manager.shutdown()
            else:
                ray.util.remove_placement_group(placement_groups["actor"][0])
            raise

        return RuntimeSlot(
            placement_groups=placement_groups,
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
        if self._runtime_cfg is None:
            raise RuntimeError("runtime is not loaded")
        payload = self._runtime_cfg.rollout.request.model_dump()
        if overrides:
            payload.update(overrides)
        return payload

    @staticmethod
    def _parse_generation_payload(
        payload: dict[str, Any],
    ) -> Generation:
        meta_info = payload.get("meta_info", {})
        token_logprobs = meta_info.get("output_token_logprobs", [])
        token_ids = Runtime._extract_token_ids(
            payload,
            token_logprobs=token_logprobs,
        )
        logprobs = Runtime._extract_logprobs(token_logprobs)
        if not logprobs:
            logprobs = [0.0] * len(token_ids)
        finish_reason = Runtime._extract_finish_reason(meta_info)
        rollout_model_version = Runtime._extract_rollout_model_version(meta_info)
        return Generation(
            token_ids=token_ids,
            logprobs=logprobs,
            rollout_model_version=rollout_model_version,
            finish_reason=finish_reason,
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
    def _extract_rollout_model_version(
        meta_info: dict[str, Any],
    ) -> str:
        weight_version = meta_info.get("weight_version")
        if weight_version is None:
            raise ValueError("generate payload missing meta_info.weight_version")
        version = str(weight_version)
        if not version:
            raise ValueError("generate payload has empty meta_info.weight_version")
        return version

    def _get_tokenizer(self):
        if self._tokenizer is None:
            assert self._tokenizer_name is not None
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer

    @staticmethod
    def _is_supported_model(model_name: str) -> bool:
        if model_name in SUPPORTED_MODELS:
            return True
        return Path(model_name).exists()
