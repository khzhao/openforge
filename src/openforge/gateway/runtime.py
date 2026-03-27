# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Sequence

from openforge.configs.models import GatewayServerConfig, OpenForgeConfig
from openforge.gateway.types import (
    ChatCompletionMessage,
    ChatCompletionTool,
    RuntimeConfig,
    chat_message_payload,
    tool_payloads,
)
from openforge.utils.models import SUPPORTED_MODELS, validate_supported_model

__all__ = [
    "Generation",
    "Runtime",
    "RuntimeSlot",
]


@dataclass(slots=True)
class Generation:
    """One generated token sequence returned by the runtime."""

    text: str
    token_ids: list[int]
    rollout_model_version: int
    rollout_log_probs: list[float]
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        if self.rollout_model_version < 0:
            raise ValueError("rollout_model_version must be >= 0")
        if len(self.rollout_log_probs) != len(self.token_ids):
            raise ValueError(
                "rollout_log_probs must align with token_ids: "
                f"{len(self.rollout_log_probs)} != {len(self.token_ids)}"
            )


@dataclass(slots=True)
class RuntimeSlot:
    """Active train+rollout runtime resources for the loaded model."""

    train_runtime: Any
    rollout_manager: Any

    async def shutdown(self) -> None:
        try:
            self.rollout_manager.shutdown()
        finally:
            await self.train_runtime.shutdown()


class Runtime:
    """Single-model runtime owner for the gateway."""

    def __init__(self, *, cfg: GatewayServerConfig) -> None:
        self._cfg = cfg
        self._loaded_model: str | None = None
        self._tokenizer_name: str | None = None
        self._tokenizer = None
        self._runtime_cfg: OpenForgeConfig | None = None
        self._slot: RuntimeSlot | None = None

    def list_models(self) -> list[str]:
        return list(SUPPORTED_MODELS)

    def current_model(self) -> str | None:
        return self._loaded_model

    def start(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> str:
        model_name = runtime_config.model.model_name_or_path
        validate_supported_model(model_name)

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
        messages: Sequence[ChatCompletionMessage],
        *,
        tools: Sequence[ChatCompletionTool] | None = None,
    ) -> list[int]:
        tokenizer = self._get_tokenizer()
        try:
            token_ids = tokenizer.apply_chat_template(
                [chat_message_payload(message) for message in messages],
                tokenize=True,
                add_generation_prompt=True,
                tools=tool_payloads(list(tools) if tools is not None else None),
            )
        except Exception as exc:
            raise Exception(
                f"failed to tokenize messages with chat template: {exc}"
            ) from exc
        return [int(token_id) for token_id in token_ids]

    def tokenize_messages_batch(
        self,
        message_batches: Sequence[Sequence[ChatCompletionMessage]],
        *,
        tools: Sequence[ChatCompletionTool] | None = None,
    ) -> list[list[int]]:
        if not message_batches:
            return []
        tokenizer = self._get_tokenizer()
        conversations = [
            [chat_message_payload(message) for message in messages]
            for messages in message_batches
        ]
        try:
            token_batches = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                tools=tool_payloads(list(tools) if tools is not None else None),
            )
        except Exception as exc:
            raise Exception(
                f"failed to tokenize messages with chat template: {exc}"
            ) from exc
        return [
            [int(token_id) for token_id in token_ids] for token_ids in token_batches
        ]

    def generate_batch(
        self,
        *,
        trajectory_ids: Sequence[str] | None = None,
        input_ids: Sequence[Sequence[int]],
        sampling_params: dict[str, Any] | None = None,
    ) -> list[Generation]:
        requests = [[int(token_id) for token_id in ids] for ids in input_ids]
        if not requests:
            return []
        resolved_trajectory_ids = None
        if trajectory_ids is not None:
            resolved_trajectory_ids = [
                str(trajectory_id) for trajectory_id in trajectory_ids
            ]
            if len(resolved_trajectory_ids) != len(requests):
                raise ValueError("trajectory_ids must align with input_ids")

        sampling_payload = self._build_sampling_params(sampling_params)
        payload = self._slot.rollout_manager.router.generate(
            sampling_payload,
            return_logprob=True,
            trajectory_ids=resolved_trajectory_ids,
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

    def train(self):
        slot = self._slot
        assert slot is not None
        return slot.train_runtime

    def status(self) -> dict[str, Any]:
        slot = self._slot
        if slot is None:
            return {
                "current_model": self._loaded_model,
                "train": {},
                "rollout": {},
                "cluster": {},
            }
        return {
            "current_model": self._loaded_model,
            "train": slot.train_runtime.status(),
            "rollout": slot.rollout_manager.status(),
            "cluster": self._cluster_status(),
        }

    def release_trajectories(self, trajectory_ids: Sequence[str]) -> None:
        if not trajectory_ids:
            return
        slot = self._slot
        assert slot is not None
        slot.rollout_manager.router.release_trajectories(
            trajectory_ids=[str(trajectory_id) for trajectory_id in trajectory_ids]
        )

    async def shutdown(self) -> None:
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
                await slot.shutdown()
        finally:
            if ray.is_initialized():
                ray.shutdown()
            stop_spawn_resource_tracker()

    def _start_slot(self, cfg: OpenForgeConfig) -> RuntimeSlot:
        import ray

        from openforge.runtime import (
            create_rollout_manager,
            create_train_runtime,
        )
        from openforge.utils.networking import get_free_port, get_host_ip
        from openforge.utils.ray import create_placement_groups

        started_ray = False
        if not ray.is_initialized():
            ray.init(
                address=os.environ.get("RAY_ADDRESS", "local"),
                num_cpus=self._cfg.cluster.total_cpus,
                num_gpus=self._cfg.cluster.total_gpus,
                log_to_driver=False,
            )
            started_ray = True

        placement_groups = create_placement_groups(cfg)
        train_runtime = None
        rollout_manager = None
        try:
            train_runtime = create_train_runtime(
                cfg,
                master_addr=get_host_ip(),
                master_port=get_free_port(start=20000),
                placement_groups=placement_groups,
            )
            rollout_manager = create_rollout_manager(
                cfg,
                placement_groups,
                log_level="warning",
            )
            train_runtime.register_rollout(rollout_manager.router_url)
        except Exception:
            try:
                if rollout_manager is not None:
                    rollout_manager.shutdown()
                if train_runtime is not None:
                    asyncio.run(train_runtime.shutdown())
                else:
                    ray.util.remove_placement_group(placement_groups["actor"][0])
            finally:
                if started_ray and ray.is_initialized():
                    ray.shutdown()
            raise

        return RuntimeSlot(
            train_runtime=train_runtime,
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
        rollout_model_version = int(weight_version)

        output_token_logprobs = meta_info.get("output_token_logprobs")
        if not isinstance(output_token_logprobs, list):
            raise ValueError("generate payload missing meta_info.output_token_logprobs")
        rollout_log_probs = [
            float(token_logprob[0]) for token_logprob in output_token_logprobs
        ]
        if len(rollout_log_probs) != len(token_ids):
            raise ValueError(
                "generate payload has misaligned output_token_logprobs: "
                f"{len(rollout_log_probs)} != {len(token_ids)}"
            )

        return {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": finish_reason_text,
            "rollout_model_version": rollout_model_version,
            "rollout_log_probs": rollout_log_probs,
        }

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer

    @staticmethod
    def _cluster_status() -> dict[str, Any]:
        import ray

        if not ray.is_initialized():
            return {}
        nodes = ray.nodes()
        return {
            "node_count": len(nodes),
            "alive_nodes": sum(1 for node in nodes if bool(node.get("Alive"))),
            "total_resources": {
                key: float(value)
                for key, value in ray.cluster_resources().items()
                if isinstance(value, (int, float))
            },
            "available_resources": {
                key: float(value)
                for key, value in ray.available_resources().items()
                if isinstance(value, (int, float))
            },
        }
