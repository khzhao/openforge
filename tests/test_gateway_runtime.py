# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import asyncio
import os
import types
from contextlib import ExitStack
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import ray

import openforge.runtime as runtime_api
import openforge.utils.networking as networking_utils
import openforge.utils.ray as ray_utils
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.runtime import Runtime
from openforge.gateway.types import (
    ChatMessage,
    RuntimeConfig,
    StartSessionRequest,
)
from openforge.utils.models import validate_supported_model


class _FakeTrainRuntime:
    async def shutdown(self) -> None:
        return

    def register_rollout(self, rollout_router_url: str) -> None:
        return


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=2, cpus_per_node=8),
    )


def _runtime_config(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> RuntimeConfig:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": model_name,
                    "tokenizer_name_or_path": model_name,
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
            }
        }
    )
    return request.runtime


def test_runtime_start_rolls_back_state_when_slot_creation_fails() -> None:
    runtime = Runtime(cfg=_server_config())
    with patch.object(
        runtime,
        "_start_slot",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    ):
        with expect_raises(RuntimeError, match="boom"):
            runtime.start(runtime_config=_runtime_config())

        assert runtime.current_model() is None
        assert runtime._slot is None
        assert runtime._runtime_cfg is None


def test_runtime_start_slot_shuts_down_ray_when_startup_fails() -> None:
    runtime = Runtime(cfg=_server_config())
    state = {
        "address": None,
        "initialized": False,
        "shutdown_called": False,
        "removed_pg": False,
    }

    def is_initialized() -> bool:
        return bool(state["initialized"])

    def init(*, address: str, log_to_driver: bool, **kwargs) -> None:
        state["address"] = address
        state["initialized"] = True

    def shutdown() -> None:
        state["shutdown_called"] = True
        state["initialized"] = False

    with ExitStack() as stack:
        stack.enter_context(patch.object(ray, "is_initialized", is_initialized))
        stack.enter_context(patch.object(ray, "init", init))
        stack.enter_context(patch.object(ray, "shutdown", shutdown))
        stack.enter_context(
            patch.object(
                ray.util,
                "remove_placement_group",
                lambda pg: state.__setitem__("removed_pg", True),
            )
        )
        stack.enter_context(
            patch.object(
                ray_utils,
                "create_placement_groups",
                lambda cfg: {"actor": ("pg", [0], [0]), "rollout": ("pg", [1], [1])},
            )
        )
        stack.enter_context(
            patch.object(networking_utils, "get_host_ip", lambda: "127.0.0.1")
        )
        stack.enter_context(
            patch.object(
                networking_utils,
                "get_free_port",
                lambda start=20000: 20000,
            )
        )
        stack.enter_context(
            patch.object(
                runtime_api,
                "create_train_runtime",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("train boom")
                ),
            )
        )
        stack.enter_context(
            patch.object(
                runtime_api,
                "create_rollout_manager",
                lambda *args, **kwargs: None,
            )
        )
        with expect_raises(RuntimeError, match="train boom"):
            runtime._start_slot(runtime._build_config(runtime_config=_runtime_config()))

    assert state["removed_pg"] is True
    assert state["shutdown_called"] is True
    assert state["address"] == "local"


def test_runtime_start_slot_uses_explicit_ray_address() -> None:
    runtime = Runtime(cfg=_server_config())
    state = {"address": None}

    def init(*, address: str, log_to_driver: bool, **kwargs) -> None:
        state["address"] = address

    with ExitStack() as stack:
        stack.enter_context(patch.object(ray, "is_initialized", lambda: False))
        stack.enter_context(patch.object(ray, "init", init))
        stack.enter_context(
            patch.object(
                ray_utils,
                "create_placement_groups",
                lambda cfg: {"actor": ("pg", [0], [0]), "rollout": ("pg", [1], [1])},
            )
        )
        stack.enter_context(
            patch.object(networking_utils, "get_host_ip", lambda: "127.0.0.1")
        )
        stack.enter_context(
            patch.object(
                networking_utils,
                "get_free_port",
                lambda start=20000: 20000,
            )
        )
        stack.enter_context(
            patch.object(
                runtime_api,
                "create_train_runtime",
                lambda *args, **kwargs: _FakeTrainRuntime(),
            )
        )
        stack.enter_context(
            patch.object(
                runtime_api,
                "create_rollout_manager",
                lambda *args, **kwargs: types.SimpleNamespace(
                    shutdown=lambda: None,
                    engine_workers=[],
                    router_url="http://127.0.0.1:31000",
                ),
            )
        )
        stack.enter_context(patch.dict("os.environ", {"RAY_ADDRESS": "ray://cluster"}))
        runtime._start_slot(runtime._build_config(runtime_config=_runtime_config()))

    assert state["address"] == "ray://cluster"


def test_runtime_start_slot_leaves_nccl_defaults_to_train_and_rollout() -> None:
    runtime = Runtime(cfg=_server_config())
    state = {
        "cumem": None,
        "nvls": None,
    }

    def init(*, address: str, log_to_driver: bool, **kwargs) -> None:
        return

    def create_train_runtime(*args, **kwargs):
        state["cumem"] = os.environ.get("NCCL_CUMEM_ENABLE")
        state["nvls"] = os.environ.get("NCCL_NVLS_ENABLE")
        return _FakeTrainRuntime()

    with ExitStack() as stack:
        stack.enter_context(patch.object(ray, "is_initialized", lambda: False))
        stack.enter_context(patch.object(ray, "init", init))
        stack.enter_context(
            patch.object(
                ray_utils,
                "create_placement_groups",
                lambda cfg: {"actor": ("pg", [0], [0]), "rollout": ("pg", [1], [1])},
            )
        )
        stack.enter_context(
            patch.object(networking_utils, "get_host_ip", lambda: "127.0.0.1")
        )
        stack.enter_context(
            patch.object(
                networking_utils,
                "get_free_port",
                lambda start=20000: 20000,
            )
        )
        stack.enter_context(
            patch.object(runtime_api, "create_train_runtime", create_train_runtime)
        )
        stack.enter_context(
            patch.object(
                runtime_api,
                "create_rollout_manager",
                lambda *args, **kwargs: types.SimpleNamespace(
                    shutdown=lambda: None,
                    engine_workers=[],
                    router_url="http://127.0.0.1:31000",
                ),
            )
        )
        stack.enter_context(patch.dict("os.environ", {}, clear=True))
        runtime._start_slot(runtime._build_config(runtime_config=_runtime_config()))

    assert state["cumem"] is None
    assert state["nvls"] is None


def test_runtime_shutdown_also_shuts_down_ray() -> None:
    runtime = Runtime(cfg=_server_config())
    state = {"slot_shutdown_called": False, "ray_shutdown_called": False}

    class FakeSlot:
        async def shutdown(self) -> None:
            state["slot_shutdown_called"] = True

    runtime._slot = FakeSlot()
    runtime._runtime_cfg = runtime._build_config(runtime_config=_runtime_config())

    with ExitStack() as stack:
        stack.enter_context(patch.object(ray, "is_initialized", lambda: True))
        stack.enter_context(
            patch.object(
                ray,
                "shutdown",
                lambda: state.__setitem__("ray_shutdown_called", True),
            )
        )
        asyncio.run(runtime.shutdown())

    assert state["slot_shutdown_called"] is True
    assert state["ray_shutdown_called"] is True
    assert runtime.current_model() is None
    assert runtime._slot is None


def test_runtime_tokenize_messages_propagates_chat_template_failure() -> None:
    runtime = Runtime(cfg=_server_config())
    runtime._runtime_cfg = runtime._build_config(runtime_config=_runtime_config())

    class FakeTokenizer:
        def __init__(self) -> None:
            self.encode_called = False

        def apply_chat_template(self, *args, **kwargs) -> list[int]:
            raise RuntimeError("template boom")

        def encode(self, *args, **kwargs) -> list[int]:
            self.encode_called = True
            raise AssertionError("encode fallback should not be used")

    tokenizer = FakeTokenizer()
    runtime._tokenizer = tokenizer

    with expect_raises(
        Exception,
        match="failed to tokenize messages with chat template: template boom",
    ):
        runtime.tokenize_messages([ChatMessage(role="user", content="hello")])

    assert tokenizer.encode_called is False


def test_runtime_generate_batch_forwards_trajectory_ids() -> None:
    runtime = Runtime(cfg=_server_config())
    runtime._runtime_cfg = runtime._build_config(runtime_config=_runtime_config())

    captured: dict[str, object] = {}

    class FakeRouter:
        def generate(self, sampling_params, **kwargs):
            captured["sampling_params"] = sampling_params
            captured["kwargs"] = kwargs
            return [
                {
                    "text": "reply-1",
                    "output_ids": [11],
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 11, "a"]],
                        "finish_reason": "stop",
                        "weight_version": 3,
                    },
                },
                {
                    "text": "reply-2",
                    "output_ids": [12],
                    "meta_info": {
                        "output_token_logprobs": [[-0.2, 12, "b"]],
                        "finish_reason": "stop",
                        "weight_version": 3,
                    },
                },
            ]

    class FakeRolloutManager:
        def __init__(self) -> None:
            self.router = FakeRouter()

    class FakeSlot:
        def __init__(self) -> None:
            self.rollout_manager = FakeRolloutManager()

    runtime._slot = FakeSlot()

    generations = runtime.generate_batch(
        trajectory_ids=["traj-1", "traj-2"],
        input_ids=[[1], [2]],
    )

    assert [generation.rollout_model_version for generation in generations] == [3, 3]
    assert captured["kwargs"] == {
        "return_logprob": True,
        "trajectory_ids": ["traj-1", "traj-2"],
        "input_ids": [[1], [2]],
    }


def test_runtime_model_config_validation_accepts_full_attention() -> None:
    class FakeConfig:
        use_sliding_window = False
        layer_types = ["full_attention", "full_attention"]

    with patch(
        "transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: FakeConfig()
    ):
        validate_supported_model("Qwen/Qwen3-8B")


def test_runtime_model_config_validation_rejects_sliding_attention() -> None:
    class FakeConfig:
        use_sliding_window = True
        layer_types = ["full_attention", "sliding_attention"]

    with patch(
        "transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: FakeConfig()
    ):
        with expect_raises(
            Exception,
            match="use_sliding_window must be false",
        ):
            validate_supported_model("Qwen/Qwen3-8B")


def test_runtime_model_config_validation_rejects_linear_attention_fields() -> None:
    class FakeConfig:
        use_sliding_window = False
        layer_types = ["full_attention"]
        linear_num_value_heads = 4
        linear_num_key_heads = 2
        linear_key_head_dim = 8
        linear_value_head_dim = 8
        linear_conv_kernel_dim = 4

    with patch(
        "transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: FakeConfig()
    ):
        with expect_raises(
            Exception,
            match="linear-attention fields are not supported",
        ):
            validate_supported_model("Qwen/Qwen3-8B")


def main() -> int:
    return run_tests(
        [
            test_runtime_start_rolls_back_state_when_slot_creation_fails,
            test_runtime_start_slot_shuts_down_ray_when_startup_fails,
            test_runtime_start_slot_uses_explicit_ray_address,
            test_runtime_shutdown_also_shuts_down_ray,
            test_runtime_tokenize_messages_propagates_chat_template_failure,
            test_runtime_generate_batch_forwards_trajectory_ids,
            test_runtime_model_config_validation_accepts_full_attention,
            test_runtime_model_config_validation_rejects_sliding_attention,
            test_runtime_model_config_validation_rejects_linear_attention_fields,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
