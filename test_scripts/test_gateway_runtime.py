# Copyright 2026 openforge

from __future__ import annotations

import pytest
import ray

import openforge.runtime as runtime_api
import openforge.utils.networking as networking_utils
import openforge.utils.ray as ray_utils
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.runtime import Runtime
from openforge.gateway.types import RuntimeConfig, StartSessionRequest


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=2, cpus_per_node=8),
    )


def _runtime_config(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> RuntimeConfig:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"kl_coef": 0.0},
                "model": {
                    "model_name_or_path": model_name,
                    "tokenizer_name_or_path": model_name,
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
            }
        }
    )
    return request.runtime


def test_runtime_start_rolls_back_state_when_slot_creation_fails(monkeypatch) -> None:
    runtime = Runtime(cfg=_server_config())
    monkeypatch.setattr(
        runtime,
        "_start_slot",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        runtime.start(runtime_config=_runtime_config())

    assert runtime.current_model() is None
    assert runtime._slot is None
    assert runtime._runtime_cfg is None
    assert runtime._tokenizer_name is None


def test_runtime_start_slot_shuts_down_ray_when_startup_fails(monkeypatch) -> None:
    runtime = Runtime(cfg=_server_config())
    state = {"initialized": False, "shutdown_called": False, "removed_pg": False}

    def is_initialized() -> bool:
        return bool(state["initialized"])

    def init(*, log_to_driver: bool) -> None:
        state["initialized"] = True

    def shutdown() -> None:
        state["shutdown_called"] = True
        state["initialized"] = False

    monkeypatch.setattr(ray, "is_initialized", is_initialized)
    monkeypatch.setattr(ray, "init", init)
    monkeypatch.setattr(ray, "shutdown", shutdown)
    monkeypatch.setattr(
        ray.util,
        "remove_placement_group",
        lambda pg: state.__setitem__("removed_pg", True),
    )

    monkeypatch.setattr(
        ray_utils,
        "create_placement_groups",
        lambda cfg: {"actor": ("pg", [0], [0]), "rollout": ("pg", [1], [1])},
    )
    monkeypatch.setattr(networking_utils, "get_host_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(networking_utils, "get_free_port", lambda start=20000: 20000)
    monkeypatch.setattr(
        runtime_api,
        "create_train_manager",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("train boom")),
    )
    monkeypatch.setattr(runtime_api, "create_rollout_manager", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_api, "register_rollout", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="train boom"):
        runtime._start_slot(runtime._build_config(runtime_config=_runtime_config()))

    assert state["removed_pg"] is True
    assert state["shutdown_called"] is True
