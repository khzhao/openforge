# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from _script_test_utils import expect_raises, run_tests

import openforge.active_state as active_state
from openforge.gateway.types import RuntimeConfig, StartSessionRequest


def _runtime_config() -> RuntimeConfig:
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "reference_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
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
                            "warmup_steps": 0,
                            "min_lr": 0.0,
                            "num_cycles": 0.5,
                        },
                    },
                    "global_batch_size": 8,
                    "mini_batch_size": 4,
                    "micro_batch_size": 1,
                    "max_rollout_policy_lag": 0,
                    "checkpoints": "./checkpoints",
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
                        "max_new_tokens": 32,
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


def test_active_state_path_uses_openforge_cache_home() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(active_state.os.environ, {"OPENFORGE_CACHE_HOME": tmpdir}):
            assert active_state.active_state_path() == (
                Path(tmpdir) / "openforge" / "active_gateway.json"
            )


def test_active_state_round_trip_gateway_and_session() -> None:
    runtime = _runtime_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        with patch.object(active_state, "active_state_path", lambda: state_path):
            active_state.save_active_gateway(host="127.0.0.1", port=8000, pid=4321)
            assert active_state.load_active_gateway_target() == ("127.0.0.1", 8000)
            assert active_state.load_active_gateway_pid() == 4321

            active_state.save_active_session(session_id="sess_123", runtime=runtime)
            assert (
                active_state.load_active_runtime_config().model_dump(mode="json")
                == runtime.model_dump(mode="json")
            )

            active_state.clear_active_session()
            assert active_state.load_active_state().session is None

            active_state.clear_active_gateway(expected_pid=4321)
            assert state_path.exists() is False


def test_load_active_runtime_config_requires_active_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        with patch.object(active_state, "active_state_path", lambda: state_path):
            active_state.save_active_gateway(host="127.0.0.1", port=8000, pid=4321)
            with expect_raises(
                AssertionError,
                "python -m openforge.cli.main session start",
            ):
                active_state.load_active_runtime_config()


def test_clear_active_gateway_respects_expected_pid() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "active_gateway.json"
        with patch.object(active_state, "active_state_path", lambda: state_path):
            active_state.save_active_gateway(host="127.0.0.1", port=8000, pid=4321)
            active_state.clear_active_gateway(expected_pid=9999)
            assert state_path.exists() is True
            active_state.clear_active_gateway(expected_pid=4321)
            assert state_path.exists() is False


def main() -> int:
    return run_tests(
        [
            test_active_state_path_uses_openforge_cache_home,
            test_active_state_round_trip_gateway_and_session,
            test_load_active_runtime_config_requires_active_session,
            test_clear_active_gateway_respects_expected_pid,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
