# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

from openforge.gateway.types import StartSessionRequest
from openforge.logging import SessionLogger


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.defined: list[tuple[str, str | None]] = []
        self.finished = False

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.defined.append((name, step_metric))

    def finish(self) -> None:
        self.finished = True


def _runtime_config(*, wandb: dict[str, object] | None):
    request = StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
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
                            "warmup_steps": 0,
                            "min_lr": 0.0,
                            "num_cycles": 0.5,
                        },
                    },
                    "global_batch_size": 1,
                    "mini_batch_size": 1,
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
                "wandb": wandb,
            }
        }
    )
    return request.runtime


def test_session_logger_uses_runtime_wandb_config() -> None:
    logger = SessionLogger()
    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(init=lambda **kwargs: fake_run)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        logger.start(
            session_id="sess-12345678",
            runtime_config=_runtime_config(
                wandb={
                    "enabled": True,
                    "project": "proj-a",
                    "entity": "team-a",
                    "name": "run-a",
                    "tags": ["alpha", "beta"],
                    "log_interval_seconds": 3.0,
                }
            ),
        )

    assert logger._run is fake_run
    assert logger._log_interval_seconds == 3.0
    logger.finish()


def test_session_logger_does_not_use_env_to_enable_wandb() -> None:
    logger = SessionLogger()

    with patch.dict("os.environ", {"WANDB_PROJECT": "env-project"}, clear=True):
        logger.start(
            session_id="sess-12345678",
            runtime_config=_runtime_config(wandb=None),
        )

    assert logger._run is None


def test_session_logger_runtime_wandb_requires_project() -> None:
    logger = SessionLogger()

    with expect_raises(
        Exception,
        match="runtime.wandb.project is required",
    ):
        logger.start(
            session_id="sess-12345678",
            runtime_config=_runtime_config(wandb={"enabled": True}),
        )


def test_session_logger_logs_train_metrics_on_wall_time_axis() -> None:
    logger = SessionLogger()
    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(init=lambda **kwargs: fake_run)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        logger.start(
            session_id="sess-12345678",
            runtime_config=_runtime_config(
                wandb={
                    "enabled": True,
                    "project": "proj-a",
                    "log_interval_seconds": 0.0,
                }
            ),
        )

    logger.record_train_update(
        {
            "policy_version": 7,
            "global_step": 3,
            "reward_mean": 1.25,
            "grad_norm": 2.5,
            "lr": 1.0e-5,
        }
    )
    logger.flush(force=True)

    assert any(
        payload.get("status/train_policy_version") == 7.0
        and payload.get("status/train_global_step") == 3.0
        and payload.get("status/train_reward_mean") == 1.25
        and payload.get("status/train_grad_norm") == 2.5
        and payload.get("status/train_lr") == 1.0e-5
        and "wall_time_s" in payload
        for payload in fake_run.logged
    )
    logger.finish()


def main() -> int:
    return run_tests(
        [
            test_session_logger_uses_runtime_wandb_config,
            test_session_logger_does_not_use_env_to_enable_wandb,
            test_session_logger_runtime_wandb_requires_project,
            test_session_logger_logs_train_metrics_on_wall_time_axis,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
