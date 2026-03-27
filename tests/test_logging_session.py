# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

from openforge.gateway.types import StartSessionRequest
from openforge.logging import SessionLogger, build_train_update


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
    assert ("train/*", "policy_version") in fake_run.defined
    assert ("rollout/*", "wall_time_s") in fake_run.defined
    assert ("rollout_by_policy/*", "policy_version") not in fake_run.defined
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


def test_build_train_update_normalizes_token_metrics_per_trajectory() -> None:
    payload = build_train_update(
        train_results=[
            [
                SimpleNamespace(metrics={"grad_norm": 2.0}),
                SimpleNamespace(metrics={"grad_norm": 4.0, "lr": 1.0e-5}),
            ]
        ],
        trajectories=[
            SimpleNamespace(trajectory_id="traj-a", final_reward=1.0),
            SimpleNamespace(trajectory_id="traj-b", final_reward=3.0),
        ],
        turns_by_trajectory_id={
            "traj-a": [
                SimpleNamespace(
                    prompt_length=4, token_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                )
            ],
            "traj-b": [
                SimpleNamespace(prompt_length=2, token_ids=[0, 1, 2, 3, 4]),
                SimpleNamespace(prompt_length=3, token_ids=[0, 1, 2, 3]),
            ],
        },
        step_time_seconds=2.0,
        global_step=3,
        policy_version=7,
    )

    assert payload["prompt_tokens"] == 4.5
    assert payload["completion_tokens"] == 5.0
    assert payload["total_tokens"] == 9.5
    assert payload["samples_per_second"] == 1.0
    assert payload["tokens_per_second"] == 9.5
    assert payload["reward_mean"] == 2.0
    assert payload["grad_norm"] == 3.0
    assert payload["lr"] == 1.0e-5


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


def test_session_logger_logs_rollout_metrics_and_version_skew() -> None:
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
        }
    )
    logger.record_runtime_status(
        rollout_status={
            "heartbeat_age_s": 0.25,
            "latest_published_train_version": 7,
            "min_weight_version": 5,
            "max_weight_version": 7,
            "stale_worker_count": 1,
        },
        cluster_status={},
    )
    logger.record_rollout_batch(
        prompt_token_counts=[3, 5],
        completion_token_counts=[2, 4],
        latency_seconds=0.5,
        pending_generate_count=6,
        tokenize_dedupe_hits=1,
    )
    logger.flush(force=True)

    assert any(
        payload.get("rollout/batch_latency_ms_mean") == 500.0
        and payload.get("rollout/tokenize_dedupe_hit_rate") == 0.5
        and payload.get("rollout/pending_generate_count") == 6.0
        and "rollout_runtime/batch_latency_ms_mean" not in payload
        for payload in fake_run.logged
    )
    assert any(
        payload.get("status/rollout_max_version_skew") == 2.0
        and payload.get("status/min_weight_version") == 5.0
        and payload.get("status/max_weight_version") == 7.0
        for payload in fake_run.logged
    )
    logger.finish()


def test_session_logger_snapshot_includes_rollout_max_version_skew() -> None:
    logger = SessionLogger()

    payload = logger.snapshot(
        train_status={"policy_version": 7},
        rollout_status={
            "latest_published_train_version": 7,
            "min_weight_version": 5,
            "max_weight_version": 7,
        },
        cluster_status={},
    )

    rollout = payload["rollout"]
    assert isinstance(rollout, dict)
    assert rollout["max_version_skew"] == 2.0


def main() -> int:
    return run_tests(
        [
            test_session_logger_uses_runtime_wandb_config,
            test_session_logger_does_not_use_env_to_enable_wandb,
            test_session_logger_runtime_wandb_requires_project,
            test_build_train_update_normalizes_token_metrics_per_trajectory,
            test_session_logger_logs_train_metrics_on_wall_time_axis,
            test_session_logger_logs_rollout_metrics_and_version_skew,
            test_session_logger_snapshot_includes_rollout_max_version_skew,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
