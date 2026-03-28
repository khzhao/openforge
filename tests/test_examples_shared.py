# Copyright 2026 openforge

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from examples import shared

import openforge.ninja as ninja
from openforge.gateway.types import RuntimeConfig, StartSessionRequest


def _runtime_config(*, global_batch_size: int) -> RuntimeConfig:
    return StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "model-a",
                    "tokenizer_name_or_path": "model-a-tokenizer",
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
                    "global_batch_size": global_batch_size,
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
            },
        }
    ).runtime


class _FakeSession:
    def __init__(self) -> None:
        self.policy_version = 0

    def current_train_policy_version(self) -> int:
        return self.policy_version


class _FakeAgent:
    def __init__(self) -> None:
        self._session_obj = _FakeSession()
        self.saved = False

    @contextmanager
    def _session(self):
        yield self._session_obj

    def _validate_call(self, call_args, call_kwargs, *, request_index: int) -> None:
        _ = (call_args, call_kwargs, request_index)

    def save(self) -> dict[str, object]:
        self.saved = True
        return {"checkpoint_path": "/tmp/checkpoint"}


def test_run_train_skips_failed_groups_and_tops_up_update() -> None:
    """run_train should replace failed groups until one update is full."""
    agent = _FakeAgent()
    progress_updates: list[dict[str, object]] = []
    execute_calls: list[list[str]] = []

    results_by_call = [
        (
            [
                ninja._GroupedExecutionResult(
                    request_index=0,
                    trajectory_ids=["traj_a0", "traj_a1"],
                    rewards=[1.0, 1.0],
                )
            ],
            [
                ninja._GroupedExecutionFailure(
                    request_index=1, error="RuntimeError: boom"
                )
            ],
        ),
        (
            [
                ninja._GroupedExecutionResult(
                    request_index=0,
                    trajectory_ids=["traj_c0", "traj_c1"],
                    rewards=[0.5, 0.5],
                )
            ],
            [],
        ),
    ]

    def fake_execute_grouped_results(
        agent_arg,
        session,
        call_specs,
        *,
        group_size: int,
        concurrency: int,
        retries: int,
        raise_on_failure: bool,
    ):
        _ = (agent_arg, session, group_size, concurrency, retries, raise_on_failure)
        execute_calls.append([spec[1]["prompt"] for spec in call_specs])
        return results_by_call.pop(0)

    def fake_wait_for_trained(session, trajectory_ids, *, timeout: float) -> None:
        assert timeout == 3600.0
        assert trajectory_ids == ["traj_a0", "traj_a1", "traj_c0", "traj_c1"]
        session.policy_version = 1

    with patch.object(ninja, "_execute_grouped_results", fake_execute_grouped_results):
        with patch.object(
            ninja, "_wait_for_trained_trajectories", fake_wait_for_trained
        ):
            with patch.object(shared.random.Random, "shuffle", lambda self, seq: None):
                summary = shared.run_train(
                    agent,
                    runtime_config=_runtime_config(global_batch_size=4),
                    inputs=[
                        {"prompt": "a"},
                        {"prompt": "b"},
                        {"prompt": "c"},
                    ],
                    group_size=2,
                    epochs=1,
                    seed=0,
                    parallelism=4,
                    retries=2,
                    wait_timeout=3600.0,
                    max_updates=1,
                    progress_callback=progress_updates.append,
                )

    assert execute_calls == [["a", "b"], ["c"]]
    assert summary["completed_updates"] == 1
    assert summary["failed_prompt_groups"] == 1
    assert summary["final_policy_version"] == 1
    assert summary["last_train_update"]["failed_prompt_groups"] == 1
    assert summary["last_train_update"]["attempted_prompt_groups"] == 3
    assert progress_updates == [summary["last_train_update"]]


def test_plan_train_batches_chunks_updates_by_global_batch_size() -> None:
    """plan_train_batches should chunk prompt groups into full updates."""
    with patch.object(shared.random.Random, "shuffle", lambda self, seq: None):
        plan = shared.plan_train_batches(
            runtime_config=_runtime_config(global_batch_size=4),
            inputs=[
                {"prompt": "a"},
                {"prompt": "b"},
                {"prompt": "c"},
            ],
            group_size=2,
            epochs=2,
            seed=0,
            max_updates=2,
        )

    assert plan["expected_updates"] == 2
    assert plan["prompt_groups_per_update"] == 2
    assert plan["global_batch_size"] == 4
    assert plan["train_groups"] == 6
    assert plan["train_groups_planned"] == 4
    assert plan["train_groups_dropped"] == 2
    assert plan["update_inputs"] == [
        [{"prompt": "a"}, {"prompt": "b"}],
        [{"prompt": "c"}, {"prompt": "a"}],
    ]
