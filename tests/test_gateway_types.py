# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

from _script_test_utils import expect_raises, run_tests

from openforge.gateway.types import RuntimeConfig


def _runtime_payload(*, algo: dict[str, object], lag: int) -> dict[str, object]:
    return {
        "algo": algo,
        "model": {
            "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "tokenizer_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "reference_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "attn_implementation": "sdpa",
        },
        "train": {
            "backend": "fsdp2",
            "config": {
                "gradient_checkpointing": True,
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
                    "lr": 1.0e-6,
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
            "ppo_epochs": 1,
            "max_rollout_policy_lag": lag,
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
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "max_new_tokens": 32,
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


def test_runtime_config_accepts_on_policy_grpo() -> None:
    cfg = RuntimeConfig.model_validate(
        _runtime_payload(algo={"name": "grpo", "kl_coef": 0.0}, lag=0)
    )
    assert cfg.algo.name == "grpo"
    assert cfg.train.max_rollout_policy_lag == 0


def test_runtime_config_rejects_stale_lag_for_grpo() -> None:
    with expect_raises(
        ValueError,
        "train.max_rollout_policy_lag must be 0 when algo.name is grpo",
    ):
        RuntimeConfig.model_validate(
            _runtime_payload(algo={"name": "grpo", "kl_coef": 0.0}, lag=1)
        )


def test_runtime_config_rejects_zero_lag_for_grpo_tis() -> None:
    with expect_raises(
        ValueError,
        "train.max_rollout_policy_lag must be > 0 when algo.name is grpo_tis",
    ):
        RuntimeConfig.model_validate(
            _runtime_payload(
                algo={"name": "grpo_tis", "kl_coef": 0.0, "tis_cap": 2.0},
                lag=0,
            )
        )


def main() -> int:
    return run_tests(
        [
            test_runtime_config_accepts_on_policy_grpo,
            test_runtime_config_rejects_stale_lag_for_grpo,
            test_runtime_config_rejects_zero_lag_for_grpo_tis,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
