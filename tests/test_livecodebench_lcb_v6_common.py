# Copyright 2026 openforge

from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

common = importlib.import_module("examples.livecodebench_lcb_v6.common")
gateway_types = importlib.import_module("openforge.gateway.types")
RuntimeConfig = gateway_types.RuntimeConfig
StartSessionRequest = gateway_types.StartSessionRequest


def _runtime_config() -> RuntimeConfig:
    return StartSessionRequest.model_validate(
        {
            "runtime": {
                "algo": {"name": "grpo", "kl_coef": 0.0},
                "model": {
                    "model_name_or_path": "model-a",
                    "tokenizer_name_or_path": "model-a-tokenizer",
                    "attn_implementation": "flash_attention_2",
                    "chat_template_kwargs": {"enable_thinking": False},
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
                    "global_batch_size": 8,
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
                        "max_new_tokens": 16,
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_train_setup_materializes_validation_split(tmp_path: Path) -> None:
    """prepare_train_setup should write a normalized validation JSONL when enabled."""
    data_dir = tmp_path / "dataset"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {
                "prompt": "train prompt",
                "reward_spec": {
                    "task_type": "stdin_stdout",
                    "tests": [{"stdin": "1", "stdout": "1"}],
                },
                "problem_id": "train-1",
            }
        ],
    )
    _write_jsonl(
        data_dir / "validation.jsonl",
        [
            {
                "prompt": "validation prompt",
                "reward_spec": {
                    "task_type": "stdin_stdout",
                    "tests": [{"stdin": "2", "stdout": "2"}],
                },
                "problem_id": "validation-1",
            }
        ],
    )
    args = Namespace(
        artifact_dir=str(tmp_path / "artifacts"),
        runtime_config=None,
        prompt_data=str(data_dir),
        validation_data=None,
        seed=0,
        group_size=8,
        total_epochs=1,
        max_train_examples=None,
        max_validation_examples=None,
        wait_timeout=7200.0,
        train_group_parallelism=None,
        train_group_retries=0,
        max_updates=None,
        validation_every_updates=2,
        train_temperature=None,
        train_top_p=None,
        train_top_k=None,
        repetition_penalty=None,
        train_max_new_tokens=None,
        judge_timeout=5.0,
        judge_memory_mb=1024,
    )

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(str(messages[0]["content"]).split())))

    with (
        patch.object(common, "load_runtime_config", return_value=_runtime_config()),
        patch.object(common, "load_tokenizer", return_value=_Tokenizer()),
    ):
        setup = common.prepare_train_setup(args)

    validation_path = tmp_path / "artifacts" / "data" / "validation.jsonl"
    assert setup["validation_path"] == validation_path
    assert validation_path.exists()
    validation_rows = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert validation_rows == [
        {
            "prompt": "validation prompt",
            "reward_spec": {
                "task_type": "stdin_stdout",
                "tests": [{"stdin": "2", "stdout": "2"}],
                "timeout_seconds": 5.0,
                "memory_limit_mb": 1024,
            },
            "problem_id": "validation-1",
        }
    ]
    assert setup["summary"]["train_examples"] == 1
    assert setup["summary"]["validation_examples"] == 1
    assert setup["summary"]["validation_data"] == str(data_dir)


def test_prepare_train_setup_filters_overlong_prompts(tmp_path: Path) -> None:
    """prepare_train_setup should drop prompts above the SDPO prompt-token cap."""
    data_dir = tmp_path / "dataset"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {
                "prompt": "short prompt",
                "reward_spec": {
                    "task_type": "stdin_stdout",
                    "tests": [{"stdin": "1", "stdout": "1"}],
                },
                "problem_id": "keep",
            },
            {
                "prompt": " ".join(["tok"] * 3000),
                "reward_spec": {
                    "task_type": "stdin_stdout",
                    "tests": [{"stdin": "2", "stdout": "2"}],
                },
                "problem_id": "drop",
            },
        ],
    )
    args = Namespace(
        artifact_dir=str(tmp_path / "artifacts"),
        runtime_config=None,
        prompt_data=str(data_dir),
        validation_data=None,
        seed=0,
        group_size=8,
        total_epochs=1,
        max_train_examples=None,
        max_validation_examples=None,
        wait_timeout=7200.0,
        train_group_parallelism=None,
        train_group_retries=0,
        max_updates=None,
        validation_every_updates=0,
        train_temperature=None,
        train_top_p=None,
        train_top_k=None,
        repetition_penalty=None,
        train_max_new_tokens=None,
        judge_timeout=6.0,
        judge_memory_mb=1024,
    )

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(str(messages[0]["content"]).split())))

    with (
        patch.object(common, "load_runtime_config", return_value=_runtime_config()),
        patch.object(common, "load_tokenizer", return_value=_Tokenizer()),
    ):
        setup = common.prepare_train_setup(args)

    assert len(setup["inputs"]) == 1
    assert setup["inputs"][0]["problem_id"] == "keep"
    assert setup["summary"]["dropped_train_examples_over_prompt_limit"] == 1
