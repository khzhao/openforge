#!/usr/bin/env python3
# Copyright 2026 openforge

"""Run a real large-model train -> rollout artifact-sync integration test.

This script uses the current refactored train and rollout modules directly:

1. Launch one `TrainWorkerGroup` with the FSDP2 backend.
2. Launch one `RolloutWorkerGroup` with a real SGLang runtime.
3. Run real inference against the rollout engine.
4. Run one or more real optimizer steps on the training side.
5. Export a disk-backed policy artifact from train rank 0.
6. Load that artifact into the rollout engine, verify the version changed, and
   run inference again.

Example:
    python test_scripts/train_rollout_large_model_sync_test_script.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import sys
from dataclasses import asdict
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import ray
import requests
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer

import openforge.rollout.group as rollout_group_module
from openforge.configs.models import OpenForgeConfig
from openforge.rollout.group import RolloutWorkerGroup
from openforge.rollout.sglang.client import SGLangControlClient
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.rollout.worker import RolloutWorker
from openforge.train.group import TrainWorkerGroup
from openforge.utils.networking import get_free_port


DISALLOWED_SERVER_ARG_OVERRIDES = {
    "host",
    "model_path",
    "port",
    "served_model_name",
    "tokenizer_path",
    "weight_version",
}


class ConfigurableRolloutWorker(RolloutWorker):
    """Rollout worker variant used only by this integration script."""

    server_arg_overrides: dict[str, Any] = {}
    startup_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 1.0

    def initialize(self, spec):
        import openforge.rollout.sglang.engine_runtime as engine_runtime_module

        engine_runtime_module.HEALTHCHECK_TIMEOUT_SECONDS = (
            type(self).startup_timeout_seconds
        )
        engine_runtime_module.HEALTHCHECK_POLL_INTERVAL_SECONDS = (
            type(self).poll_interval_seconds
        )
        return super().initialize(spec)

    def _build_runtime_spec(self) -> SGLangEngineSpec:
        spec = super()._build_runtime_spec()
        overrides = dict(type(self).server_arg_overrides)
        invalid_keys = DISALLOWED_SERVER_ARG_OVERRIDES.intersection(overrides)
        if invalid_keys:
            blocked = ", ".join(sorted(invalid_keys))
            raise ValueError(
                f"--server-args-json cannot override reserved keys: {blocked}"
            )
        spec.server_args.update(overrides)
        return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch real train and rollout workers, sync a policy artifact, "
            "and verify the rollout engine loads the new checkpoint."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path or Hugging Face repo ID shared by train and rollout.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path or repo ID. Defaults to --model-path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write one sentence about distributed training and RLHF systems.",
        help="Prompt encoded into the synthetic training batch.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=64,
        help="Maximum number of prompt tokens kept in the synthetic batch.",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=8,
        help="Minimum number of prompt tokens enforced in the synthetic batch.",
    )
    parser.add_argument(
        "--inference-prompt",
        type=str,
        default="Write one short sentence about reinforcement learning systems.",
        help="Prompt used for live rollout inference requests.",
    )
    parser.add_argument(
        "--inference-max-new-tokens",
        type=int,
        default=24,
        help="max_new_tokens used for live rollout inference requests.",
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=0.0,
        help="temperature used for live rollout inference requests.",
    )
    parser.add_argument(
        "--inference-top-p",
        type=float,
        default=1.0,
        help="top_p used for live rollout inference requests.",
    )
    parser.add_argument(
        "--inference-timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout used for live rollout inference requests.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip live /generate checks and only verify rollout control-plane behavior.",
    )
    parser.add_argument(
        "--train-world-size",
        type=int,
        default=None,
        help=(
            "Number of train ranks. Each rank consumes one GPU. "
            "Defaults to all visible GPUs minus one rollout GPU."
        ),
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1,
        help="Number of optimizer steps to take before syncing.",
    )
    parser.add_argument(
        "--policy-version",
        type=int,
        default=1,
        help="Policy version written into the rollout runtime during sync.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Train-side AdamW learning rate for the integration step.",
    )
    parser.add_argument(
        "--param-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Train-side FSDP2 parameter dtype.",
    )
    parser.add_argument(
        "--reduce-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Train-side FSDP2 reduce dtype.",
    )
    parser.add_argument(
        "--amp-precision",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Autocast precision for the training forward pass.",
    )
    parser.add_argument(
        "--enable-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA autocast during the train forward pass.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model gradient checkpointing on the train side.",
    )
    parser.add_argument(
        "--offload-mode",
        choices=("none", "cpu"),
        default="none",
        help="Train-side FSDP2 offload mode.",
    )
    parser.add_argument(
        "--cpus-per-train-worker",
        type=int,
        default=4,
        help="Logical CPUs reserved per train worker actor.",
    )
    parser.add_argument(
        "--rollout-cpus-per-engine",
        type=int,
        default=4,
        help="Logical CPUs reserved per rollout engine actor.",
    )
    parser.add_argument(
        "--cluster-gpus",
        type=int,
        default=None,
        help="Logical GPUs exposed to local Ray. Defaults to the required total.",
    )
    parser.add_argument(
        "--cluster-cpus",
        type=int,
        default=None,
        help="Logical CPUs exposed to local Ray. Defaults to the required total.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum rollout-engine startup wait time.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for rollout readiness.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Client timeout used for rollout control requests.",
    )
    parser.add_argument(
        "--server-args-json",
        type=str,
        default=None,
        help=(
            "Additional SGLang server args as a JSON object. "
            f"Reserved keys {sorted(DISALLOWED_SERVER_ARG_OVERRIDES)!r} are blocked."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory used for train checkpoints and Ray temp files.",
    )
    return parser.parse_args()


def parse_server_arg_overrides(raw_json: str | None) -> dict[str, Any]:
    if raw_json is None:
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("--server-args-json must decode to a JSON object")
    return dict(parsed)


def resolve_gpu_layout(
    args: argparse.Namespace,
    *,
    visible_gpus: int,
) -> tuple[int, int]:
    cluster_gpus = args.cluster_gpus or visible_gpus
    if cluster_gpus > visible_gpus:
        raise ValueError(
            f"--cluster-gpus={cluster_gpus} exceeds visible GPUs={visible_gpus}"
        )

    train_world_size = args.train_world_size
    if train_world_size is None:
        train_world_size = cluster_gpus - 1

    if train_world_size <= 0:
        raise ValueError(
            "train_world_size must be at least 1 after reserving one rollout GPU"
        )

    required_gpus = train_world_size + 1
    if cluster_gpus < required_gpus:
        raise ValueError(
            f"Need at least {required_gpus} cluster GPUs for "
            f"{train_world_size} train ranks plus one rollout engine, got {cluster_gpus}"
        )

    return train_world_size, cluster_gpus


def validate_environment(args: argparse.Namespace) -> tuple[int, int, int, int]:
    visible_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible_gpus == 0:
        raise RuntimeError(
            "This integration script requires visible CUDA devices. "
            "No GPUs are currently available."
        )

    train_world_size, cluster_gpus = resolve_gpu_layout(
        args,
        visible_gpus=visible_gpus,
    )
    required_gpus = train_world_size + 1
    if visible_gpus < required_gpus:
        raise RuntimeError(
            f"Need at least {required_gpus} visible GPUs, found {visible_gpus}"
        )

    required_cpus = (
        train_world_size * args.cpus_per_train_worker
        + args.rollout_cpus_per_engine
        + 2
    )
    cluster_cpus = args.cluster_cpus or required_cpus
    if cluster_cpus < required_cpus:
        raise ValueError(
            f"--cluster-cpus must be at least {required_cpus}, got {cluster_cpus}"
        )

    if args.train_steps <= 0:
        raise ValueError("--train-steps must be > 0")
    if args.max_prompt_tokens < 2:
        raise ValueError("--max-prompt-tokens must be >= 2")
    if args.min_prompt_tokens < 2:
        raise ValueError("--min-prompt-tokens must be >= 2")

    return visible_gpus, train_world_size, cluster_gpus, cluster_cpus


def make_config(
    args: argparse.Namespace,
    *,
    checkpoints_dir: Path,
    cluster_gpus: int,
    cluster_cpus: int,
) -> OpenForgeConfig:
    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "test"},
            "gateway": {
                "host": "127.0.0.1",
                "port": 8000,
            },
            "model": {
                "model_name_or_path": args.model_path,
                "tokenizer_name_or_path": args.tokenizer_path or args.model_path,
            },
            "cluster": [
                {
                    "node_pool": "default",
                    "num_nodes": 1,
                    "num_gpus_per_node": cluster_gpus,
                    "num_cpus_per_node": cluster_cpus,
                }
            ],
            "train": {
                "backend": "fsdp2",
                "backend_config": {
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "reshard_after_forward": False,
                    "mixed_precision": {
                        "param_dtype": args.param_dtype,
                        "reduce_dtype": args.reduce_dtype,
                    },
                    "offload": {
                        "mode": args.offload_mode,
                        "pin_memory": args.offload_mode == "cpu",
                    },
                    "amp": {
                        "enabled": args.enable_amp,
                        "precision": args.amp_precision,
                        "use_grad_scaler": args.amp_precision == "float16",
                    },
                    "optim": {
                        "lr": args.learning_rate,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 0,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                        "min_lr_rate": None,
                    },
                },
                "global_batch_size": args.train_world_size,
                "mini_batch_size": args.train_world_size,
                "micro_batch_size": 1,
                "checkpoints_dir": str(checkpoints_dir),
                "cpus_per_worker": args.cpus_per_train_worker,
                "parallelism": {
                    "data_parallel_size": args.train_world_size,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
                "placement": {
                    "node_pool": "default",
                    "strategy": "PACK",
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 16,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                    "spaces_between_words": True,
                },
                "engine_topology": "regular",
                "engines": [
                    {
                        "name": "regular",
                        "role": "regular",
                        "replicas": 1,
                        "gpus_per_engine": 1,
                        "cpus_per_engine": args.rollout_cpus_per_engine,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "placement": {
                            "node_pool": "default",
                            "strategy": "PACK",
                        },
                    }
                ],
            },
        }
    )


def build_batch(tokenizer: AutoTokenizer, args: argparse.Namespace) -> TensorDict:
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not token_ids:
        raise RuntimeError("tokenizer produced no tokens for the prompt")

    while len(token_ids) < args.min_prompt_tokens:
        token_ids = token_ids + token_ids
    token_ids = token_ids[: args.max_prompt_tokens]

    if len(token_ids) < 2:
        raise RuntimeError("synthetic batch needs at least two tokens")

    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    token_count = int(token_tensor.numel())
    return TensorDict(
        {
            "tokens": token_tensor,
            "cu_seqlens": torch.tensor([0, token_count], dtype=torch.int32),
            "loss_mask": torch.ones(token_count, dtype=torch.float32),
            "rewards": torch.ones(token_count, dtype=torch.float32),
            "position_ids": torch.arange(token_count, dtype=torch.long),
        },
        batch_size=[],
    )


def make_train_microbatches(
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> list[list[TensorDict]]:
    per_rank: list[list[TensorDict]] = []
    for _ in range(args.train_world_size):
        per_rank.append([build_batch(tokenizer, args)])
    return per_rank


def print_json(label: str, payload: Any) -> None:
    print(f"{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def decode_json(raw_body: str) -> Any:
    if not raw_body:
        return None
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OSError(f"request POST {path} failed: {exc}") from exc

    raw_body = response.text
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"request POST {path} failed with status {response.status_code}: "
            f"{raw_body.strip()}"
        ) from exc
    return response.status_code, decode_json(raw_body)


def build_generate_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "text": args.inference_prompt,
        "sampling_params": {
            "max_new_tokens": args.inference_max_new_tokens,
            "temperature": args.inference_temperature,
            "top_p": args.inference_top_p,
        },
        "stream": False,
    }


def run_rollout_inference(
    endpoint_url: str | None,
    args: argparse.Namespace,
    *,
    label: str,
) -> Any:
    if endpoint_url is None:
        raise RuntimeError("rollout endpoint did not expose a URL for inference")
    payload = build_generate_payload(args)
    print_json(f"{label}_request", payload)
    status, response = post_json(
        endpoint_url,
        "/generate",
        payload,
        timeout=max(args.inference_timeout_seconds, args.request_timeout_seconds),
    )
    if status != HTTPStatus.OK:
        raise RuntimeError(f"/generate returned unexpected status {status}")
    print_json(f"{label}_response", response)
    return response


def perform_artifact_sync(
    train_group: TrainWorkerGroup,
    rollout_group: RolloutWorkerGroup,
    *,
    step: int,
    policy_version: int,
) -> list[object]:
    artifact = train_group.export_policy_artifact(
        step=step,
        policy_version=policy_version,
    )
    print_json("policy_artifact", asdict(artifact))
    return rollout_group.load_policy_artifact(artifact)


def rollout_client(endpoint_url: str | None) -> SGLangControlClient:
    if endpoint_url is None:
        raise RuntimeError("rollout endpoint did not expose a URL")
    return SGLangControlClient(endpoint_url)


def main() -> int:
    args = parse_args()
    _visible_gpus, train_world_size, cluster_gpus, cluster_cpus = validate_environment(
        args
    )
    args.train_world_size = train_world_size
    server_arg_overrides = parse_server_arg_overrides(args.server_args_json)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    def run(artifacts_dir: Path) -> int:
        checkpoints_dir = artifacts_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        cfg = make_config(
            args,
            checkpoints_dir=checkpoints_dir,
            cluster_gpus=cluster_gpus,
            cluster_cpus=cluster_cpus,
        )
        microbatches = make_train_microbatches(tokenizer, args)

        print(f"artifacts_dir={artifacts_dir}")
        print(f"model_path={args.model_path}")
        print(f"tokenizer_path={tokenizer_path}")
        print("sync_mode=artifact")
        print(f"train_world_size={args.train_world_size}")
        print(f"cluster_gpus={cluster_gpus}")
        print(f"cluster_cpus={cluster_cpus}")
        print_json("rollout_server_args_overrides", server_arg_overrides)

        ConfigurableRolloutWorker.server_arg_overrides = server_arg_overrides
        ConfigurableRolloutWorker.startup_timeout_seconds = (
            args.startup_timeout_seconds
        )
        ConfigurableRolloutWorker.poll_interval_seconds = (
            args.poll_interval_seconds
        )

        train_group: TrainWorkerGroup | None = None
        rollout_group: RolloutWorkerGroup | None = None

        with TemporaryDirectory(prefix="ofray_") as ray_temp_dir:
            print(f"ray_temp_dir={ray_temp_dir}")
            ray.init(
                num_cpus=cluster_cpus,
                num_gpus=cluster_gpus,
                include_dashboard=False,
                ignore_reinit_error=False,
                log_to_driver=True,
                _temp_dir=ray_temp_dir,
            )
            try:
                with mock.patch.object(
                    rollout_group_module,
                    "RolloutWorker",
                    ConfigurableRolloutWorker,
                ):
                    rollout_group = RolloutWorkerGroup(cfg)
                    routable_endpoints = rollout_group.initialize()
                if len(routable_endpoints) != 1:
                    raise RuntimeError(
                        f"expected exactly one routable rollout endpoint, got {routable_endpoints}"
                    )
                endpoint = routable_endpoints[0]
                print_json("rollout_endpoint", asdict(endpoint))

                client = rollout_client(endpoint.url)
                print(f"rollout_healthy={client.health_generate(timeout=5.0)}")
                print_json(
                    "rollout_server_info",
                    client.get_server_info(timeout=args.request_timeout_seconds),
                )
                print(
                    f"initial_rollout_weight_version="
                    f"{client.get_weight_version(timeout=args.request_timeout_seconds)!r}"
                )
                if not args.skip_inference:
                    run_rollout_inference(
                        endpoint.url,
                        args,
                        label="rollout_generate_before_training",
                    )

                master_port = get_free_port(start=29500)
                train_group = TrainWorkerGroup(
                    cfg,
                    master_addr="127.0.0.1",
                    master_port=master_port,
                    strategy="PACK",
                )
                train_states = train_group.initialize()
                print_json(
                    "train_states",
                    [asdict(state) for state in train_states],
                )

                for step in range(1, args.train_steps + 1):
                    results = train_group.step(microbatches, global_step=step)
                    print_json(
                        f"train_step_{step}_results",
                        [asdict(result) for result in results],
                    )

                snapshot_before = client.check_weights(
                    action="snapshot",
                    timeout=max(args.request_timeout_seconds, 30.0),
                )
                print_json("weights_checker_snapshot_before", snapshot_before)

                synced_endpoints = perform_artifact_sync(
                    train_group,
                    rollout_group,
                    step=args.train_steps,
                    policy_version=args.policy_version,
                )
                if len(synced_endpoints) != 1:
                    raise RuntimeError(
                        f"expected one routable endpoint after sync, got {synced_endpoints}"
                    )
                synced_endpoint = synced_endpoints[0]
                print_json("rollout_endpoint_after_first_sync", asdict(synced_endpoint))
                if synced_endpoint.policy_version != args.policy_version:
                    raise RuntimeError(
                        "rollout endpoint did not report the requested policy version after sync: "
                        f"{synced_endpoint.policy_version!r} != {args.policy_version!r}"
                    )

                updated_weight_version = client.get_weight_version(
                    timeout=args.request_timeout_seconds
                )
                print(f"updated_rollout_weight_version={updated_weight_version!r}")
                if updated_weight_version != str(args.policy_version):
                    raise RuntimeError(
                        "rollout weight_version did not match the requested policy version: "
                        f"{updated_weight_version!r} != {args.policy_version!r}"
                    )
                if not args.skip_inference:
                    run_rollout_inference(
                        synced_endpoint.url,
                        args,
                        label="rollout_generate_after_first_sync",
                    )

                compare_error: str | None = None
                try:
                    client.check_weights(
                        action="compare",
                        timeout=max(args.request_timeout_seconds, 30.0),
                    )
                except RuntimeError as exc:
                    compare_error = str(exc)
                else:
                    raise RuntimeError(
                        "weights_checker compare unexpectedly succeeded after the first sync"
                    )

                print("weights_checker_compare_after_first_sync:")
                print(compare_error)

                snapshot_after_first_sync = client.check_weights(
                    action="snapshot",
                    timeout=max(args.request_timeout_seconds, 30.0),
                )
                print_json(
                    "weights_checker_snapshot_after_first_sync",
                    snapshot_after_first_sync,
                )

                synced_endpoints = perform_artifact_sync(
                    train_group,
                    rollout_group,
                    step=args.train_steps,
                    policy_version=args.policy_version,
                )
                synced_endpoint = synced_endpoints[0]
                print_json(
                    "rollout_endpoint_after_idempotent_sync",
                    asdict(synced_endpoint),
                )
                if synced_endpoint.policy_version != args.policy_version:
                    raise RuntimeError(
                        "rollout endpoint changed away from the requested policy version "
                        f"after the idempotent sync: {synced_endpoint.policy_version!r}"
                    )
                compare_after_idempotent_sync = client.check_weights(
                    action="compare",
                    timeout=max(args.request_timeout_seconds, 30.0),
                )
                print_json(
                    "weights_checker_compare_after_idempotent_sync",
                    compare_after_idempotent_sync,
                )
                if not args.skip_inference:
                    run_rollout_inference(
                        synced_endpoint.url,
                        args,
                        label="rollout_generate_after_idempotent_sync",
                    )

                print("SUCCESS: train -> rollout artifact sync verified.")
                return 0
            finally:
                if train_group is not None:
                    try:
                        train_group.shutdown()
                    except Exception as exc:  # pragma: no cover - cleanup path
                        print(f"train_group.shutdown failed: {exc}", file=sys.stderr)
                if rollout_group is not None:
                    try:
                        rollout_group.shutdown()
                    except Exception as exc:  # pragma: no cover - cleanup path
                        print(f"rollout_group.shutdown failed: {exc}", file=sys.stderr)
                ray.shutdown()

    if args.artifacts_dir is None:
        with TemporaryDirectory(prefix="openforge_large_model_sync_") as temp_dir:
            return run(Path(temp_dir))
    return run(args.artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())
