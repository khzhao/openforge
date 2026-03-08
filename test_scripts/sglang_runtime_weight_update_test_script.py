#!/usr/bin/env python3
# Copyright 2026 openforge

"""Launch a real SGLang runtime and verify live weight patching.

This script exercises the actual `SGLangEngineRuntime.update_weights_from_tensor`
path against a real SGLang HTTP server. It:
- launches one runtime
- loads the same Hugging Face model on CPU
- selects one floating-point parameter
- mutates a short prefix of that tensor locally
- pushes the mutated tensor into the live SGLang server
- verifies the update generically via SGLang's weights_checker

Example:
    python test_scripts/sglang_runtime_weight_update_test_script.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --weight-name model.layers.0.self_attn.q_proj.weight
"""

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping
from typing import Any

import torch

import openforge.rollout.sglang.engine_runtime as runtime_module
from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.utils.networking import get_free_port

DISALLOWED_SERVER_ARG_OVERRIDES = {
    "host",
    "model_path",
    "port",
    "tokenizer_path",
}

COMMON_WEIGHT_NAME_CANDIDATES = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Launch a real SGLang runtime and verify live weight updates.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path or Hugging Face repo ID to serve and load locally.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path or repo ID. Defaults to --model-path.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Optional served model name exposed by the server.",
    )
    parser.add_argument(
        "--weight-name",
        type=str,
        default=None,
        help="Exact HF parameter name to patch. Defaults to auto-selection.",
    )
    parser.add_argument(
        "--mutate-elements",
        type=int,
        default=8,
        help="Number of leading flattened elements to modify before syncing.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.25,
        help="Delta added to each mutated element.",
    )
    parser.add_argument(
        "--bind-host",
        type=str,
        default="0.0.0.0",
        help="Host the SGLang server binds to.",
    )
    parser.add_argument(
        "--connect-host",
        type=str,
        default="127.0.0.1",
        help="Host used by the local client to reach the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Defaults to the first free port at or above 30000.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout used by the runtime client.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum time to wait for runtime readiness.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for startup readiness.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to SGLang and HF model loading.",
    )
    parser.add_argument(
        "--skip-server-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass skip_server_warmup to SGLang server args.",
    )
    parser.add_argument(
        "--initial-weight-version",
        type=str,
        default="0",
        help="Initial weight version passed to the server at launch.",
    )
    parser.add_argument(
        "--updated-weight-version",
        type=str,
        default="1",
        help="Required integer-string policy version applied during the update.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size. This test currently requires 1.",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size. This test currently requires 1.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size. This test currently requires 1.",
    )
    parser.add_argument(
        "--attn-cp-size",
        type=int,
        default=1,
        help="Attention/context parallel size. This test currently requires 1.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert parallel size. This test currently requires 1.",
    )
    parser.add_argument(
        "--server-args-json",
        type=str,
        default=None,
        help=(
            "Extra SGLang ServerArgs entries as a JSON object. "
            f"Core keys {sorted(DISALLOWED_SERVER_ARG_OVERRIDES)!r} are reserved."
        ),
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the server running until interrupted with Ctrl-C.",
    )
    return parser.parse_args()


def _parse_server_arg_overrides(raw_json: str | None) -> dict[str, Any]:
    """Parse extra server args JSON."""
    if raw_json is None:
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, Mapping):
        raise ValueError("--server-args-json must decode to a JSON object")
    invalid_keys = DISALLOWED_SERVER_ARG_OVERRIDES.intersection(parsed)
    if invalid_keys:
        blocked = ", ".join(sorted(invalid_keys))
        raise ValueError(
            f"--server-args-json cannot override reserved keys: {blocked}"
        )
    return dict(parsed)


def _build_runtime_spec(args: argparse.Namespace) -> SGLangEngineSpec:
    """Construct the runtime spec used by the weight update test."""
    port = args.port if args.port is not None else get_free_port(start=30000)
    tokenizer_path = args.tokenizer_path or args.model_path
    served_model_name = args.served_model_name or args.model_path

    server_args: dict[str, Any] = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "host": args.bind_host,
        "port": port,
        "trust_remote_code": args.trust_remote_code,
        "skip_server_warmup": args.skip_server_warmup,
        "served_model_name": served_model_name,
        "weight_version": args.initial_weight_version,
        "dp_size": args.dp_size,
        "pp_size": args.pp_size,
        "tp_size": args.tp_size,
        "attn_cp_size": args.attn_cp_size,
        "ep_size": args.ep_size,
    }
    server_args.update(_parse_server_arg_overrides(args.server_args_json))

    return SGLangEngineSpec(
        engine_id=0,
        name="sglang-runtime-weight-update-test",
        role="regular",
        host=args.connect_host,
        port=port,
        bootstrap_port=None,
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        policy_version=None,
        server_args=server_args,
    )


def _print_json(label: str, payload: Any) -> None:
    """Print a labeled JSON payload."""
    print(f"{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_reference_state_dict(
    *,
    model_path: str,
    trust_remote_code: bool,
) -> dict[str, torch.Tensor]:
    """Load the HF model locally on CPU and return its state dict."""
    from transformers import AutoModelForCausalLM

    print(f"Loading reference Hugging Face model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    try:
        return {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
    finally:
        del model
        gc.collect()


def _candidate_weight_names(
    state_dict: Mapping[str, torch.Tensor],
    explicit_name: str | None,
) -> list[str]:
    """Return candidate weight names in probe order."""
    if explicit_name is not None:
        return [explicit_name]

    names: list[str] = []
    seen: set[str] = set()

    def maybe_add(name: str) -> None:
        if name in state_dict and name not in seen:
            tensor = state_dict[name]
            if torch.is_floating_point(tensor):
                names.append(name)
                seen.add(name)

    for name in COMMON_WEIGHT_NAME_CANDIDATES:
        maybe_add(name)

    for name, tensor in state_dict.items():
        if name in seen:
            continue
        if not torch.is_floating_point(tensor):
            continue
        if name.endswith(".weight") and tensor.ndim >= 2:
            names.append(name)
            seen.add(name)

    for name, tensor in state_dict.items():
        if name in seen:
            continue
        if torch.is_floating_point(tensor):
            names.append(name)
            seen.add(name)

    return names


def _choose_weight_name_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    explicit_name: str | None,
) -> tuple[str, torch.Tensor]:
    """Select a floating-point tensor name from the HF state dict."""
    candidates = _candidate_weight_names(state_dict, explicit_name)
    if not candidates:
        raise RuntimeError("no floating-point tensors were available for update")

    name = candidates[0]
    if name not in state_dict:
        raise KeyError(f"weight {name!r} was not found in the HF state dict")
    tensor = state_dict[name]
    if not torch.is_floating_point(tensor):
        raise TypeError(f"weight {name!r} is not floating-point")
    return name, tensor.detach().cpu().contiguous().clone()


def _mutate_tensor(
    tensor: torch.Tensor,
    *,
    mutate_elements: int,
    delta: float,
) -> tuple[torch.Tensor, int]:
    """Clone and modify a short tensor prefix."""
    mutated = tensor.detach().cpu().contiguous().clone()
    flat = mutated.reshape(-1)
    mutate_count = min(max(mutate_elements, 1), flat.numel())
    original_prefix = flat[:mutate_count].clone()
    delta_tensor = torch.tensor(delta, dtype=flat.dtype)
    flat[:mutate_count] = flat[:mutate_count] + delta_tensor
    if torch.equal(flat[:mutate_count], original_prefix):
        raise RuntimeError(
            "local mutation had no effect after dtype rounding; increase --delta"
        )
    return mutated, mutate_count


def _validate_parallelism(args: argparse.Namespace) -> None:
    """Reject configurations the script does not verify correctly."""
    if any(
        value != 1
        for value in (
            args.dp_size,
            args.pp_size,
            args.tp_size,
            args.attn_cp_size,
            args.ep_size,
        )
    ):
        raise ValueError(
            "this script currently requires dp/pp/tp/attn_cp/ep sizes to all be 1"
        )


def _serialize_named_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> list[str]:
    """Serialize named tensors into SGLang flattened buckets."""
    from openforge.rollout.sglang.serialization import (
        serialize_named_tensors_for_sglang,
    )

    return serialize_named_tensors_for_sglang(named_tensors)


def _apply_live_update(
    runtime: SGLangEngineRuntime,
    *,
    serialized_named_tensors: list[str],
    weight_version: str,
) -> dict[str, Any]:
    """Pause, flush, update, and resume the SGLang server."""
    paused = False
    try:
        runtime.pause_generation(mode="abort")
        paused = True
        runtime.flush_cache()
        return runtime.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format="flattened_bucket",
            flush_cache=False,
            abort_all_requests=False,
            weight_version=weight_version,
        )
    finally:
        if paused:
            runtime.continue_generation()


def _verify_update_via_weights_checker(
    runtime: SGLangEngineRuntime,
    *,
    weight_name: str,
    serialized_named_tensors: list[str],
    weight_version: str,
) -> dict[str, Any]:
    """Verify a live update through SGLang's generic weights checker."""
    snapshot_before = runtime.check_weights(action="snapshot")
    _print_json("weights_checker_snapshot_before", snapshot_before)

    update_response = _apply_live_update(
        runtime,
        serialized_named_tensors=serialized_named_tensors,
        weight_version=weight_version,
    )

    compare_error: str | None = None
    try:
        runtime.check_weights(action="compare")
    except RuntimeError as exc:
        compare_error = str(exc)
    else:
        raise RuntimeError(
            "weights_checker compare unexpectedly succeeded after a weight update"
        )

    if weight_name not in compare_error:
        raise RuntimeError(
            "weights_checker compare failed, but not on the requested weight: "
            f"{compare_error}"
        )

    print("weights_checker_compare_after_first_update:")
    print(compare_error)

    snapshot_after = runtime.check_weights(action="snapshot")
    _print_json("weights_checker_snapshot_after", snapshot_after)

    idempotent_update_response = _apply_live_update(
        runtime,
        serialized_named_tensors=serialized_named_tensors,
        weight_version=weight_version,
    )
    _print_json("idempotent_update_response", idempotent_update_response)

    compare_after_idempotent_update = runtime.check_weights(action="compare")
    _print_json(
        "weights_checker_compare_after_idempotent_update",
        compare_after_idempotent_update,
    )
    return update_response


def main() -> int:
    """Run the live SGLang weight update test."""
    args = parse_args()
    _validate_parallelism(args)

    runtime_module.HEALTHCHECK_TIMEOUT_SECONDS = args.startup_timeout_seconds
    runtime_module.HEALTHCHECK_POLL_INTERVAL_SECONDS = args.poll_interval_seconds

    spec = _build_runtime_spec(args)
    runtime = SGLangEngineRuntime(
        spec,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    print(f"Launching SGLangEngineRuntime at {spec.url}")
    _print_json("server_args", spec.server_args)

    state_dict = _load_reference_state_dict(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        runtime.start()
        process = runtime.process
        process_pid = None if process is None else process.pid
        print(f"Runtime healthy. child_pid={process_pid}")
        print(f"/health_generate={runtime.is_healthy()}")
        print(f"/flush_cache={runtime.flush_cache()}")
        print(f"initial_weight_version={runtime.get_weight_version()!r}")

        weight_name, reference_tensor = _choose_weight_name_from_state_dict(
            state_dict=state_dict,
            explicit_name=args.weight_name,
        )
        del state_dict
        gc.collect()

        print(f"selected_weight_name={weight_name}")
        print(f"selected_weight_shape={tuple(reference_tensor.shape)}")
        print(f"selected_weight_dtype={reference_tensor.dtype}")

        mutated_tensor, mutate_count = _mutate_tensor(
            reference_tensor,
            mutate_elements=args.mutate_elements,
            delta=args.delta,
        )
        print(f"mutated_elements={mutate_count}")

        serialized_named_tensors = _serialize_named_tensors(
            [(weight_name, mutated_tensor)]
        )

        verification_mode = "weights_checker"
        print(f"verification_mode={verification_mode}")
        update_response = _verify_update_via_weights_checker(
            runtime,
            weight_name=weight_name,
            serialized_named_tensors=serialized_named_tensors,
            weight_version=args.updated_weight_version,
        )

        _print_json("update_response", update_response)

        actual_weight_version = runtime.get_weight_version()
        print(f"updated_weight_version={actual_weight_version!r}")
        if actual_weight_version != args.updated_weight_version:
            raise RuntimeError(
                "live update completed but weight_version did not match: "
                f"{actual_weight_version!r} != {args.updated_weight_version!r}"
            )

        print(f"SUCCESS: live SGLang weight update verified via {verification_mode}.")

        if args.keep_running:
            print("")
            print(f"Server is still running at {spec.url}")
            print("Press Ctrl-C to stop it.")
            while True:
                time.sleep(1.0)

        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down the runtime.")
        return 130
    except Exception as exc:
        print(f"Weight update test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime.process is not None:
            runtime.stop()
            print("Runtime stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
