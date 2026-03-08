#!/usr/bin/env python3
# Copyright 2026 openforge

"""Minimal CPU repro of train -> rollout live weight syncing.

This script uses the real SGLang flattened-bucket serializer, but it replaces
the actual SGLang server with a tiny spawned rollout process. That gives us the
smallest local repro of the live weight-sync loop:

1. Build a tiny training model and a tiny rollout model with identical weights.
2. Run one optimizer step on the training model only.
3. Bucketize the updated training state dict.
4. Serialize each bucket with the real SGLang bucket serializer.
5. Send those serialized buckets to a fake rollout process that mirrors
   `update_weights_from_tensor(...)`.
6. Verify the rollout model matches the training model again.

This is intentionally the colocated-style path only. It is a sandbox for the
train -> bucketize -> live apply flow before wiring a real SGLang server or
distributed update groups.
"""

import argparse
import multiprocessing
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection

import torch
import torch.nn.functional as F
from sglang.srt.utils import MultiprocessingSerializer

from openforge.rollout.sglang.serialization import (
    FlattenedTensorBucket,
    serialize_named_tensors_for_sglang,
)

VOCAB_SIZE = 32
HIDDEN_SIZE = 16
MAX_POSITIONS = 16


class TinyLM(torch.nn.Module):
    """Small deterministic LM-like module with stable state-dict names."""

    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        max_positions: int = MAX_POSITIONS,
    ) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = torch.nn.Embedding(max_positions, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)
        return self.lm_head(hidden)


@dataclass(slots=True)
class BucketRecord:
    index: int
    tensor_count: int
    payload_bytes: int
    serialized_blob_count: int
    metadata_names: list[str]
    metadata_dtypes: list[str]
    metadata_shapes: list[list[int]]


def _build_model_kwargs(args: argparse.Namespace) -> dict[str, int]:
    return {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "max_positions": args.max_positions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal CPU repro of train -> rollout live weight syncing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for model init and the training step.",
    )
    parser.add_argument(
        "--buffer-bytes",
        type=int,
        default=1024,
        help="Max raw tensor bytes to accumulate before flushing a bucket.",
    )
    parser.add_argument(
        "--policy-version",
        type=int,
        default=1,
        help="Policy version written into the fake rollout engine.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.25,
        help="Learning rate for the single training step.",
    )
    parser.add_argument(
        "--show-state-dict",
        action="store_true",
        help="Print the ordered state-dict keys before syncing.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help="Vocabulary size used for the toy model.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN_SIZE,
        help="Hidden size used for the toy model.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=MAX_POSITIONS,
        help="Maximum position embedding length used for the toy model.",
    )
    args = parser.parse_args()
    if args.vocab_size < 6:
        raise ValueError("--vocab-size must be >= 6 for the fixed test batch")
    if args.max_positions < 4:
        raise ValueError("--max-positions must be >= 4 for the fixed test batch")
    return args


def _named_leaf_tensors(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor | torch.nn.Parameter]:
    named_tensors: dict[str, torch.Tensor | torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        named_tensors[name] = param
    for name, buffer in model.named_buffers():
        named_tensors[name] = buffer
    return named_tensors


def _clone_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    return [(name, tensor.detach().clone()) for name, tensor in named_tensors]


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _bucketize_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    max_bucket_bytes: int,
) -> list[list[tuple[str, torch.Tensor]]]:
    if max_bucket_bytes <= 0:
        raise ValueError("max_bucket_bytes must be > 0")

    buckets: list[list[tuple[str, torch.Tensor]]] = []
    current_bucket: list[tuple[str, torch.Tensor]] = []
    current_bytes = 0

    for name, tensor in named_tensors:
        tensor_bytes = _tensor_nbytes(tensor)
        if current_bucket and current_bytes + tensor_bytes > max_bucket_bytes:
            buckets.append(current_bucket)
            current_bucket = []
            current_bytes = 0

        current_bucket.append((name, tensor))
        current_bytes += tensor_bytes

    if current_bucket:
        buckets.append(current_bucket)
    return buckets


def _make_input_batch() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    return input_ids, labels


def _run_single_train_step(
    model: torch.nn.Module,
    *,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    input_ids, labels = _make_input_batch()

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _max_output_diff(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    input_ids, _ = _make_input_batch()
    with torch.no_grad():
        logits_a = model_a(input_ids)
        logits_b = model_b(input_ids)
    return float((logits_a - logits_b).abs().max().item())


def _make_bucket_record(
    index: int,
    bucket: list[tuple[str, torch.Tensor]],
    serialized_blobs: list[str],
) -> BucketRecord:
    flattened_bucket = FlattenedTensorBucket(named_tensors=bucket)
    metadata_names: list[str] = []
    metadata_dtypes: list[str] = []
    metadata_shapes: list[list[int]] = []

    for item in flattened_bucket.get_metadata():
        metadata_names.append(str(item.name))
        metadata_dtypes.append(str(item.dtype).replace("torch.", ""))
        metadata_shapes.append(list(item.shape))

    return BucketRecord(
        index=index,
        tensor_count=len(bucket),
        payload_bytes=sum(_tensor_nbytes(tensor) for _, tensor in bucket),
        serialized_blob_count=len(serialized_blobs),
        metadata_names=metadata_names,
        metadata_dtypes=metadata_dtypes,
        metadata_shapes=metadata_shapes,
    )


def _rollout_engine_main(
    conn: Connection,
    *,
    seed: int,
    model_kwargs: dict[str, int],
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyLM(**model_kwargs)

    named_leaves = _named_leaf_tensors(model)
    weight_version: str | None = None

    try:
        while True:
            message = conn.recv()
            command = message["cmd"]

            if command == "shutdown":
                conn.send({"ok": True})
                return

            if command == "get_weight_version":
                conn.send({"ok": True, "weight_version": weight_version})
                continue

            if command == "compute_logits":
                with torch.no_grad():
                    logits = model(message["input_ids"])
                conn.send({"ok": True, "logits": logits})
                continue

            if command != "update_weights_from_tensor":
                conn.send({"ok": False, "error": f"unknown command: {command}"})
                continue

            load_format = message["load_format"]
            requested_weight_version = message["weight_version"]
            if load_format != "flattened_bucket":
                raise ValueError(f"unsupported load_format: {load_format!r}")
            if requested_weight_version is None:
                raise ValueError("weight_version is required")

            for blob in message["serialized_named_tensors"]:
                payload = MultiprocessingSerializer.deserialize(blob)
                bucket = FlattenedTensorBucket(
                    flattened_tensor=payload["flattened_tensor"],
                    metadata=payload["metadata"],
                )
                for name, value in bucket.reconstruct_tensors():
                    target = named_leaves[name]
                    copied = value.to(dtype=target.dtype, device=target.device)
                    if isinstance(target, torch.nn.Parameter):
                        target.data.copy_(copied)
                    else:
                        target.copy_(copied)

            weight_version = requested_weight_version
            conn.send(
                {
                    "ok": True,
                    "weight_version": weight_version,
                    "serialized_blob_count": len(message["serialized_named_tensors"]),
                }
            )
    except BaseException as exc:
        conn.send({"ok": False, "error": repr(exc)})
        raise
    finally:
        conn.close()


class FakeRolloutEngine:
    """Small parent-side client around the spawned rollout process."""

    def __init__(self, *, seed: int, model_kwargs: dict[str, int]) -> None:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._process = ctx.Process(
            target=_rollout_engine_main,
            args=(child_conn,),
            kwargs={
                "seed": seed,
                "model_kwargs": model_kwargs,
            },
        )
        self._process.start()
        child_conn.close()

    def _recv_ok(self) -> dict[str, object]:
        response = self._conn.recv()
        if response.get("ok") is not True:
            raise RuntimeError(f"rollout process error: {response.get('error')}")
        return response

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        weight_version: str | None = None,
    ) -> dict[str, object]:
        self._conn.send(
            {
                "cmd": "update_weights_from_tensor",
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "weight_version": weight_version,
            }
        )
        return self._recv_ok()

    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._conn.send({"cmd": "compute_logits", "input_ids": input_ids})
        return self._recv_ok()["logits"]  # type: ignore[return-value]

    def get_weight_version(self) -> str | None:
        self._conn.send({"cmd": "get_weight_version"})
        return self._recv_ok()["weight_version"]  # type: ignore[return-value]

    def shutdown(self) -> None:
        try:
            self._conn.send({"cmd": "shutdown"})
            self._recv_ok()
        finally:
            self._conn.close()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5.0)


def _max_output_diff_against_rollout(
    train_model: torch.nn.Module,
    rollout_engine: FakeRolloutEngine,
) -> float:
    input_ids, _ = _make_input_batch()
    with torch.no_grad():
        train_logits = train_model(input_ids)
    rollout_logits = rollout_engine.compute_logits(input_ids)
    return float((train_logits - rollout_logits).abs().max().item())


def main() -> None:
    args = parse_args()
    model_kwargs = _build_model_kwargs(args)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        train_model = TinyLM(**model_kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        rollout_model = TinyLM(**model_kwargs)

    if args.show_state_dict:
        print("state_dict keys:")
        for name in train_model.state_dict().keys():
            print(f"  - {name}")

    initial_diff = _max_output_diff(train_model, rollout_model)
    loss = _run_single_train_step(train_model, learning_rate=args.learning_rate)
    pre_sync_diff = _max_output_diff(train_model, rollout_model)

    named_tensors = _clone_named_tensors(train_model.state_dict().items())
    buckets = _bucketize_named_tensors(
        named_tensors,
        max_bucket_bytes=args.buffer_bytes,
    )

    rollout_engine = FakeRolloutEngine(seed=args.seed, model_kwargs=model_kwargs)
    bucket_records: list[BucketRecord] = []
    try:
        for index, bucket in enumerate(buckets):
            serialized_blobs = serialize_named_tensors_for_sglang(bucket)
            bucket_records.append(_make_bucket_record(index, bucket, serialized_blobs))
            rollout_engine.update_weights_from_tensor(
                serialized_named_tensors=serialized_blobs,
                load_format="flattened_bucket",
                weight_version=str(args.policy_version),
            )

        post_sync_diff = _max_output_diff_against_rollout(train_model, rollout_engine)
        rollout_weight_version = rollout_engine.get_weight_version()
    finally:
        rollout_engine.shutdown()

    if rollout_weight_version != str(args.policy_version):
        raise AssertionError(
            f"expected rollout weight_version={args.policy_version}, "
            f"got {rollout_weight_version!r}"
        )
    if pre_sync_diff <= 0.0:
        raise AssertionError("training step did not change the model outputs")
    if post_sync_diff > 1e-6:
        raise AssertionError(
            f"post-sync rollout mismatch too large: max_abs_diff={post_sync_diff}"
        )

    print(f"train_step_loss={loss:.6f}")
    print(f"initial_max_abs_diff={initial_diff:.6f}")
    print(f"pre_sync_max_abs_diff={pre_sync_diff:.6f}")
    print(f"post_sync_max_abs_diff={post_sync_diff:.6f}")
    print(
        "model_shape="
        f"vocab={args.vocab_size} hidden={args.hidden_size} "
        f"max_positions={args.max_positions} "
        f"param_count={sum(parameter.numel() for parameter in train_model.parameters())}"
    )
    print(
        f"policy_version={rollout_weight_version} "
        f"bucket_count={len(bucket_records)}"
    )
    print("bucket_manifest:")
    for record in bucket_records:
        print(
            f"  - bucket={record.index} "
            f"tensor_count={record.tensor_count} "
            f"payload_bytes={record.payload_bytes} "
            f"serialized_blobs={record.serialized_blob_count}"
        )
        print(
            f"    names={record.metadata_names} "
            f"dtypes={record.metadata_dtypes} "
            f"shapes={record.metadata_shapes}"
        )
    print("SUCCESS: rollout weights match training weights after live bucket sync.")


if __name__ == "__main__":
    main()
