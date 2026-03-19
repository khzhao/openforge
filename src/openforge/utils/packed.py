# Copyright 2026 openforge

import base64
import pickle

import torch

from openforge.utils.torch import get_torch_dtype_name

__all__ = [
    "build_tensor_bucket_meta",
    "flatten_tensor_bucket",
    "pack_micro_batch",
    "serialize_tensor_bucket",
]


def flatten_tensor_bucket(bucket: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    """Flatten a tensor bucket into a single tensor."""
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    return FlattenedTensorBucket(named_tensors=bucket).get_flattened_tensor()


def pack_micro_batch(micro_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pack a micro batch into a single tensor."""
    batch_size = int(micro_batch["lengths"].shape[0])
    lengths = micro_batch["lengths"].tolist()
    cu_seqlens = [0]
    tokens = []
    advantages = []
    position_ids = []
    loss_mask = []

    for index in range(batch_size):
        length = int(lengths[index])
        tokens.append(micro_batch["tokens"][index, :length])
        advantages.append(micro_batch["advantages"][index, :length])
        position_ids.append(torch.arange(length, dtype=torch.long))
        cu_seqlens.append(cu_seqlens[-1] + length)

        sample_loss_mask = micro_batch["loss_mask"][index, : max(length - 1, 0)]
        loss_mask.append(sample_loss_mask)
        if index < batch_size - 1:
            loss_mask.append(torch.zeros(1, dtype=sample_loss_mask.dtype))

    return {
        "tokens": torch.cat(tokens),
        "position_ids": torch.cat(position_ids),
        "advantages": torch.cat(advantages),
        "loss_mask": torch.cat(loss_mask),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
    }


def serialize_tensor_bucket(bucket: list[tuple[str, torch.Tensor]]) -> str:
    """Serialize a tensor bucket into a string."""
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    flattened_bucket = FlattenedTensorBucket(named_tensors=bucket)
    payload = {
        "flattened_tensor": flattened_bucket.get_flattened_tensor(),
        "metadata": flattened_bucket.get_metadata(),
    }
    return base64.b64encode(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("utf-8")


def build_tensor_bucket_meta(
    bucket: list[tuple[str, torch.Tensor]],
) -> dict[str, list[object]]:
    """Build a tensor bucket metadata dictionary."""
    return {
        "names": [name for name, _ in bucket],
        "dtypes": [get_torch_dtype_name(tensor.dtype) for _, tensor in bucket],
        "shapes": [list(tensor.shape) for _, tensor in bucket],
    }
