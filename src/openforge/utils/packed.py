# Copyright 2026 openforge

import torch

from openforge.utils.torch import get_torch_dtype_name

__all__ = [
    "build_tensor_bucket_meta",
    "flatten_tensor_bucket",
    "serialize_tensor_bucket",
]


def flatten_tensor_bucket(bucket: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    """Flatten a tensor bucket into a single tensor."""
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    return FlattenedTensorBucket(named_tensors=bucket).get_flattened_tensor()


def serialize_tensor_bucket(bucket: list[tuple[str, torch.Tensor]]) -> str:
    """Serialize a tensor bucket into a string."""
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    flattened_bucket = FlattenedTensorBucket(named_tensors=bucket)
    payload = {
        "flattened_tensor": flattened_bucket.get_flattened_tensor(),
        "metadata": flattened_bucket.get_metadata(),
    }
    return MultiprocessingSerializer.serialize(payload, output_str=True)


def build_tensor_bucket_meta(
    bucket: list[tuple[str, torch.Tensor]],
) -> dict[str, list[object]]:
    """Build a tensor bucket metadata dictionary."""
    return {
        "names": [name for name, _ in bucket],
        "dtypes": [get_torch_dtype_name(tensor.dtype) for _, tensor in bucket],
        "shapes": [list(tensor.shape) for _, tensor in bucket],
    }
