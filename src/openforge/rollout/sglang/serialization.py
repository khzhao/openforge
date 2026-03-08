# Copyright 2026 openforge

from collections import defaultdict
from collections.abc import Iterable

import torch
from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket


def bucket_named_tensors_for_sglang(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> list[list[tuple[str, torch.Tensor]]]:
    """Group named tensors into SGLang-compatible flattened buckets."""
    named_tensors = list(named_tensors)
    if not named_tensors:
        return []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        return [named_tensors]

    named_tensors_by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = (
        defaultdict(list)
    )
    for name, tensor in named_tensors:
        named_tensors_by_dtype[tensor.dtype].append((name, tensor))
    return list(named_tensors_by_dtype.values())


def serialize_named_tensors_for_sglang(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> list[str]:
    """Serialize HF-format named tensors into SGLang flattened buckets."""
    serialized_buckets: list[str] = []
    for group in bucket_named_tensors_for_sglang(named_tensors):
        flattened_bucket = FlattenedTensorBucket(named_tensors=group)
        flattened_tensor_data = {
            "flattened_tensor": flattened_bucket.get_flattened_tensor(),
            "metadata": flattened_bucket.get_metadata(),
        }
        serialized_buckets.append(
            MultiprocessingSerializer.serialize(
                flattened_tensor_data,
                output_str=True,
            )
        )
    return serialized_buckets
