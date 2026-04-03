# Copyright 2026 openforge


from pathlib import Path

__all__ = [
    "SUPPORTED_MODELS",
    "is_supported_model",
    "validate_supported_model",
]


SUPPORTED_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def is_supported_model(model_name: str) -> bool:
    """Return whether the model name is allowlisted or points to a local path."""
    return model_name in SUPPORTED_MODELS or Path(model_name).exists()


def validate_supported_model(model_name: str) -> None:
    """Raise if the model name or config is incompatible with OpenForge."""
    if not is_supported_model(model_name):
        raise Exception(f"unsupported model: {model_name}")

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        raise Exception(
            f"failed to load model config for compatibility checks: {model_name}"
        ) from exc

    if bool(getattr(config, "use_sliding_window", False)):
        raise Exception("unsupported model config: use_sliding_window must be false")

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        unsupported_layer_types = sorted(
            {
                str(layer_type)
                for layer_type in layer_types
                if layer_type != "full_attention"
            }
        )
        if unsupported_layer_types:
            raise Exception(
                "unsupported model config: layer_types must all be "
                f"'full_attention', got {unsupported_layer_types}"
            )

    linear_attention_fields = {
        field_name: getattr(config, field_name)
        for field_name in (
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
        if getattr(config, field_name, None) is not None
    }
    if linear_attention_fields:
        raise Exception(
            "unsupported model config: linear-attention fields are not supported "
            f"by OpenForge packing: {sorted(linear_attention_fields)}"
        )
