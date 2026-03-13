# Copyright 2026 openforge

from typing import Any

from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str, **kwargs: Any) -> AutoTokenizer:
    """Load a tokenizer from a model name or path."""
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
