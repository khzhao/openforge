# Copyright 2026 openforge

from typing import Any

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)


def load_tokenizer(model_name_or_path: str, **kwargs: Any) -> AutoTokenizer:
    """Load a tokenizer from a model name or path."""
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def load_processor(model_name_or_path: str, **kwargs: Any) -> AutoProcessor:
    """Load a processor from a model name or path."""
    processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
    if isinstance(processor, PreTrainedTokenizerBase) or not isinstance(
        processor, ProcessorMixin
    ):
        processor = None
    return processor
