# Copyright 2026 openforge

from __future__ import annotations

import os
from collections.abc import MutableMapping

__all__ = [
    "NCCL_ENV_DEFAULTS",
    "apply_nccl_env_defaults",
]


NCCL_ENV_DEFAULTS = {
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
}


def apply_nccl_env_defaults(
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set conservative NCCL defaults without overriding explicit user values."""
    target_env = os.environ if env is None else env
    for key, value in NCCL_ENV_DEFAULTS.items():
        target_env.setdefault(key, value)
    return {key: str(target_env[key]) for key in NCCL_ENV_DEFAULTS}
