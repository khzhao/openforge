# Copyright 2026 openforge

from __future__ import annotations

import multiprocessing
from multiprocessing.process import BaseProcess
from typing import Any, Sequence

from openforge.rollout.router.server import serve_openforge_router
from openforge.rollout.router.types import RouterSpec

__all__ = [
    "launch_openforge_router",
]


def launch_openforge_router(
    spec: RouterSpec,
    *,
    worker_metadata: Sequence[dict[str, Any]],
) -> BaseProcess:
    """Start the rollout router process."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=serve_openforge_router,
        args=(spec, list(worker_metadata)),
        name=spec.router_name,
    )
    process.start()
    return process
