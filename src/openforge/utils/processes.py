# Copyright 2026 openforge

from __future__ import annotations

from multiprocessing.process import BaseProcess

__all__ = ["stop_spawned_process"]


def stop_spawned_process(
    process: BaseProcess | None,
    *,
    timeout: float,
) -> None:
    """Stop one multiprocessing child."""
    if process is None:
        return

    if process.is_alive():
        try:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(process.pid, include_parent=True)
        except Exception:
            process.terminate()
        process.join(timeout=timeout)
        if process.is_alive():
            process.kill()
            process.join(timeout=timeout)
