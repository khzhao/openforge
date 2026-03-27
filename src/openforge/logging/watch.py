# Copyright 2026 openforge

from __future__ import annotations

from typing import Any

__all__ = ["render_status"]


def render_status(payload: dict[str, Any]) -> str:
    """Render the status of a session."""
    session_id = payload.get("session_id") or "-"
    wall_time_s = float(payload.get("wall_time_s", 0.0))
    gateway = payload.get("gateway", {})
    train = payload.get("train", {})
    rollout = payload.get("rollout", {})
    cluster = payload.get("cluster", {})
    latest_update = train.get("latest_update") or {}
    lines = [
        f"session {session_id}  wall_time_s={wall_time_s:.1f}",
        (
            "gateway"
            f"  heartbeat_age_s={float(gateway.get('heartbeat_age_s', 0.0)):.1f}"
            f"  pending_generates={int(gateway.get('pending_generate_count', 0))}"
        ),
        (
            "train"
            f"  active={bool(train.get('active', False))}"
            f"  heartbeat_age_s={_format_float(train.get('heartbeat_age_s'))}"
            f"  last_update_age_s={_format_float(train.get('last_update_age_s'))}"
            f"  global_step={_format_int(train.get('global_step'))}"
            f"  policy_version={_format_int(train.get('policy_version'))}"
            f"  reward_mean={_format_float(latest_update.get('reward_mean'))}"
            f"  grad_norm={_format_float(latest_update.get('grad_norm'))}"
            f"  lr={_format_float(latest_update.get('lr'))}"
        ),
        (
            "rollout"
            f"  heartbeat_age_s={_format_float(rollout.get('heartbeat_age_s'))}"
            f"  published_version={_format_int(rollout.get('latest_published_train_version'))}"
            f"  min_weight_version={_format_int(rollout.get('min_weight_version'))}"
            f"  max_weight_version={_format_int(rollout.get('max_weight_version'))}"
            f"  stale_workers={_format_int(rollout.get('stale_worker_count'))}"
        ),
    ]

    total_resources = cluster.get("total_resources", {})
    available_resources = cluster.get("available_resources", {})
    if isinstance(total_resources, dict) and isinstance(available_resources, dict):
        lines.append(
            "cluster"
            f"  alive_nodes={_format_int(cluster.get('alive_nodes'))}"
            f"  cpu={_format_float(available_resources.get('CPU'))}/{_format_float(total_resources.get('CPU'))}"
            f"  gpu={_format_float(available_resources.get('GPU'))}/{_format_float(total_resources.get('GPU'))}"
        )

    workers = rollout.get("workers", {})
    if isinstance(workers, dict) and workers:
        lines.append("")
        lines.append("workers")
        for worker_name, worker in sorted(workers.items()):
            lines.append(
                f"  {worker_name}"
                f"  healthy={bool(worker.get('healthy', False))}"
                f"  state={worker.get('state', '-')}"
                f"  weight_version={_format_int(worker.get('weight_version'))}"
                f"  active_trajectories={_format_int(worker.get('active_trajectory_count'))}"
            )

    return "\n".join(lines)


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(value))
