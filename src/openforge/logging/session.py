# Copyright 2026 openforge

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch

from openforge.gateway.types import RuntimeConfig

if TYPE_CHECKING:
    from openforge.data import Trajectory, Turn
    from openforge.gateway.runtime import Generation

__all__ = ["SessionLogger", "build_train_update"]


def build_train_update(
    *,
    train_results: list[list[object]],
    trajectories: list[Trajectory],
    turns_by_trajectory_id: dict[str, list[Turn]],
    step_time_seconds: float,
    global_step: int,
    policy_version: int,
) -> dict[str, object]:
    """Aggregate one training step into a logging payload."""
    metrics_by_name: dict[str, list[float]] = {}
    for rank_results in train_results:
        for result in rank_results:
            metrics = getattr(result, "metrics", {})
            for key, value in metrics.items():
                metrics_by_name.setdefault(str(key), []).append(float(value))
    aggregated_metrics = {
        key: sum(values) / len(values)
        for key, values in metrics_by_name.items()
        if values
    }

    rewards = [float(trajectory.final_reward) for trajectory in trajectories]
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    turns = [
        turn
        for trajectory in trajectories
        for turn in turns_by_trajectory_id[trajectory.trajectory_id]
    ]
    prompt_tokens = sum(turn.prompt_length for turn in turns)
    completion_tokens = sum(len(turn.token_ids) - turn.prompt_length for turn in turns)
    payload: dict[str, object] = {
        "policy_version": policy_version,
        "global_step": global_step,
        "step_time_s": step_time_seconds,
        "reward_mean": float(reward_tensor.mean()),
        "reward_std": float(reward_tensor.std(unbiased=False)),
        "reward_min": float(reward_tensor.min()),
        "reward_max": float(reward_tensor.max()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "samples_per_second": (
            len(trajectories) / step_time_seconds if step_time_seconds > 0.0 else 0.0
        ),
        "tokens_per_second": (
            (prompt_tokens + completion_tokens) / step_time_seconds
            if step_time_seconds > 0.0
            else 0.0
        ),
    }
    payload.update(aggregated_metrics)
    return payload


class SessionLogger:
    """Session-scoped rollout/train logging with optional W&B integration."""

    DEFAULT_LOG_INTERVAL_SECONDS = 2.0

    def __init__(self) -> None:
        self._run = None
        self._session_id: str | None = None
        self._start_monotonic: float | None = None
        self._last_gateway_heartbeat_monotonic: float | None = None
        self._latest_train_update: dict[str, object] | None = None
        self._latest_rollout_status: dict[str, Any] = {}
        self._latest_cluster_status: dict[str, Any] = {}
        self._pending_generate_count = 0
        self._log_interval_seconds = self.DEFAULT_LOG_INTERVAL_SECONDS
        self._window_started_monotonic = 0.0
        self._last_flush_monotonic = 0.0
        self._rollout_window = self._new_rollout_bucket()
        self._rollout_by_policy: dict[int, dict[str, float]] = defaultdict(
            self._new_rollout_bucket
        )

    def start(self, *, session_id: str, runtime_config: RuntimeConfig) -> None:
        self.finish()
        now = time.monotonic()
        self._session_id = session_id
        self._start_monotonic = now
        self._last_gateway_heartbeat_monotonic = now
        self._window_started_monotonic = now
        self._last_flush_monotonic = now
        self._latest_train_update = None
        self._latest_rollout_status = {}
        self._latest_cluster_status = {}
        self._pending_generate_count = 0
        self._rollout_window = self._new_rollout_bucket()
        self._rollout_by_policy = defaultdict(self._new_rollout_bucket)
        self._log_interval_seconds = self._resolve_log_interval_seconds(
            runtime_config=runtime_config
        )
        self._run = self._start_wandb(
            session_id=session_id,
            runtime_config=runtime_config,
        )

    def record_rollout_batch(
        self,
        *,
        prompt_token_counts: list[int],
        completion_token_counts: list[int],
        policy_versions: list[int],
        latency_seconds: float,
        pending_generate_count: int,
        tokenize_dedupe_hits: int,
    ) -> None:
        if not prompt_token_counts:
            return
        now = time.monotonic()
        self._last_gateway_heartbeat_monotonic = now
        self._pending_generate_count = pending_generate_count
        batch_size = len(prompt_token_counts)
        window = self._rollout_window
        window["batches"] += 1.0
        window["trajectories"] += float(batch_size)
        window["latency_seconds"] += float(latency_seconds)
        window["tokenize_requests"] += float(batch_size)
        window["tokenize_dedupe_hits"] += float(max(0, tokenize_dedupe_hits))

        for prompt_tokens, completion_tokens, policy_version in zip(
            prompt_token_counts,
            completion_token_counts,
            policy_versions,
            strict=True,
        ):
            total_tokens = int(prompt_tokens) + int(completion_tokens)
            window["prompt_tokens"] += float(prompt_tokens)
            window["completion_tokens"] += float(completion_tokens)
            window["total_tokens"] += float(total_tokens)
            bucket = self._rollout_by_policy[int(policy_version)]
            bucket["trajectories"] += 1.0
            bucket["prompt_tokens"] += float(prompt_tokens)
            bucket["completion_tokens"] += float(completion_tokens)
            bucket["total_tokens"] += float(total_tokens)
            bucket["latency_seconds"] += float(latency_seconds) / batch_size

    def record_generations(
        self,
        *,
        input_ids_per_item: list[list[int]],
        generations: list[Generation],
        latency_seconds: float,
        pending_generate_count: int,
        tokenize_dedupe_hits: int,
    ) -> None:
        self.record_rollout_batch(
            prompt_token_counts=[len(input_ids) for input_ids in input_ids_per_item],
            completion_token_counts=[
                len(generation.token_ids) for generation in generations
            ],
            policy_versions=[
                generation.rollout_model_version for generation in generations
            ],
            latency_seconds=latency_seconds,
            pending_generate_count=pending_generate_count,
            tokenize_dedupe_hits=tokenize_dedupe_hits,
        )

    def record_train_update(self, payload: dict[str, object]) -> None:
        self._latest_train_update = dict(payload)
        self._last_gateway_heartbeat_monotonic = time.monotonic()
        if self._run is None:
            return
        policy_version = int(payload["policy_version"])
        metrics = {
            f"train/{key}": value
            for key, value in payload.items()
            if key != "policy_version"
        }
        self._run.log({"policy_version": policy_version, **metrics})

    def record_runtime_status(
        self,
        *,
        rollout_status: dict[str, Any],
        cluster_status: dict[str, Any],
    ) -> None:
        self._latest_rollout_status = dict(rollout_status)
        self._latest_cluster_status = dict(cluster_status)

    def flush(self, *, force: bool = False) -> None:
        if self._run is None:
            if force:
                self._reset_rollout_window()
            return

        now = time.monotonic()
        elapsed = now - self._last_flush_monotonic
        if not force and elapsed < self._log_interval_seconds:
            return

        wall_time_seconds = self.wall_time_seconds(now=now)
        window_elapsed = max(now - self._window_started_monotonic, 1e-6)
        window = self._rollout_window
        runtime_payload = {
            "wall_time_s": wall_time_seconds,
            "rollout_runtime/trajectories_per_sec": (
                window["trajectories"] / window_elapsed
            ),
            "rollout_runtime/prompt_tokens_per_sec": (
                window["prompt_tokens"] / window_elapsed
            ),
            "rollout_runtime/completion_tokens_per_sec": (
                window["completion_tokens"] / window_elapsed
            ),
            "rollout_runtime/total_tokens_per_sec": (
                window["total_tokens"] / window_elapsed
            ),
            "rollout_runtime/batch_latency_ms_mean": (
                1000.0 * window["latency_seconds"] / max(window["batches"], 1.0)
            ),
            "rollout_runtime/tokenize_dedupe_hit_rate": (
                window["tokenize_dedupe_hits"] / max(window["tokenize_requests"], 1.0)
            ),
            "rollout_runtime/pending_generate_count": float(
                self._pending_generate_count
            ),
        }
        self._run.log(runtime_payload)

        status_payload = {
            "wall_time_s": wall_time_seconds,
        }
        rollout_status = self._latest_rollout_status
        if rollout_status:
            status_payload.update(
                {
                    "status/min_weight_version": float(
                        rollout_status.get("min_weight_version", 0)
                    ),
                    "status/max_weight_version": float(
                        rollout_status.get("max_weight_version", 0)
                    ),
                    "status/stale_worker_count": float(
                        rollout_status.get("stale_worker_count", 0)
                    ),
                    "status/rollout_heartbeat_age_s": float(
                        rollout_status.get("heartbeat_age_s", 0.0)
                    ),
                }
            )
        latest_train_update = self._latest_train_update
        if latest_train_update is not None:
            status_payload.update(
                {
                    "status/train_policy_version": float(
                        latest_train_update["policy_version"]
                    ),
                    "status/train_global_step": float(
                        latest_train_update["global_step"]
                    ),
                    "status/train_reward_mean": float(
                        latest_train_update["reward_mean"]
                    ),
                    "status/train_grad_norm": float(
                        latest_train_update.get("grad_norm", 0.0)
                    ),
                    "status/train_lr": float(latest_train_update.get("lr", 0.0)),
                }
            )
        if len(status_payload) > 1:
            self._run.log(status_payload)

        cluster_status = self._latest_cluster_status
        if cluster_status:
            total_resources = cluster_status.get("total_resources", {})
            available_resources = cluster_status.get("available_resources", {})
            self._run.log(
                {
                    "wall_time_s": wall_time_seconds,
                    "status/cluster_total_cpus": float(total_resources.get("CPU", 0.0)),
                    "status/cluster_total_gpus": float(total_resources.get("GPU", 0.0)),
                    "status/cluster_available_cpus": float(
                        available_resources.get("CPU", 0.0)
                    ),
                    "status/cluster_available_gpus": float(
                        available_resources.get("GPU", 0.0)
                    ),
                    "status/cluster_alive_nodes": float(
                        cluster_status.get("alive_nodes", 0)
                    ),
                }
            )

        for policy_version, bucket in sorted(self._rollout_by_policy.items()):
            self._run.log(
                {
                    "policy_version": int(policy_version),
                    "rollout_by_policy/trajectories": bucket["trajectories"],
                    "rollout_by_policy/prompt_tokens": bucket["prompt_tokens"],
                    "rollout_by_policy/completion_tokens": bucket["completion_tokens"],
                    "rollout_by_policy/total_tokens": bucket["total_tokens"],
                    "rollout_by_policy/batch_latency_ms_mean": (
                        1000.0
                        * bucket["latency_seconds"]
                        / max(bucket["trajectories"], 1.0)
                    ),
                }
            )

        self._last_flush_monotonic = now
        self._reset_rollout_window(now=now)

    def snapshot(
        self,
        *,
        train_status: dict[str, Any],
        rollout_status: dict[str, Any],
        cluster_status: dict[str, Any],
    ) -> dict[str, object]:
        now = time.monotonic()
        return {
            "session_id": self._session_id,
            "wall_time_s": self.wall_time_seconds(now=now),
            "gateway": {
                "heartbeat_age_s": self._heartbeat_age_seconds(now=now),
                "pending_generate_count": self._pending_generate_count,
            },
            "train": {
                **train_status,
                "latest_update": self._latest_train_update,
            },
            "rollout": rollout_status,
            "cluster": cluster_status,
        }

    def finish(self) -> None:
        run = self._run
        if run is not None:
            self.flush(force=True)
        self._run = None
        self._session_id = None
        self._start_monotonic = None
        self._last_gateway_heartbeat_monotonic = None
        self._latest_train_update = None
        self._latest_rollout_status = {}
        self._latest_cluster_status = {}
        self._pending_generate_count = 0
        self._reset_rollout_window()
        if run is not None:
            run.finish()

    def wall_time_seconds(self, *, now: float | None = None) -> float:
        if self._start_monotonic is None:
            return 0.0
        resolved_now = time.monotonic() if now is None else now
        return max(0.0, resolved_now - self._start_monotonic)

    def _heartbeat_age_seconds(self, *, now: float) -> float:
        if self._last_gateway_heartbeat_monotonic is None:
            return 0.0
        return max(0.0, now - self._last_gateway_heartbeat_monotonic)

    @staticmethod
    def _new_rollout_bucket() -> dict[str, float]:
        return {
            "batches": 0.0,
            "trajectories": 0.0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "latency_seconds": 0.0,
            "tokenize_requests": 0.0,
            "tokenize_dedupe_hits": 0.0,
        }

    def _reset_rollout_window(self, *, now: float | None = None) -> None:
        resolved_now = time.monotonic() if now is None else now
        self._window_started_monotonic = resolved_now
        self._rollout_window = self._new_rollout_bucket()
        self._rollout_by_policy = defaultdict(self._new_rollout_bucket)

    def _resolve_log_interval_seconds(self, *, runtime_config: RuntimeConfig) -> float:
        wandb_config = runtime_config.wandb
        if wandb_config is not None and wandb_config.log_interval_seconds is not None:
            return float(wandb_config.log_interval_seconds)
        return self.DEFAULT_LOG_INTERVAL_SECONDS

    def _start_wandb(self, *, session_id: str, runtime_config: RuntimeConfig):
        wandb_config = runtime_config.wandb
        if wandb_config is None or not wandb_config.enabled:
            return None
        project = wandb_config.project
        if project is None or not project.strip():
            raise Exception(
                "runtime.wandb.project is required when runtime.wandb.enabled is true"
            )
        import wandb

        entity = wandb_config.entity
        name = wandb_config.name
        if name is None:
            name = (
                f"{str(runtime_config.model.model_name_or_path).split('/')[-1]}-"
                f"{session_id[:8]}"
            )
        tags = [tag.strip() for tag in (wandb_config.tags or []) if tag.strip()] or None

        run = wandb.init(
            project=project.strip(),
            entity=entity or None,
            id=session_id,
            resume="allow",
            name=name,
            tags=tags,
            config={
                "session_id": session_id,
                "runtime": runtime_config.model_dump(mode="json"),
            },
        )
        run.define_metric("policy_version")
        run.define_metric("train/*", step_metric="policy_version")
        run.define_metric("rollout_by_policy/*", step_metric="policy_version")
        run.define_metric("wall_time_s")
        run.define_metric("rollout_runtime/*", step_metric="wall_time_s")
        run.define_metric("status/*", step_metric="wall_time_s")
        return run
