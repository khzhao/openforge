# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from openforge.rollout.router.types import RouterSpec
from openforge.rollout.sglang.client import SGLangClient
from openforge.train.server import TrainServerClient

__all__ = ["serve_openforge_router"]

logger = logging.getLogger(__name__)


class _RouterState:
    """In-process state machine for the rollout router service."""

    def __init__(
        self,
        *,
        spec: RouterSpec,
        worker_metadata: Sequence[dict[str, Any]],
    ) -> None:
        self.spec = spec
        self._condition = threading.Condition()
        self._latest_published_train_version = 0
        self._train_client: TrainServerClient | None = None
        self._reconcile_stop = threading.Event()
        self._reconcile_thread: threading.Thread | None = None
        self._worker_metadata = list(worker_metadata)
        self._worker_names = [
            str(worker_spec["worker_name"]) for worker_spec in self._worker_metadata
        ]
        self._worker_indexes = {
            worker_name: index for index, worker_name in enumerate(self._worker_names)
        }
        self._workers_by_name = {
            str(worker_spec["worker_name"]): worker_spec
            for worker_spec in self._worker_metadata
        }
        self._worker_clients = {
            worker_name: SGLangClient(str(worker_spec["worker_url"]))
            for worker_name, worker_spec in self._workers_by_name.items()
        }
        self._worker_versions = {
            worker_name: self._read_worker_version(worker_name)
            for worker_name in self._worker_names
        }
        self._worker_health = dict.fromkeys(self._worker_names, False)
        self._worker_states = dict.fromkeys(self._worker_names, "active")
        self._worker_trajectory_counts = dict.fromkeys(self._worker_names, 0)
        self._trajectory_workers: dict[str, str] = {}
        self._trajectory_expected_versions: dict[str, int] = {}
        self._worker_sync_errors = dict.fromkeys(self._worker_names, None)
        self._last_health_refresh_monotonic = 0.0
        self._last_reconcile_monotonic = time.monotonic()
        self._latest_published_train_version = max(
            self._worker_versions.values(),
            default=0,
        )
        self._refresh_worker_health(force=True)

    def start(self) -> None:
        """Start the reconcile loop."""
        if self._reconcile_thread is not None:
            return
        self._reconcile_stop.clear()
        self._reconcile_thread = threading.Thread(
            target=self._reconcile_loop,
            name=f"{self.spec.router_name}-reconcile",
            daemon=True,
        )
        self._reconcile_thread.start()

    def shutdown(self) -> None:
        """Stop the reconcile loop."""
        self._reconcile_stop.set()
        with self._condition:
            self._condition.notify_all()
        reconcile_thread = self._reconcile_thread
        self._reconcile_thread = None
        if reconcile_thread is not None:
            reconcile_thread.join(timeout=float(self.spec.request_timeout_secs))

    def can_generate(self) -> bool:
        """Return whether any healthy worker can serve requests."""
        self._refresh_worker_health(force=True)
        with self._condition:
            return any(self._worker_health.values())

    def register_train_server(self, train_server_url: str) -> None:
        """Register the train control-plane URL."""
        self._train_client = TrainServerClient(train_server_url)
        with self._condition:
            self._condition.notify_all()

    def receive_policy_version(self, policy_version: int) -> None:
        """Receive the latest stable trainer policy version."""
        if policy_version < 0:
            raise ValueError("policy_version must be >= 0")
        with self._condition:
            if any(
                version > policy_version for version in self._worker_versions.values()
            ):
                raise RuntimeError(
                    "cannot roll rollout workers back to an older policy_version"
                )
            self._latest_published_train_version = policy_version
            self._mark_outdated_workers_draining_locked()
            self._condition.notify_all()

    def handle_generate(
        self,
        sampling_params: dict[str, Any],
        *,
        trajectory_id: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route one generate request."""
        self._refresh_worker_health()
        release_after = trajectory_id is None
        resolved_trajectory_id = trajectory_id or f"traj_ephemeral_{uuid.uuid4().hex}"

        worker_name, expected_version = self._reserve_trajectory(resolved_trajectory_id)
        try:
            payload = self._worker_clients[worker_name].generate(
                sampling_params=sampling_params,
                timeout=self._request_timeout(timeout),
                **kwargs,
            )
            self._assert_payload_version(
                resolved_trajectory_id,
                payload,
                expected_version=expected_version,
            )
            return payload
        finally:
            if release_after:
                self.release_trajectory_bindings([resolved_trajectory_id])

    def handle_generate_batch(
        self,
        *,
        trajectory_ids: Sequence[str] | None = None,
        input_ids: Sequence[Sequence[int]],
        sampling_params: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Route one batched generate request."""
        requests = [[int(token_id) for token_id in ids] for ids in input_ids]
        if not requests:
            return []

        if trajectory_ids is None:
            resolved_trajectory_ids = [
                f"traj_ephemeral_{uuid.uuid4().hex}" for _ in requests
            ]
            release_after = True
        else:
            resolved_trajectory_ids = [
                str(trajectory_id) for trajectory_id in trajectory_ids
            ]
            if len(resolved_trajectory_ids) != len(requests):
                raise ValueError("trajectory_ids must align with input_ids")
            release_after = False

        self._refresh_worker_health()
        assignments = [
            self._reserve_trajectory(trajectory_id)
            for trajectory_id in resolved_trajectory_ids
        ]

        indexes_by_worker: dict[str, list[int]] = {}
        expected_versions: dict[str, int] = {}
        for index, (worker_name, expected_version) in enumerate(assignments):
            indexes_by_worker.setdefault(worker_name, []).append(index)
            expected_versions[resolved_trajectory_ids[index]] = expected_version

        grouped_workers = [
            (
                worker_name,
                self._worker_clients[worker_name],
                indexes,
            )
            for worker_name, indexes in indexes_by_worker.items()
        ]

        outputs: list[dict[str, Any] | None] = [None] * len(resolved_trajectory_ids)
        try:
            with ThreadPoolExecutor(
                max_workers=max(1, len(grouped_workers))
            ) as executor:
                future_map = {
                    executor.submit(
                        client.generate,
                        sampling_params=sampling_params,
                        timeout=self._request_timeout(timeout),
                        input_ids=[requests[index] for index in indexes],
                        **kwargs,
                    ): (worker_name, indexes)
                    for worker_name, client, indexes in grouped_workers
                }
                for future, (worker_name, indexes) in future_map.items():
                    payload_group = future.result()
                    payloads = self._normalize_payloads(
                        payload_group,
                        expected_size=len(indexes),
                        worker_name=worker_name,
                    )
                    for index, payload in zip(indexes, payloads, strict=True):
                        trajectory_id = resolved_trajectory_ids[index]
                        self._assert_payload_version(
                            trajectory_id,
                            payload,
                            expected_version=expected_versions[trajectory_id],
                        )
                        outputs[index] = payload
        finally:
            if release_after:
                self.release_trajectory_bindings(resolved_trajectory_ids)

        return [output for output in outputs if output is not None]

    def release_trajectory_bindings(self, trajectory_ids: Sequence[str]) -> None:
        """Release pinned trajectories."""
        if not trajectory_ids:
            return
        with self._condition:
            for trajectory_id in trajectory_ids:
                worker_name = self._trajectory_workers.pop(str(trajectory_id), None)
                self._trajectory_expected_versions.pop(str(trajectory_id), None)
                if worker_name is None:
                    continue
                self._worker_trajectory_counts[worker_name] -= 1
            self._condition.notify_all()

    def status_payload(self) -> dict[str, Any]:
        """Return router status."""
        self._refresh_worker_health(force=True)
        now = time.monotonic()
        with self._condition:
            min_weight_version = min(self._worker_versions.values(), default=0)
            max_weight_version = max(self._worker_versions.values(), default=0)
            return {
                "heartbeat_age_s": max(0.0, now - self._last_reconcile_monotonic),
                "latest_published_train_version": self._latest_published_train_version,
                "min_weight_version": min_weight_version,
                "max_weight_version": max_weight_version,
                "stale_worker_count": sum(
                    1
                    for version in self._worker_versions.values()
                    if version < self._latest_published_train_version
                ),
                "workers": {
                    worker_name: {
                        "healthy": self._worker_health[worker_name],
                        "state": self._worker_states[worker_name],
                        "weight_version": self._worker_versions[worker_name],
                        "worker_url": self._workers_by_name[worker_name]["worker_url"],
                        "world_size": self._workers_by_name[worker_name]["world_size"],
                        "active_trajectory_count": self._worker_trajectory_counts[
                            worker_name
                        ],
                        "last_sync_error": self._worker_sync_errors[worker_name],
                    }
                    for worker_name in self._worker_names
                },
                "trajectories": {
                    trajectory_id: {
                        "worker_name": worker_name,
                        "expected_version": self._trajectory_expected_versions[
                            trajectory_id
                        ],
                    }
                    for trajectory_id, worker_name in self._trajectory_workers.items()
                },
            }

    def _refresh_worker_health(self, *, force: bool = False) -> None:
        if not force:
            now = time.monotonic()
            if now - self._last_health_refresh_monotonic < min(
                1.0, float(self.spec.health_check_interval_secs)
            ):
                return

        timeout = float(self.spec.health_check_timeout_secs)

        def check_worker(worker_name: str) -> tuple[str, bool]:
            try:
                healthy = self._worker_clients[worker_name].health_generate(
                    timeout=timeout
                )
            except Exception:
                healthy = False
            return worker_name, healthy

        if not self._worker_names:
            health_by_name = {}
        elif len(self._worker_names) == 1:
            health_by_name = dict([check_worker(self._worker_names[0])])
        else:
            with ThreadPoolExecutor(max_workers=len(self._worker_names)) as executor:
                health_by_name = dict(executor.map(check_worker, self._worker_names))

        with self._condition:
            self._last_health_refresh_monotonic = time.monotonic()
            self._worker_health.update(health_by_name)
            self._condition.notify_all()

    def _reserve_trajectory(self, trajectory_id: str) -> tuple[str, int]:
        with self._condition:
            worker_name = self._trajectory_workers.get(trajectory_id)
            if worker_name is not None:
                if self._worker_states[worker_name] == "updating":
                    raise RuntimeError(
                        f"trajectory {trajectory_id} is bound to updating worker {worker_name}"
                    )
                return worker_name, self._trajectory_expected_versions[trajectory_id]

            while True:
                candidates = self._active_worker_candidates_locked()
                if candidates:
                    worker_name = min(
                        candidates,
                        key=lambda name: (
                            self._worker_trajectory_counts[name],
                            self._worker_indexes[name],
                        ),
                    )
                    self._trajectory_workers[trajectory_id] = worker_name
                    self._trajectory_expected_versions[trajectory_id] = (
                        self._worker_versions[worker_name]
                    )
                    self._worker_trajectory_counts[worker_name] += 1
                    return (
                        worker_name,
                        self._trajectory_expected_versions[trajectory_id],
                    )
                self._condition.wait(float(self.spec.health_check_interval_secs))

    def _active_worker_candidates_locked(self) -> list[str]:
        active_workers = [
            worker_name
            for worker_name in self._worker_names
            if self._worker_states[worker_name] == "active"
            and self._worker_health[worker_name]
        ]
        if not active_workers:
            return []
        serving_version = max(
            self._worker_versions[worker_name] for worker_name in active_workers
        )
        return [
            worker_name
            for worker_name in active_workers
            if self._worker_versions[worker_name] == serving_version
        ]

    def _mark_outdated_workers_draining_locked(self) -> None:
        for worker_name in self._worker_names:
            if (
                self._worker_versions[worker_name]
                < self._latest_published_train_version
                and self._worker_states[worker_name] == "active"
            ):
                self._worker_states[worker_name] = "draining"

    def _reconcile_loop(self) -> None:
        while not self._reconcile_stop.is_set():
            self._last_reconcile_monotonic = time.monotonic()
            self._refresh_worker_health(force=True)
            ready_worker_names, target_version = self._take_ready_workers()
            if not ready_worker_names:
                with self._condition:
                    if self._reconcile_stop.is_set():
                        return
                    self._condition.wait(float(self.spec.health_check_interval_secs))
                continue
            self._sync_ready_workers(
                ready_worker_names=ready_worker_names,
                target_version=target_version,
            )

    def _take_ready_workers(self) -> tuple[list[str], int]:
        with self._condition:
            if self._train_client is None:
                return [], self._latest_published_train_version
            target_version = self._latest_published_train_version
            ready_worker_names = [
                worker_name
                for worker_name in self._worker_names
                if self._worker_health[worker_name]
                and self._worker_states[worker_name] == "draining"
                and self._worker_trajectory_counts[worker_name] == 0
                and self._worker_versions[worker_name] < target_version
            ]
            for worker_name in ready_worker_names:
                self._worker_states[worker_name] = "updating"
                self._worker_sync_errors[worker_name] = None
            return ready_worker_names, target_version

    def _sync_ready_workers(
        self,
        *,
        ready_worker_names: Sequence[str],
        target_version: int,
    ) -> None:
        assert self._train_client is not None
        workers = [
            {
                "worker_url": str(
                    self._workers_by_name[worker_name]["worker_url"]
                ).rstrip("/"),
                "world_size": int(self._workers_by_name[worker_name]["world_size"]),
            }
            for worker_name in ready_worker_names
        ]
        try:
            payload = self._train_client.request_weight_sync(
                workers=workers,
                target_version=target_version,
                mode="distributed",
                timeout=max(
                    float(self.spec.request_timeout_secs),
                    float(self.spec.health_check_timeout_secs),
                ),
            )
            results = payload.get("results", [])
            if not isinstance(results, list):
                raise RuntimeError("train server returned invalid sync results")
        except Exception as exc:
            error = str(exc)
            with self._condition:
                for worker_name in ready_worker_names:
                    self._worker_states[worker_name] = "draining"
                    self._worker_sync_errors[worker_name] = error
                self._condition.notify_all()
            logger.warning("rollout worker sync request failed: %s", error)
            return

        results_by_url: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("worker_url"), str):
                results_by_url[result["worker_url"].rstrip("/")] = result

        with self._condition:
            for worker_name in ready_worker_names:
                worker_url = str(
                    self._workers_by_name[worker_name]["worker_url"]
                ).rstrip("/")
                result = results_by_url.get(worker_url)
                if not isinstance(result, dict):
                    self._worker_states[worker_name] = "draining"
                    self._worker_sync_errors[worker_name] = (
                        "missing sync result from train server"
                    )
                    continue

                if bool(result.get("ok")):
                    self._worker_versions[worker_name] = target_version
                    if (
                        self._worker_versions[worker_name]
                        < self._latest_published_train_version
                    ):
                        self._worker_states[worker_name] = "draining"
                    else:
                        self._worker_states[worker_name] = "active"
                    self._worker_sync_errors[worker_name] = None
                else:
                    self._worker_states[worker_name] = "draining"
                    self._worker_sync_errors[worker_name] = str(
                        result.get("error", "unknown sync failure")
                    )
            self._condition.notify_all()

    def _request_timeout(self, timeout: float | None) -> float:
        if timeout is not None:
            return timeout
        return max(30.0, float(self.spec.request_timeout_secs))

    @staticmethod
    def _normalize_payloads(
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        expected_size: int,
        worker_name: str,
    ) -> list[dict[str, Any]]:
        payloads = [payload] if isinstance(payload, dict) else list(payload)
        if len(payloads) != expected_size:
            raise RuntimeError(
                "rollout worker returned mismatched batch size: "
                f"{len(payloads)} != {expected_size} on worker {worker_name}"
            )
        return payloads

    def _assert_payload_version(
        self,
        trajectory_id: str,
        payload: dict[str, Any],
        *,
        expected_version: int,
    ) -> None:
        meta_info = payload.get("meta_info", {})
        if not isinstance(meta_info, dict):
            raise RuntimeError("generate payload missing meta_info")
        weight_version = meta_info.get("weight_version")
        if weight_version is None:
            raise RuntimeError("generate payload missing meta_info.weight_version")
        version = int(weight_version)
        if version != expected_version:
            raise RuntimeError(
                f"trajectory {trajectory_id} expected policy_version={expected_version}, "
                f"got {version}"
            )

    def _read_worker_version(self, worker_name: str) -> int:
        try:
            version = self._worker_clients[worker_name].get_weight_version(
                timeout=float(self.spec.health_check_timeout_secs)
            )
        except Exception:
            return 0
        if version is None:
            return 0
        return int(version)


def serve_openforge_router(
    spec: RouterSpec,
    worker_metadata: Sequence[dict[str, Any]],
) -> None:
    """Serve the rollout router in a dedicated process."""
    state = _RouterState(spec=spec, worker_metadata=worker_metadata)
    app = _create_router_app(state)
    config = uvicorn.Config(
        app,
        host=spec.router_ip,
        port=spec.router_port,
        log_level=spec.log_level or "warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    state.start()
    try:
        server.run()
    finally:
        state.shutdown()


def _create_router_app(state: _RouterState) -> FastAPI:
    app = FastAPI(title="OpenForge Rollout Router")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health_generate")
    async def health_generate() -> dict[str, bool]:
        healthy = await asyncio.to_thread(state.can_generate)
        if not healthy:
            raise HTTPException(status_code=503, detail="no healthy rollout workers")
        return {"ok": True}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return await asyncio.to_thread(state.status_payload)

    @app.post("/register_train_server")
    async def register_train_server(request: Request) -> dict[str, Any]:
        payload = await _request_json_object(request)
        train_server_url = payload.get("train_server_url")
        if not isinstance(train_server_url, str) or not train_server_url.strip():
            raise HTTPException(
                status_code=400,
                detail="register_train_server requires train_server_url as a string",
            )
        normalized_url = train_server_url.strip()
        await asyncio.to_thread(state.register_train_server, normalized_url)
        return {"ok": True, "train_server_url": normalized_url}

    @app.post("/receive_policy_version")
    async def receive_policy_version(request: Request) -> dict[str, Any]:
        payload = await _request_json_object(request)
        policy_version = payload.get("policy_version")
        if not isinstance(policy_version, int):
            raise HTTPException(
                status_code=400,
                detail="receive_policy_version requires policy_version as an integer",
            )
        await asyncio.to_thread(state.receive_policy_version, policy_version)
        return {"ok": True, "policy_version": policy_version}

    @app.post("/generate")
    async def generate(request: Request) -> Any:
        payload = await _request_json_object(request)
        try:
            return await asyncio.to_thread(_dispatch_generate, state, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/release_trajectories")
    async def release_trajectories(request: Request) -> dict[str, Any]:
        payload = await _request_json_object(request)
        trajectory_ids = payload.get("trajectory_ids")
        if not isinstance(trajectory_ids, list):
            raise HTTPException(
                status_code=400,
                detail="release_trajectories requires trajectory_ids as a list",
            )
        await asyncio.to_thread(
            state.release_trajectory_bindings,
            [str(trajectory_id) for trajectory_id in trajectory_ids],
        )
        return {"released": len(trajectory_ids)}

    return app


async def _request_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"invalid JSON payload: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object payload")
    return payload


def _dispatch_generate(
    state: _RouterState,
    payload: dict[str, Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    generate_kwargs = dict(payload)
    sampling_params = generate_kwargs.pop("sampling_params", None)
    if not isinstance(sampling_params, dict):
        raise ValueError("generate requires sampling_params as an object")

    timeout = generate_kwargs.pop("timeout", None)
    timeout_value = None if timeout is None else float(timeout)
    trajectory_ids = generate_kwargs.pop("trajectory_ids", None)
    if trajectory_ids is not None:
        if not isinstance(trajectory_ids, list):
            raise ValueError("trajectory_ids must be a list when provided")
        input_ids = generate_kwargs.pop("input_ids", None)
        if not isinstance(input_ids, list):
            raise ValueError("batched generate requires input_ids as a list")
        return state.handle_generate_batch(
            trajectory_ids=[str(trajectory_id) for trajectory_id in trajectory_ids],
            input_ids=input_ids,
            sampling_params=sampling_params,
            timeout=timeout_value,
            **generate_kwargs,
        )

    input_ids = generate_kwargs.get("input_ids")
    if _looks_like_batched_input_ids(input_ids):
        batched_input_ids = generate_kwargs.pop("input_ids")
        return state.handle_generate_batch(
            input_ids=batched_input_ids,
            sampling_params=sampling_params,
            timeout=timeout_value,
            **generate_kwargs,
        )

    return state.handle_generate(
        sampling_params,
        timeout=timeout_value,
        **generate_kwargs,
    )


def _looks_like_batched_input_ids(input_ids: Any) -> bool:
    return (
        isinstance(input_ids, list)
        and bool(input_ids)
        and all(isinstance(item, list) for item in input_ids)
    )
