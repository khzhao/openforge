# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import ray
import requests
from fastapi import FastAPI, HTTPException, Request

from openforge.rollout.router.client import RolloutRouterClient
from openforge.rollout.router.types import RouterSpec
from openforge.rollout.router.utils import (
    LaunchedRouterServer,
    launch_openforge_router,
    stop_openforge_router,
)
from openforge.rollout.sglang.client import SGLangClient
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.sglang.types import EngineAddr, EngineSpec
from openforge.train.fsdp2.weight_updater import WeightUpdater

__all__ = ["RolloutRouter"]


class RolloutRouter:
    """HTTP router and routing controller for rollout traffic."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    HEALTHCHECK_WAIT_SECONDS = 1.0
    POLL_INTERVAL_SECONDS = 0.1
    SHUTDOWN_TIMEOUT_SECONDS = 30.0

    def __init__(self) -> None:
        self._launched_server: LaunchedRouterServer | None = None
        self._condition = threading.Condition()
        self._target_policy_version = 0

    def initialize(
        self,
        *,
        spec: RouterSpec,
        engine_workers: Sequence[Engine],
        engine_specs: Sequence[EngineSpec],
        engine_addrs: dict[str, EngineAddr],
    ) -> None:
        self.spec = spec
        self._client = RolloutRouterClient(spec.url)
        self._engine_workers = list(engine_workers)
        self._engine_specs = list(engine_specs)
        self._engine_addrs = dict(engine_addrs)
        self._worker_names = [
            engine_spec.engine_name for engine_spec in self._engine_specs
        ]
        self._worker_indexes = {
            worker_name: index for index, worker_name in enumerate(self._worker_names)
        }
        self._workers_by_name = {
            engine_spec.engine_name: worker
            for engine_spec, worker in zip(
                self._engine_specs,
                self._engine_workers,
                strict=True,
            )
        }
        self._worker_clients = {
            worker_name: SGLangClient(self._engine_addrs[worker_name].url)
            for worker_name in self._worker_names
        }
        self._worker_versions = {
            worker_name: self._read_worker_version(self._workers_by_name[worker_name])
            for worker_name in self._worker_names
        }
        self._worker_health = dict.fromkeys(self._worker_names, True)
        self._worker_states = dict.fromkeys(self._worker_names, "active")
        self._worker_trajectory_counts = dict.fromkeys(self._worker_names, 0)
        self._trajectory_workers: dict[str, str] = {}
        self._trajectory_versions: dict[str, int] = {}
        self._target_policy_version = max(self._worker_versions.values(), default=0)
        self._refresh_worker_health()
        self._app = _create_router_app(self)

    @property
    def url(self) -> str:
        return self.spec.url

    @property
    def engine_workers(self) -> list[Engine]:
        return list(self._engine_workers)

    @property
    def engine_specs(self) -> list[EngineSpec]:
        return list(self._engine_specs)

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        return dict(self._engine_addrs)

    def worker_names(self) -> list[str]:
        return list(self._worker_names)

    def worker_for_name(self, worker_name: str) -> Engine:
        return self._workers_by_name[worker_name]

    def health_status(self) -> dict[str, bool]:
        self._refresh_worker_health()
        with self._condition:
            return dict(self._worker_health)

    def current_policy_version(self) -> int:
        with self._condition:
            return max(self._worker_versions.values(), default=0)

    def trajectory_worker(self, trajectory_id: str) -> str | None:
        with self._condition:
            return self._trajectory_workers.get(str(trajectory_id))

    def trajectory_version(self, trajectory_id: str) -> int | None:
        with self._condition:
            return self._trajectory_versions.get(str(trajectory_id))

    def can_generate(self) -> bool:
        self._refresh_worker_health()
        with self._condition:
            return any(self._worker_health.values())

    def launch(self) -> None:
        if self._launched_server is not None:
            raise RuntimeError(
                f"rollout router {self.spec.router_name} is already running"
            )
        self._launched_server = launch_openforge_router(
            self._app,
            host=self.spec.router_ip,
            port=self.spec.router_port,
            log_level=self.spec.log_level,
            name=self.spec.router_name,
        )
        self.wait_until_ready()

    def shutdown(self) -> None:
        launched_server = self._launched_server
        self._launched_server = None
        stop_openforge_router(
            launched_server,
            timeout=self.SHUTDOWN_TIMEOUT_SECONDS,
            name=self.spec.router_name,
        )

    def is_healthy(self) -> bool:
        return self._client.health(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def generate(
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return self._client.generate(
            sampling_params=sampling_params,
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, self.spec.request_timeout_secs),
            **kwargs,
        )

    def release_trajectories(self, *, trajectory_ids: Sequence[str]) -> dict[str, Any]:
        return self._client.release_trajectories(
            trajectory_ids=trajectory_ids,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )

    def handle_generate(
        self,
        sampling_params: dict[str, Any],
        *,
        trajectory_id: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        if not trajectory_ids:
            return
        with self._condition:
            for trajectory_id in trajectory_ids:
                worker_name = self._trajectory_workers.pop(str(trajectory_id), None)
                self._trajectory_versions.pop(str(trajectory_id), None)
                if worker_name is None:
                    continue
                self._worker_trajectory_counts[worker_name] -= 1
            self._condition.notify_all()

    def sync_weights(
        self,
        *,
        train_group: Any,
        policy_version: int,
        mode: str = "auto",
        bucket_bytes: int = 256 << 20,
    ) -> None:
        if policy_version < 0:
            raise ValueError("policy_version must be >= 0")

        updater = WeightUpdater(train_group, bucket_bytes=bucket_bytes)
        with self._condition:
            if any(
                version > policy_version for version in self._worker_versions.values()
            ):
                raise RuntimeError(
                    "cannot roll rollout workers back to an older policy_version"
                )
            self._target_policy_version = policy_version
            self._mark_outdated_workers_draining_locked()
            self._condition.notify_all()

        while True:
            self._refresh_worker_health()
            with self._condition:
                outdated = [
                    worker_name
                    for worker_name in self._worker_names
                    if self._worker_versions[worker_name] != policy_version
                ]
                if not outdated:
                    return

                unhealthy = [
                    worker_name
                    for worker_name in outdated
                    if not self._worker_health[worker_name]
                ]
                if unhealthy:
                    raise RuntimeError(
                        f"cannot sync unhealthy rollout workers: {sorted(unhealthy)}"
                    )

                ready = [
                    worker_name
                    for worker_name in outdated
                    if self._worker_trajectory_counts[worker_name] == 0
                    and self._worker_states[worker_name] != "updating"
                ]
                if not ready:
                    self._condition.wait(self.HEALTHCHECK_WAIT_SECONDS)
                    continue

                for worker_name in ready:
                    self._worker_states[worker_name] = "updating"
                rollout_workers = [
                    self._workers_by_name[worker_name] for worker_name in ready
                ]

            try:
                updater.sync(
                    rollout_workers,
                    policy_version=policy_version,
                    mode=mode,
                )
            except Exception:
                with self._condition:
                    for worker_name in ready:
                        self._worker_states[worker_name] = "draining"
                    self._condition.notify_all()
                raise

            with self._condition:
                for worker_name in ready:
                    self._worker_versions[worker_name] = policy_version
                    self._worker_states[worker_name] = "active"
                self._condition.notify_all()

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + max(
            self.REQUEST_TIMEOUT_SECONDS,
            self.spec.worker_startup_timeout_secs,
        )
        while time.monotonic() < deadline:
            launched_server = self._launched_server
            if launched_server is None or not launched_server.thread.is_alive():
                raise RuntimeError(
                    f"rollout router {self.spec.router_name} exited before becoming ready"
                )
            try:
                if self._client.health_generate(timeout=self.REQUEST_TIMEOUT_SECONDS):
                    return
            except requests.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout router {self.spec.router_name} did not become ready in time"
        )

    def _refresh_worker_health(self) -> None:
        health_by_name: dict[str, bool] = {}
        for worker_name in self._worker_names:
            try:
                healthy = self._worker_clients[worker_name].health_generate(
                    timeout=self.HEALTHCHECK_WAIT_SECONDS
                )
            except Exception:
                healthy = False
            health_by_name[worker_name] = healthy

        with self._condition:
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
                return worker_name, self._trajectory_versions[trajectory_id]

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
                    self._trajectory_versions[trajectory_id] = self._worker_versions[
                        worker_name
                    ]
                    self._worker_trajectory_counts[worker_name] += 1
                    return worker_name, self._trajectory_versions[trajectory_id]
                self._condition.wait(self.HEALTHCHECK_WAIT_SECONDS)

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
            if self._worker_versions[worker_name] < self._target_policy_version:
                if self._worker_states[worker_name] == "active":
                    self._worker_states[worker_name] = "draining"

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

    @staticmethod
    def _read_worker_version(worker: Engine) -> int:
        try:
            version = ray.get(worker.get_weight_version.remote())
        except Exception:
            return 0
        if version is None:
            return 0
        return int(version)


def _create_router_app(router: RolloutRouter) -> FastAPI:
    app = FastAPI(title="OpenForge Rollout Router")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health_generate")
    async def health_generate() -> dict[str, bool]:
        healthy = await asyncio.to_thread(router.can_generate)
        if not healthy:
            raise HTTPException(status_code=503, detail="no healthy rollout workers")
        return {"ok": True}

    @app.post("/generate")
    async def generate(request: Request) -> Any:
        payload = await _request_json_object(request)
        try:
            return await asyncio.to_thread(_dispatch_generate, router, payload)
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
            router.release_trajectory_bindings,
            [str(trajectory_id) for trajectory_id in trajectory_ids],
        )
        return {"released": len(trajectory_ids)}

    return app


async def _request_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid JSON payload: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object payload")
    return payload


def _dispatch_generate(
    router: RolloutRouter,
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
        return router.handle_generate_batch(
            trajectory_ids=[str(trajectory_id) for trajectory_id in trajectory_ids],
            input_ids=input_ids,
            sampling_params=sampling_params,
            timeout=timeout_value,
            **generate_kwargs,
        )

    input_ids = generate_kwargs.get("input_ids")
    if _looks_like_batched_input_ids(input_ids):
        batched_input_ids = generate_kwargs.pop("input_ids")
        return router.handle_generate_batch(
            input_ids=batched_input_ids,
            sampling_params=sampling_params,
            timeout=timeout_value,
            **generate_kwargs,
        )

    return router.handle_generate(
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
