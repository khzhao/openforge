# Copyright 2026 openforge

from __future__ import annotations

import time
from typing import Any, Callable
from uuid import uuid4

import httpx

from openforge.gateway.types import ChatCompletionResponse, ModelListResponse

from .common import _AUTO_CONCURRENCY_CAP


class _ActiveSession:
    REQUEST_TIMEOUT_SECONDS = 3600.0
    EXPORT_TIMEOUT_SECONDS = 3600.0
    VALIDATION_WAIT_TIMEOUT_SECONDS = 300.0
    STATUS_POLL_INTERVAL_SECONDS = 0.1
    MAX_CONNECTIONS = _AUTO_CONCURRENCY_CAP
    END_RETRIES = 3
    END_RETRY_DELAY_SECONDS = 0.02

    def __init__(self, gateway_target: tuple[str, int]) -> None:
        host, port = gateway_target
        self._base_url = f"http://{host}:{port}"
        self._http: httpx.Client | None = None
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        assert self._session_id is not None, "no active session"
        return self._session_id

    def __enter__(self) -> _ActiveSession:
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(self.REQUEST_TIMEOUT_SECONDS, connect=10.0),
            limits=httpx.Limits(
                max_connections=self.MAX_CONNECTIONS,
                max_keepalive_connections=self.MAX_CONNECTIONS,
            ),
        )
        response = self.get("/current_session")
        if response.status_code == 404:
            self.__exit__(None, None, None)
            raise AssertionError(
                "no active session recorded; "
                "run `python -m openforge.cli.main session start --runtime-config ...` first"
            )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        self._session_id = str(payload["session_id"])
        return self

    def __exit__(self, *_: object) -> None:
        client = self._http
        self._http = None
        self._session_id = None
        if client is not None:
            client.close()

    def client(
        self,
        *,
        group_id: str | None = None,
        trajectory_id: str | None = None,
        used: bool = False,
        purpose: str = "train",
    ) -> _TrajectoryClient:
        return _TrajectoryClient(
            post=self.post,
            retry_post=self._retry_post,
            session_id=self.session_id,
            trajectory_id=trajectory_id or f"traj_{uuid4().hex}",
            group_id=group_id,
            used=used,
            purpose=purpose,
        )

    def trajectory_groups(
        self,
        *,
        counts: list[int],
        group_ids: list[str | None],
        purpose: str = "train",
    ) -> list[list[_TrajectoryClient]]:
        if len(counts) != len(group_ids):
            raise ValueError("counts must align with group_ids")
        if any(count <= 0 for count in counts):
            raise ValueError("count must be >= 1")
        if not counts:
            return []

        response = self.post(
            "/start_trajectory_groups",
            {
                "session_id": self.session_id,
                "counts": counts,
                "group_ids": group_ids,
                "purpose": purpose,
            },
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        trajectory_id_groups = payload["trajectory_ids"]
        assert isinstance(trajectory_id_groups, list)
        return [
            [
                self.client(
                    group_id=group_id,
                    trajectory_id=str(trajectory_id),
                    used=True,
                    purpose=purpose,
                )
                for trajectory_id in trajectory_ids
            ]
            for trajectory_ids, group_id in zip(
                trajectory_id_groups,
                group_ids,
                strict=True,
            )
        ]

    def end_clients(
        self,
        clients: list["_TrajectoryClient"],
        *,
        rewards: list[float],
    ) -> None:
        if len(clients) != len(rewards):
            raise ValueError("rewards must align with clients")
        if not clients:
            return
        self._retry_post(
            "/end_trajectories",
            {
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
                "final_rewards": rewards,
            },
        )

    def fail_clients(self, clients: list["_TrajectoryClient"]) -> None:
        if not clients:
            return
        self._retry_post(
            "/error_trajectories",
            {
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
            },
        )

    def export_checkpoint(self) -> dict[str, Any]:
        response = self.post(
            "/export_checkpoint",
            {"session_id": self.session_id},
            timeout=self.EXPORT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return payload

    def current_policy_version(self) -> int:
        response = self.get("/current_session")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return int(payload["policy_version"])

    def status(self) -> dict[str, object]:
        response = self.get("/status")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return payload

    def wait_for_rollout_policy_version(
        self,
        *,
        policy_version: int,
        timeout: float = VALIDATION_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            rollout = status.get("rollout", {})
            if isinstance(rollout, dict):
                min_weight_version = int(rollout.get("min_weight_version", 0))
                if min_weight_version >= policy_version:
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for rollout to load policy_version "
                    f"{policy_version}"
                )
            time.sleep(self.STATUS_POLL_INTERVAL_SECONDS)

    def log_validation_update(self, payload: dict[str, object]) -> None:
        self._retry_post(
            "/log_validation",
            {
                "session_id": self.session_id,
                "payload": payload,
            },
        )

    def trajectory_statuses(self, trajectory_ids: list[str]) -> dict[str, str]:
        response = self._retry_post(
            "/trajectory_statuses",
            {
                "session_id": self.session_id,
                "trajectory_ids": trajectory_ids,
            },
        )
        payload = response.json()
        assert isinstance(payload, dict)
        trajectories = payload["trajectories"]
        assert isinstance(trajectories, list)
        return {
            str(trajectory["trajectory_id"]): str(trajectory["status"])
            for trajectory in trajectories
        }

    def agent_client(self, client: "_TrajectoryClient") -> _AgentClient:
        return _AgentClient(self, client)

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> httpx.Response:
        client = self._http_client()
        if timeout is None:
            return client.post(path, json=payload)
        return client.post(path, json=payload, timeout=timeout)

    def get(self, path: str) -> httpx.Response:
        client = self._http_client()
        return client.get(path)

    def _http_client(self) -> httpx.Client:
        assert self._http is not None, "gateway is not connected"
        return self._http

    def _retry_post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(self.END_RETRIES):
            try:
                response = self.post(path, payload)
                response.raise_for_status()
                return response
            except httpx.TransportError:
                if attempt + 1 == self.END_RETRIES:
                    raise
                time.sleep(self.END_RETRY_DELAY_SECONDS)
        raise AssertionError("unreachable")


class _TrajectoryClient:
    """Own one trajectory underneath the gateway's active session."""

    def __init__(
        self,
        *,
        post: Callable[[str, dict[str, Any]], httpx.Response],
        retry_post: Callable[[str, dict[str, Any]], httpx.Response],
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
        used: bool = False,
        purpose: str = "train",
    ) -> None:
        self._post = post
        self._retry_post = retry_post
        self._session_id = session_id
        self._trajectory_id = trajectory_id
        self._group_id = group_id
        self._used = used
        self._purpose = purpose

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def group_id(self) -> str | None:
        return self._group_id

    @property
    def purpose(self) -> str:
        return self._purpose

    def mark_used(self) -> None:
        self._used = True

    def finish(self, reward: float) -> None:
        if not self._used:
            self._start()
        response = self._retry_post(
            "/end_trajectory",
            {
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "final_reward": reward,
            },
        )
        response.raise_for_status()

    def fail(self) -> None:
        if not self._used:
            return
        response = self._retry_post(
            "/error_trajectory",
            {
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
            },
        )
        response.raise_for_status()

    def _start(self) -> None:
        response = self._post(
            "/start_trajectory",
            {
                "session_id": self._session_id,
                "group_id": self._group_id,
                "purpose": self._purpose,
            },
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        self._trajectory_id = str(payload["trajectory_id"])
        self._used = True


class _ChatCompletionsClient:
    def __init__(
        self,
        session: _ActiveSession,
        trajectory: _TrajectoryClient,
    ) -> None:
        self._session = session
        self._trajectory = trajectory

    def create(self, **kwargs: Any) -> ChatCompletionResponse:
        self._trajectory.mark_used()
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None:
            extra_body = {}
        else:
            extra_body = dict(extra_body)
        payload = dict(kwargs)
        payload.update(extra_body)
        payload["_openforge"] = {
            "session_id": self._trajectory.session_id,
            "trajectory_id": self._trajectory.trajectory_id,
            "group_id": self._trajectory.group_id,
            "purpose": self._trajectory.purpose,
        }
        response = self._session.post("/v1/chat/completions", payload)
        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())


class _ChatClient:
    def __init__(self, session: _ActiveSession, trajectory: _TrajectoryClient) -> None:
        self.completions = _ChatCompletionsClient(session, trajectory)


class _ModelsClient:
    def __init__(self, session: _ActiveSession) -> None:
        self._session = session

    def list(self) -> ModelListResponse:
        response = self._session.get("/v1/models")
        response.raise_for_status()
        return ModelListResponse.model_validate(response.json())


class _AgentClient:
    def __init__(self, session: _ActiveSession, trajectory: _TrajectoryClient) -> None:
        self.chat = _ChatClient(session, trajectory)
        self.models = _ModelsClient(session)

    def close(self) -> None:
        return None
