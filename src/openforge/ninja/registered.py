# Copyright 2026 openforge

from __future__ import annotations

import inspect
from functools import update_wrapper
from typing import Any, Callable

from openforge.configs.models import GatewayServerConfig

from .common import (
    _coerce_reward,
    _function_expects_client,
    _normalize_requests,
    _resolve_concurrency,
    _resolve_gateway_target,
    _validate_registered_function,
)
from .execute import _execute_grouped, _execute_many
from .session import _ActiveSession, _TrajectoryClient


class _RegisteredAgent:
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        gateway_config: GatewayServerConfig | None,
    ) -> None:
        _validate_registered_function(func)
        self._func = func
        self._gateway_config = gateway_config
        self._expects_client = _function_expects_client(func)
        self._signature = inspect.signature(func)
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._validate_call(args, kwargs)
        with self._session() as session:
            return self._invoke(session, args, kwargs)

    def sample(
        self,
        *args: Any,
        requests: list[dict[str, Any]] | None = None,
        concurrency: int | None = None,
        group_size: int = 1,
        num_rollouts: int | None = None,
        retries: int = 0,
        **kwargs: Any,
    ) -> Any:
        if num_rollouts is not None:
            if group_size != 1:
                raise ValueError("group_size cannot be combined with num_rollouts")
            group_size = num_rollouts
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if retries < 0:
            raise ValueError("retries must be >= 0")

        call_specs, single_request = _normalize_requests(
            args=args,
            kwargs=kwargs,
            requests=requests,
        )
        for request_index, (call_args, call_kwargs) in enumerate(call_specs):
            self._validate_call(
                call_args,
                call_kwargs,
                request_index=request_index if not single_request else None,
            )
        resolved_concurrency = _resolve_concurrency(
            concurrency=concurrency,
            job_count=len(call_specs) * group_size,
        )
        with self._session() as session:
            if group_size == 1:
                results = _execute_many(
                    self,
                    session,
                    call_specs,
                    concurrency=resolved_concurrency,
                    retries=retries,
                    purpose="train",
                )
            else:
                results = _execute_grouped(
                    self,
                    session,
                    call_specs,
                    group_size=group_size,
                    concurrency=resolved_concurrency,
                    retries=retries,
                    purpose="train",
                )
        if single_request:
            return results[0]
        return results

    def save(self) -> dict[str, Any]:
        with self._session() as session:
            return session.export_checkpoint()

    def _call_body(
        self,
        session: _ActiveSession,
        client: _TrajectoryClient,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> float:
        return _coerce_reward(
            self._call_func(session, client, call_args, call_kwargs),
            context=f"{self.__name__} return value",
        )

    def _invoke(
        self,
        session: _ActiveSession,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        *,
        purpose: str = "train",
    ) -> Any:
        client = session.client(purpose=purpose)
        try:
            reward = self._call_body(session, client, call_args, call_kwargs)
        except Exception:
            client.fail()
            raise

        client.finish(reward)
        return reward

    def _call_func(
        self,
        session: _ActiveSession,
        client: _TrajectoryClient,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Any:
        if not self._expects_client:
            return self._func(*call_args, **call_kwargs)

        agent_client = session.agent_client(client)
        try:
            return self._func(agent_client, *call_args, **call_kwargs)
        finally:
            agent_client.close()

    def _session(self) -> _ActiveSession:
        return _ActiveSession(_resolve_gateway_target(self._gateway_config))

    def _validate_call(
        self,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        *,
        request_index: int | None = None,
    ) -> None:
        bind_args = call_args
        if self._expects_client:
            bind_args = (object(), *call_args)
        try:
            self._signature.bind(*bind_args, **call_kwargs)
        except TypeError as exc:
            if request_index is None:
                raise TypeError(f"{self.__name__} call is invalid: {exc}") from exc
            raise TypeError(
                f"{self.__name__} request[{request_index}] is invalid: {exc}"
            ) from exc


def agent(
    gateway_config: GatewayServerConfig | None = None,
) -> Callable[[Callable[..., Any]], _RegisteredAgent]:
    """Register a function as an OpenForge agent."""

    def decorator(func: Callable[..., Any]) -> _RegisteredAgent:
        return _RegisteredAgent(
            func=func,
            gateway_config=gateway_config,
        )

    return decorator
