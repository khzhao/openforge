# Copyright 2026 openforge

import json
from http import HTTPStatus
from typing import Any

import requests


class SGLangControlClient:
    """Small client for SGLang server control and metadata endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def health_generate(self, *, timeout: float = 1.0) -> bool:
        return self._ok("GET", "/health_generate", timeout=timeout)

    def flush_cache(self, *, timeout: float = 5.0) -> bool:
        return self._ok("GET", "/flush_cache", timeout=timeout)

    def get_model_info(self, *, timeout: float = 5.0) -> dict[str, Any]:
        return self._request_json_dict("GET", "/model_info", timeout=timeout)

    def get_server_info(self, *, timeout: float = 5.0) -> dict[str, Any]:
        return self._request_json_dict("GET", "/server_info", timeout=timeout)

    def get_weight_version(self, *, timeout: float = 5.0) -> str | None:
        payload = self.get_model_info(timeout=timeout)
        version = payload.get("weight_version")
        return None if version is None else str(version)

    def generate(
        self,
        *,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._request_json_dict(
            "POST",
            "/generate",
            payload=payload,
            timeout=timeout,
        )

    def pause_generation(
        self,
        *,
        mode: str = "abort",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._request_json_dict(
            "POST",
            "/pause_generation",
            payload={"mode": mode},
            timeout=timeout,
        )

    def continue_generation(
        self,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._request_json_dict(
            "POST",
            "/continue_generation",
            payload={},
            timeout=timeout,
        )

    def update_weights_from_disk(
        self,
        *,
        model_path: str,
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
        is_async: bool = False,
        torch_empty_cache: bool = False,
        keep_pause: bool = False,
        recapture_cuda_graph: bool = False,
        token_step: int = 0,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = {
            "model_path": model_path,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
            "is_async": is_async,
            "torch_empty_cache": torch_empty_cache,
            "keep_pause": keep_pause,
            "recapture_cuda_graph": recapture_cuda_graph,
            "token_step": token_step,
        }
        return self._request_json_dict(
            "POST",
            "/update_weights_from_disk",
            payload=payload,
            timeout=timeout,
        )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
        }
        return self._request_json_dict(
            "POST",
            "/update_weights_from_tensor",
            payload=payload,
            timeout=timeout,
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "weight_update_group",
        backend: str = "nccl",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        }
        return self._request_json_dict(
            "POST",
            "/init_weights_update_group",
            payload=payload,
            timeout=timeout,
        )

    def update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
        load_format: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
            "load_format": load_format,
        }
        return self._request_json_dict(
            "POST",
            "/update_weights_from_distributed",
            payload=payload,
            timeout=timeout,
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str = "weight_update_group",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._request_json_dict(
            "POST",
            "/destroy_weights_update_group",
            payload={"group_name": group_name},
            timeout=timeout,
        )

    def check_weights(
        self,
        *,
        action: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._request_json_dict(
            "POST",
            "/weights_checker",
            payload={"action": action},
            timeout=timeout,
        )

    def _ok(self, method: str, path: str, *, timeout: float) -> bool:
        try:
            status, _ = self._request(
                method,
                path,
                timeout=timeout,
                raise_for_status=False,
            )
        except OSError:
            return False
        return status == HTTPStatus.OK

    def _request_json_dict(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        _, body = self._request(method, path, payload=payload, timeout=timeout)
        if not isinstance(body, dict):
            raise RuntimeError(f"sglang {path} did not return a JSON object")
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
        raise_for_status: bool = True,
    ) -> tuple[int, Any]:
        headers = self._headers()
        request_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": timeout,
        }
        if payload is not None:
            request_kwargs["json"] = payload
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                **request_kwargs,
            )
        except requests.RequestException as exc:
            raise OSError(f"sglang request {method} {path} failed: {exc}") from exc

        raw_body = response.text
        if raise_for_status:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"sglang request {method} {path} failed with status "
                    f"{response.status_code}: {raw_body.strip()}"
                ) from exc
        return response.status_code, self._decode_body(raw_body)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = self.api_key
        if token is None:
            return headers
        headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _decode_body(raw_body: str) -> Any:
        if not raw_body:
            return None
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body
