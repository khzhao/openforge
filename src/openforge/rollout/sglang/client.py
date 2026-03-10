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
        admin_api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.admin_api_key = admin_api_key

    def health_generate(self, *, timeout: float = 1.0) -> bool:
        try:
            status, _ = self._request(
                "GET",
                "/health_generate",
                timeout=timeout,
                raise_for_status=False,
            )
        except OSError:
            return False
        return status == HTTPStatus.OK

    def flush_cache(self, *, timeout: float = 5.0) -> bool:
        try:
            status, _ = self._request(
                "GET",
                "/flush_cache",
                timeout=timeout,
                raise_for_status=False,
            )
        except OSError:
            return False
        return status == HTTPStatus.OK

    def get_model_info(self, *, timeout: float = 5.0) -> dict[str, Any]:
        _, payload = self._request("GET", "/model_info", timeout=timeout)
        if not isinstance(payload, dict):
            raise RuntimeError("sglang /model_info did not return a JSON object")
        return payload

    def get_server_info(self, *, timeout: float = 5.0) -> dict[str, Any]:
        _, payload = self._request("GET", "/server_info", timeout=timeout)
        if not isinstance(payload, dict):
            raise RuntimeError("sglang /server_info did not return a JSON object")
        return payload

    def get_weight_version(self, *, timeout: float = 5.0) -> str | None:
        payload = self.get_model_info(timeout=timeout)
        version = payload.get("weight_version")
        return None if version is None else str(version)

    def update_weights_from_disk(
        self,
        *,
        model_path: str,
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload = {
            "model_path": model_path,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "abort_all_requests": abort_all_requests,
            "weight_version": weight_version,
        }
        _, body = self._request(
            "POST",
            "/update_weights_from_disk",
            payload=payload,
            timeout=timeout,
        )
        if not isinstance(body, dict):
            raise RuntimeError(
                "sglang /update_weights_from_disk did not return a JSON object"
            )
        return body

    def check_weights(
        self,
        *,
        action: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        _, body = self._request(
            "POST",
            "/weights_checker",
            payload={"action": action},
            timeout=timeout,
        )
        if not isinstance(body, dict):
            raise RuntimeError("sglang /weights_checker did not return a JSON object")
        return body

    def pause_generation(
        self,
        *,
        mode: str = "abort",
        timeout: float = 5.0,
    ) -> Any:
        _, body = self._request(
            "POST",
            "/pause_generation",
            payload={"mode": mode},
            timeout=timeout,
        )
        return body

    def continue_generation(self, *, timeout: float = 5.0) -> Any:
        _, body = self._request(
            "POST",
            "/continue_generation",
            payload={},
            timeout=timeout,
        )
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
        headers = self._headers(path)
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

    def _headers(self, path: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = (
            self.api_key
            if path == "/health_generate"
            else (self.admin_api_key or self.api_key)
        )
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
