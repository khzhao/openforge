# Copyright 2026 openforge

import json
from http import HTTPStatus
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SGLangServerClient:
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
        for path in ("/model_info", "/get_model_info"):
            try:
                _, payload = self._request("GET", path, timeout=timeout)
            except RuntimeError:
                continue
            if isinstance(payload, dict):
                return payload
            raise RuntimeError(f"sglang {path} did not return a JSON object")
        raise RuntimeError("sglang model info endpoint was not available")

    def get_server_info(self, *, timeout: float = 5.0) -> dict[str, Any]:
        for path in ("/server_info", "/get_server_info"):
            try:
                _, payload = self._request("GET", path, timeout=timeout)
            except RuntimeError:
                continue
            if isinstance(payload, dict):
                return payload
            raise RuntimeError(f"sglang {path} did not return a JSON object")
        raise RuntimeError("sglang server info endpoint was not available")

    def get_weight_version(self, *, timeout: float = 5.0) -> str | None:
        payload = self.get_model_info(timeout=timeout)
        version = payload.get("weight_version")
        return None if version is None else str(version)

    def update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
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
        _, body = self._request(
            "POST",
            "/update_weights_from_distributed",
            payload=payload,
            timeout=timeout,
        )
        if not isinstance(body, dict):
            raise RuntimeError(
                "sglang /update_weights_from_distributed did not return a JSON object"
            )
        return body

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
        _, body = self._request(
            "POST",
            "/update_weights_from_tensor",
            payload=payload,
            timeout=timeout,
        )
        if not isinstance(body, dict):
            raise RuntimeError(
                "sglang /update_weights_from_tensor did not return a JSON object"
            )
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
        data = None
        headers = self._headers(path)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request = Request(
            url=f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
                return response.status, self._decode_body(raw_body)
        except HTTPError as exc:
            raw_body = exc.read().decode("utf-8")
            if not raise_for_status:
                return exc.code, self._decode_body(raw_body)
            raise RuntimeError(
                f"sglang request {method} {path} failed with status {exc.code}: "
                f"{raw_body.strip()}"
            ) from exc
        except URLError as exc:
            raise OSError(f"sglang request {method} {path} failed: {exc}") from exc

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
