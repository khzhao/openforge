#!/usr/bin/env python3

"""Minimal CPU-only smoke test for the OpenForge SGLang router runtime.

This script:
1. Starts tiny local HTTP stub workers.
2. Builds a ``RouterSpec`` and manages the router through ``openforge.rollout.sglang.router.Router``.
3. Verifies the router becomes healthy.
4. Sends requests through the router and checks that workers handle them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import time
from contextlib import closing
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests

from openforge.rollout.sglang.router import Router
from openforge.rollout.types import RouterSpec


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def wait_for_ok(url: str, *, timeout: float, interval: float = 0.2) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == HTTPStatus.OK:
                return
            last_error = RuntimeError(f"{url} returned {response.status_code}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error!r}")


def terminate_process(process: mp.Process, *, name: str) -> None:
    if not process.is_alive():
        process.join(timeout=1.0)
        return
    process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=5.0)
    if process.exitcode not in (0, -15, -9):
        raise RuntimeError(f"{name} exited with unexpected code {process.exitcode}")


@dataclass
class WorkerConfig:
    worker_id: str
    host: str
    port: int
    model_id: str = "stub-model"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def make_worker_handler(cfg: WorkerConfig) -> type[BaseHTTPRequestHandler]:
    class StubWorkerHandler(BaseHTTPRequestHandler):
        server_version = "StubSGLangWorker/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))

        def _send_json(
            self,
            payload: dict[str, Any] | list[Any],
            *,
            status: int = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/health_generate"}:
                self._send_json({"ok": True, "worker_id": cfg.worker_id})
                return
            if self.path in {"/model_info", "/get_model_info"}:
                self._send_json(
                    {
                        "id": cfg.model_id,
                        "object": "model",
                        "model_path": cfg.model_id,
                        "weight_version": "0",
                    }
                )
                return
            if self.path in {"/server_info", "/get_server_info"}:
                self._send_json(
                    {
                        "worker_id": cfg.worker_id,
                        "model_path": cfg.model_id,
                        "internal_states": [
                            {
                                "last_gen_throughput": 0.0,
                                "avg_spec_accept_length": None,
                            }
                        ],
                    }
                )
                return
            if self.path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": cfg.model_id,
                                "object": "model",
                                "owned_by": "stub",
                            }
                        ],
                    }
                )
                return
            self._send_json(
                {"error": f"unsupported GET path {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/flush_cache":
                self._send_json({"ok": True})
                return

            request = self._read_json()

            if self.path == "/generate":
                prompt = request.get("text", "")
                self._send_json(
                    {
                        "text": f"{cfg.worker_id} handled: {prompt}",
                        "worker_id": cfg.worker_id,
                        "meta_info": {
                            "id": cfg.worker_id,
                            "finish_reason": "stop",
                        },
                    }
                )
                return

            if self.path == "/v1/chat/completions":
                prompt = ""
                messages = request.get("messages", [])
                if messages:
                    prompt = str(messages[-1].get("content", ""))
                self._send_json(
                    {
                        "id": f"chatcmpl-{cfg.worker_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": cfg.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"{cfg.worker_id} handled: {prompt}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                return

            self._send_json(
                {"error": f"unsupported POST path {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )

    return StubWorkerHandler


def run_stub_worker(cfg: WorkerConfig) -> None:
    handler_cls = make_worker_handler(cfg)
    server = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=20)
    parser.add_argument("--request-timeout", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")

    mp.set_start_method("spawn", force=True)

    host = args.host
    router_port = args.router_port or find_free_port()
    worker_ports = [find_free_port() for _ in range(args.num_workers)]

    worker_configs = [
        WorkerConfig(worker_id=f"worker-{index}", host=host, port=port)
        for index, port in enumerate(worker_ports)
    ]
    worker_processes = [
        mp.Process(target=run_stub_worker, args=(cfg,), daemon=True)
        for cfg in worker_configs
    ]

    for process in worker_processes:
        process.start()

    router = Router()
    try:
        for cfg in worker_configs:
            wait_for_ok(f"{cfg.url}/health", timeout=args.startup_timeout)

        router_spec = RouterSpec(
            router_name="cpu-smoke-router",
            router_ip=host,
            router_port=router_port,
            policy="round_robin",
            worker_urls=[],
            request_timeout_secs=args.request_timeout,
            worker_startup_timeout_secs=args.startup_timeout,
            worker_startup_check_interval=1,
            health_check_timeout_secs=1,
            health_check_interval_secs=2,
        )
        router.initialize(router_spec)
        for cfg in worker_configs:
            router.add_worker(cfg.url)

        router.launch()
        if not router.is_healthy():
            raise RuntimeError("router reported unhealthy after launch")

        seen_workers: set[str] = set()
        for request_index in range(len(worker_configs) * 2):
            response = requests.post(
                f"{router.url}/generate",
                json={
                    "text": f"hello from openforge router smoke test #{request_index}",
                    "sampling_params": {"max_new_tokens": 8},
                    "stream": False,
                },
                timeout=5.0,
            )
            response.raise_for_status()
            payload = response.json()
            worker_id = payload.get("worker_id")
            if worker_id not in {cfg.worker_id for cfg in worker_configs}:
                raise RuntimeError(f"Unexpected generate response: {payload!r}")
            seen_workers.add(worker_id)

        models = requests.get(f"{router.url}/v1/models", timeout=5.0)
        models.raise_for_status()

        if len(worker_configs) > 1 and len(seen_workers) < 2:
            raise RuntimeError(
                f"round_robin smoke test only saw one worker: {sorted(seen_workers)}"
            )

        print(
            "SUCCESS "
            f"router={router.url} "
            f"workers={len(worker_configs)} "
            f"seen_workers={sorted(seen_workers)}"
        )
        return 0
    finally:
        router.shutdown()
        for index, process in enumerate(worker_processes):
            terminate_process(process, name=f"worker-{index}")


if __name__ == "__main__":
    raise SystemExit(main())
