#!/usr/bin/env python3

"""Minimal CPU-only smoke test for creating an SGLang router.

This script does not use any OpenForge code. It:
1. Starts one or more tiny local HTTP stub workers.
2. Launches ``sglang_router`` via its Python API.
3. Verifies the router becomes healthy.
4. Sends a small ``/generate`` request through the router.

The worker stubs implement only the small subset of endpoints needed for a
regular-mode router smoke test.
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
from sglang_router.launch_router import launch_router
from sglang_router.router_args import RouterArgs


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


def run_router(router_args: RouterArgs, error_queue: mp.Queue[str]) -> None:
    try:
        launch_router(router_args)
    except Exception as exc:  # noqa: BLE001
        error_queue.put(repr(exc))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
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

    router_process: mp.Process | None = None
    try:
        for cfg in worker_configs:
            wait_for_ok(
                f"http://{cfg.host}:{cfg.port}/health",
                timeout=args.startup_timeout,
            )

        router_args = RouterArgs(
            worker_urls=[f"http://{cfg.host}:{cfg.port}" for cfg in worker_configs],
            host=host,
            port=router_port,
            policy="round_robin",
            request_timeout_secs=10,
            worker_startup_timeout_secs=10,
            worker_startup_check_interval=1,
            health_check_timeout_secs=1,
            health_check_interval_secs=2,
        )

        error_queue: mp.Queue[str] = mp.Queue()
        router_process = mp.Process(
            target=run_router,
            args=(router_args, error_queue),
            daemon=True,
        )
        router_process.start()

        wait_for_ok(
            f"http://{host}:{router_port}/health",
            timeout=args.startup_timeout,
        )

        response = requests.post(
            f"http://{host}:{router_port}/generate",
            json={
                "text": "hello from cpu smoke test",
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

        models = requests.get(
            f"http://{host}:{router_port}/v1/models",
            timeout=5.0,
        )
        models.raise_for_status()

        print(
            "SUCCESS "
            f"router=http://{host}:{router_port} "
            f"workers={len(worker_configs)} "
            f"handled_by={worker_id}"
        )
        return 0
    except Exception:
        if router_process is not None and not router_process.is_alive():
            try:
                child_error = error_queue.get_nowait()
            except Exception:  # noqa: BLE001
                child_error = None
            if child_error is not None:
                raise RuntimeError(f"Router process failed: {child_error}") from None
        raise
    finally:
        if router_process is not None:
            terminate_process(router_process, name="router")
        for index, process in enumerate(worker_processes):
            terminate_process(process, name=f"worker-{index}")


if __name__ == "__main__":
    raise SystemExit(main())
