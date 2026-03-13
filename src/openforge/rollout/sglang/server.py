# Copyright 2026 openforge

import time
from multiprocessing.process import BaseProcess
from typing import Any

from openforge.configs.models import OpenForgeConfig
from openforge.configs.topology import ParallelismConfig

from .client import SGLangClient
from .utils import (
    generate_sglang_server_args,
    kill_sglang_process_tree,
    launch_sglang_process,
)

HEALTHCHECK_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0


class SGLangServer:
    """Managed lifecycle for one SGLang HTTP server."""

    def __init__(
        self,
        *,
        name: str,
        host: str,
        port: int,
        model_path: str,
        parallelism: ParallelismConfig | None = None,
        server_args: dict[str, Any] | None = None,
        enable_memory_saver: bool = False,
        policy_version: int | None = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.model_path = model_path
        self.parallelism = parallelism or ParallelismConfig()
        self.server_args = dict(server_args or {})
        self.enable_memory_saver = enable_memory_saver
        self.policy_version = policy_version
        self.client = SGLangClient(self.url)
        self.request_timeout_seconds = request_timeout_seconds
        self.process: BaseProcess | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(
        self,
        *,
        cfg: OpenForgeConfig,
        engine_replica_index: int,
        num_nodes: int,
        node_rank: int,
        dist_init_addr: str,
        nccl_port: int,
        override_server_args: dict[str, Any] | None = None,
    ) -> None:
        if self.process is not None and self.process.is_alive():
            return
        launch_server_args = dict(self.server_args)
        if override_server_args is not None:
            launch_server_args.update(override_server_args)
        self.process = launch_sglang_process(
            generate_sglang_server_args(
                cfg,
                engine_replica_index,
                model_path=self.model_path,
                host=self.host,
                port=self.port,
                num_nodes=num_nodes,
                node_rank=node_rank,
                dist_init_addr=dist_init_addr,
                nccl_port=nccl_port,
                parallelism_config=self.parallelism,
                enable_memory_saver=self.enable_memory_saver,
                override_server_args=launch_server_args or None,
            )
        )
        self._wait_until_ready()

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            if not kill_sglang_process_tree(process.pid):
                process.terminate()
            process.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        self.process = None

    def is_healthy(self) -> bool:
        process = self.process
        if process is None or not process.is_alive():
            return False
        return self.client.health_generate(timeout=self.request_timeout_seconds)

    def get_server_info(self) -> dict[str, Any]:
        return self.client.get_server_info(timeout=self.request_timeout_seconds)

    def get_model_info(self) -> dict[str, Any]:
        return self.client.get_model_info(timeout=self.request_timeout_seconds)

    def get_weight_version(self) -> str | None:
        return self.client.get_weight_version(timeout=self.request_timeout_seconds)

    def generate(
        self,
        *,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.client.generate(
            payload=payload,
            timeout=self.request_timeout_seconds if timeout is None else timeout,
        )

    def pause_generation(
        self,
        *,
        mode: str = "abort",
    ) -> dict[str, Any]:
        return self.client.pause_generation(
            mode=mode,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def continue_generation(self) -> dict[str, Any]:
        return self.client.continue_generation(
            timeout=max(self.request_timeout_seconds, 30.0),
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
    ) -> dict[str, Any]:
        payload = self.client.update_weights_from_disk(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            is_async=is_async,
            torch_empty_cache=torch_empty_cache,
            keep_pause=keep_pause,
            recapture_cuda_graph=recapture_cuda_graph,
            token_step=token_step,
            timeout=self._weight_update_timeout(),
        )
        self.model_path = model_path
        self._maybe_set_policy_version(weight_version)
        return payload

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        payload = self.client.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            timeout=self._weight_update_timeout(),
        )
        self._maybe_set_policy_version(weight_version)
        return payload

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "weight_update_group",
        *,
        backend: str = "nccl",
    ) -> dict[str, Any]:
        return self.client.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
            timeout=self._weight_update_timeout(),
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
    ) -> dict[str, Any]:
        payload = self.client.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            load_format=load_format,
            timeout=self._weight_update_timeout(),
        )
        self._maybe_set_policy_version(weight_version)
        return payload

    def destroy_weights_update_group(
        self,
        *,
        group_name: str = "weight_update_group",
    ) -> dict[str, Any]:
        return self.client.destroy_weights_update_group(
            group_name=group_name,
            timeout=self._weight_update_timeout(),
        )

    def check_weights(self, *, action: str) -> dict[str, Any]:
        return self.client.check_weights(
            action=action,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def flush_cache(self) -> bool:
        return self.client.flush_cache(timeout=self.request_timeout_seconds)

    def _maybe_set_policy_version(self, weight_version: str | None) -> None:
        if weight_version is None:
            return
        try:
            self.policy_version = int(weight_version)
        except ValueError:
            pass

    def _weight_update_timeout(self) -> float:
        return max(self.request_timeout_seconds, 300.0)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.name} exited before becoming healthy"
                )
            if self.client.health_generate(timeout=self.request_timeout_seconds):
                self._wait_for_flush_cache()
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"rollout engine {self.name} did not become healthy in time")

    def _wait_for_flush_cache(self) -> None:
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.name} exited before cache flush"
                )
            if self.client.flush_cache(timeout=self.request_timeout_seconds):
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"rollout engine {self.name} never acknowledged flush_cache")
