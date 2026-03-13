# Copyright 2026 openforge

import threading

from openforge.configs.rollout import RolloutEndpoint
from openforge.rollout.types import RolloutWorkerSpec
from openforge.utils.networking import get_free_port
from openforge.utils.ray import (
    get_current_ray_gpu_ids,
    get_current_ray_node_ip_address,
)

__all__ = ["RolloutWorker"]


class RolloutWorker:
    """Thin Ray-facing wrapper over one SGLang server."""

    def initialize(self, spec: RolloutWorkerSpec) -> RolloutEndpoint:
        from openforge.rollout.sglang.server import SGLangServer

        self.spec = spec
        assigned_gpu_ids = get_current_ray_gpu_ids()
        expected_gpu_count = spec.parallelism.world_size
        if len(assigned_gpu_ids) != expected_gpu_count:
            raise RuntimeError(
                "RolloutWorker expected Ray to assign "
                f"{expected_gpu_count} GPU(s), got {assigned_gpu_ids}"
            )
        self.runtime = SGLangServer(
            name=self._engine_name(),
            host=spec.host,
            port=spec.port,
            model_path=spec.cfg.model.model_name_or_path,
            parallelism=spec.parallelism,
            policy_version=spec.policy_version,
            request_timeout_seconds=10.0,
        )
        self.runtime.start(
            cfg=spec.cfg,
            engine_replica_index=spec.engine_replica_index,
            num_nodes=1,
            node_rank=0,
            dist_init_addr=f"{spec.host}:{self.allocate_port(spec.port + 1)}",
            nccl_port=self.allocate_port(spec.port + 2),
        )
        self.pending_runtime_thread: threading.Thread | None = None
        self.pending_runtime_result: dict[str, object] | None = None
        self.pending_runtime_error: BaseException | None = None
        return self.endpoint()

    def endpoint(self) -> RolloutEndpoint:
        healthy = self.runtime.is_healthy()
        return RolloutEndpoint(
            name=self._engine_name(),
            role=self.spec.role,
            host=self.spec.host,
            port=self.spec.port,
            disaggregation_bootstrap_port=None,
            url=self.runtime.url if healthy else None,
            healthy=healthy,
            policy_version=self.runtime.policy_version,
            model_path=self.runtime.model_path,
        )

    def pause_generation(self, mode: str = "abort") -> None:
        self._assert_success(
            self.runtime.pause_generation(mode=mode),
            context="pause_generation",
        )

    def continue_generation(self) -> None:
        self._assert_success(
            self.runtime.continue_generation(),
            context="continue_generation",
        )

    def get_weight_version(self) -> str | None:
        return self.runtime.get_weight_version()

    def distributed_world_size(self) -> int:
        return self.spec.parallelism.world_size

    def update_weights_from_disk(
        self,
        *,
        model_path: str,
        policy_version: int,
        keep_pause: bool = True,
    ) -> None:
        self._assert_success(
            self.runtime.update_weights_from_disk(
                model_path=model_path,
                weight_version=str(policy_version),
                keep_pause=keep_pause,
            ),
            context="update_weights_from_disk",
        )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        policy_version: int,
        load_format: str = "flattened_bucket",
        flush_cache: bool = False,
    ) -> None:
        self._assert_success(
            self.runtime.update_weights_from_tensor(
                serialized_named_tensors=serialized_named_tensors,
                load_format=load_format,
                flush_cache=flush_cache,
                weight_version=str(policy_version),
            ),
            context="update_weights_from_tensor",
        )

    def begin_init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        self._start_pending_runtime_call(
            context="init_weights_update_group",
            runtime_call=lambda: self.runtime.init_weights_update_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend=backend,
            ),
        )

    def begin_update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        policy_version: int,
        load_format: str | None = None,
        flush_cache: bool = False,
    ) -> None:
        self._start_pending_runtime_call(
            context="update_weights_from_distributed",
            runtime_call=lambda: self.runtime.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                load_format=load_format,
                flush_cache=flush_cache,
                weight_version=str(policy_version),
            ),
        )

    def wait_pending_runtime_call(self) -> None:
        thread = self.pending_runtime_thread
        if thread is None:
            raise RuntimeError("no pending runtime call")
        thread.join()
        self.pending_runtime_thread = None
        if self.pending_runtime_error is not None:
            error = self.pending_runtime_error
            self.pending_runtime_error = None
            raise error
        result = self.pending_runtime_result
        self.pending_runtime_result = None
        assert result is not None
        context = str(result["context"])
        self._assert_success(
            dict(result["response"]),
            context=context,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._assert_success(
            self.runtime.destroy_weights_update_group(group_name=group_name),
            context="destroy_weights_update_group",
        )

    def check_weights(self, *, action: str) -> dict[str, object]:
        return self.runtime.check_weights(action=action)

    def shutdown(self) -> None:
        self.runtime.stop()

    @staticmethod
    def node_ip_address() -> str:
        return get_current_ray_node_ip_address()

    @staticmethod
    def allocate_port(start: int) -> int:
        return get_free_port(start=start)

    def _engine_name(self) -> str:
        return f"{self.spec.group_name}-{self.spec.engine_replica_index}"

    def _start_pending_runtime_call(self, *, context: str, runtime_call) -> None:
        if self.pending_runtime_thread is not None:
            raise RuntimeError("runtime call already in progress")

        self.pending_runtime_result = None
        self.pending_runtime_error = None

        def worker() -> None:
            try:
                response = runtime_call()
                self.pending_runtime_result = {
                    "context": context,
                    "response": response,
                }
            except BaseException as exc:
                self.pending_runtime_error = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.pending_runtime_thread = thread

    @staticmethod
    def _assert_success(response: dict[str, object], *, context: str) -> None:
        if response.get("success") is True or response.get("status") == "ok":
            return
        raise RuntimeError(f"{context} failed: {response}")
