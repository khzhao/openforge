# Copyright 2026 openforge

from dataclasses import replace
from typing import Sequence
from uuid import uuid4

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import RolloutEndpoint
from openforge.configs.topology import PlacementStrategy
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.rollout.sglang.router import PDRouterWorker
from openforge.rollout.types import RolloutWorkerSpec
from openforge.rollout.worker import RolloutWorker
from openforge.utils.ray import ray_placement_group_strategy

__all__ = ["RolloutWorkerGroup"]


class RolloutWorkerGroup:
    """Ray actor group that owns the configured rollout engine fleet."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        strategy: PlacementStrategy | str | None = None,
    ) -> None:
        self.cfg = cfg
        self._resolved = cfg.rollout.resolve(cfg.cluster)
        self._placement_groups: list[PlacementGroup] = []
        self._workers: list[ray.actor.ActorHandle] = []
        self._router_worker: ray.actor.ActorHandle | None = None
        self._engine_endpoints: list[RolloutEndpoint] = []
        self._router_endpoint: RolloutEndpoint | None = None

        self._create_workers(strategy=strategy)

    def initialize(self) -> list[RolloutEndpoint]:
        specs = [self._build_worker_spec(engine) for engine in self._resolved.engines]
        self._engine_endpoints = ray.get(
            [
                worker.initialize.remote(spec)
                for worker, spec in zip(self._workers, specs, strict=True)
            ]
        )

        if self.cfg.rollout.engine_topology == "pd":
            self._router_endpoint = self._initialize_router()
            return [self._router_endpoint]

        self._router_endpoint = None
        return self.routable_endpoints()

    def shutdown(self) -> None:
        if self._router_worker is not None:
            ray.get(self._router_worker.shutdown.remote())
            self._router_worker = None
            self._router_endpoint = None

        ray.get([worker.shutdown.remote() for worker in self._workers])
        self._workers.clear()
        self._engine_endpoints.clear()

        for pg in self._placement_groups:
            ray.util.remove_placement_group(pg)
        self._placement_groups.clear()

    def engine_endpoints(self) -> list[RolloutEndpoint]:
        self._engine_endpoints = ray.get(
            [worker.endpoint.remote() for worker in self._workers]
        )
        return self._engine_endpoints

    def routable_endpoints(self) -> list[RolloutEndpoint]:
        if self.cfg.rollout.engine_topology == "pd":
            if self._router_worker is None:
                return []
            assert self._router_endpoint is not None
            self._router_endpoint = ray.get(self._router_worker.endpoint.remote())
            return [self._router_endpoint] if self._router_endpoint.healthy else []

        return [
            endpoint
            for endpoint in self.engine_endpoints()
            if endpoint.role == "regular"
            and endpoint.url is not None
            and endpoint.healthy
        ]

    def load_policy_artifact(
        self,
        artifact: PolicyArtifactRef,
        *,
        engine_ids: Sequence[int] | None = None,
    ) -> list[RolloutEndpoint]:
        target_workers = self._target_workers(engine_ids)
        ray.get(
            [worker.load_policy_artifact.remote(artifact) for worker in target_workers]
        )
        self._refresh_after_update()
        return self.routable_endpoints()

    def open_tensor_update(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
        policy_version: int,
    ) -> TensorUpdateSession:
        assignments = self._target_assignments(
            engine_ids=engine_ids,
            default_mode="tensor",
        )
        session = TensorUpdateSession(
            session_id=f"tensor-{policy_version}-{uuid4().hex}",
            policy_version=policy_version,
            load_format="flattened_bucket",
            engine_ids=[engine.engine_id for _, engine in assignments],
            transport_metadata={
                "rollout_workers": [worker for worker, _ in assignments],
                "rollout_engines": [engine for _, engine in assignments],
            },
        )
        worker_session = replace(session, transport_metadata={})
        ray.get(
            [
                worker.begin_tensor_update.remote(worker_session)
                for worker, _ in assignments
            ]
        )
        return session

    def open_distributed_update(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
        policy_version: int,
    ) -> DistributedUpdateSession:
        assignments = self._target_assignments(
            engine_ids=engine_ids,
            default_mode="distributed",
        )
        counts = [engine.gpus_per_engine for _, engine in assignments]
        rank_offsets: dict[int, int] = {}
        cumulative = 1
        for (_, engine), count in zip(assignments, counts, strict=True):
            rank_offsets[engine.engine_id] = cumulative
            cumulative += count

        master_worker = self._workers[0]
        master_endpoint = self.engine_endpoints()[0]
        master_port = ray.get(master_worker.allocate_port.remote(start=41000))
        session = DistributedUpdateSession(
            session_id=f"distributed-{policy_version}-{uuid4().hex}",
            policy_version=policy_version,
            load_format="flattened_bucket",
            engine_ids=[engine.engine_id for _, engine in assignments],
            master_addr=master_endpoint.host,
            master_port=master_port,
            group_name=f"openforge-rollout-sync-{policy_version}",
            world_size=cumulative,
            backend="nccl"
            if any(engine.gpus_per_engine > 0 for _, engine in assignments)
            else "gloo",
            rank_offsets=rank_offsets,
            transport_metadata={
                "rollout_workers": [worker for worker, _ in assignments],
                "rollout_engines": [engine for _, engine in assignments],
            },
        )
        worker_session = replace(session, transport_metadata={})
        ray.get(
            [
                worker.begin_distributed_update.remote(worker_session)
                for worker, _ in assignments
            ]
        )
        return session

    def commit_update(
        self,
        session: TensorUpdateSession | DistributedUpdateSession,
    ) -> list[RolloutEndpoint]:
        assignments = self._target_assignments_from_session(session)
        if isinstance(session, TensorUpdateSession):
            worker_session = replace(session, transport_metadata={})
            ray.get(
                [
                    worker.finish_tensor_update.remote(worker_session)
                    for worker, _ in assignments
                ]
            )
        else:
            worker_session = replace(session, transport_metadata={})
            ray.get(
                [
                    worker.finish_distributed_update.remote(worker_session)
                    for worker, _ in assignments
                ]
            )
        self._refresh_after_update()
        return self.routable_endpoints()

    def abort_update(
        self,
        session: TensorUpdateSession | DistributedUpdateSession,
    ) -> None:
        assignments = self._target_assignments_from_session(session)
        ray.get(
            [
                worker.abort_update.remote(session_id=session.session_id)
                for worker, _ in assignments
            ]
        )

    def pause_generation(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
        mode: str = "abort",
    ) -> None:
        ray.get(
            [
                worker.pause_generation.remote(mode=mode)
                for worker in self._target_workers(engine_ids)
            ]
        )

    def continue_generation(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
    ) -> None:
        ray.get(
            [
                worker.continue_generation.remote()
                for worker in self._target_workers(engine_ids)
            ]
        )

    def _create_workers(
        self,
        *,
        strategy: PlacementStrategy | str | None,
    ) -> None:
        remote_worker = ray.remote(RolloutWorker)
        resolved_by_group: dict[str, list] = {}
        for engine in self._resolved.engines:
            resolved_by_group.setdefault(engine.group_name, []).append(engine)

        for group in self.cfg.rollout.engines:
            group_strategy = strategy or group.placement.strategy

            bundles = [
                {"GPU": group.gpus_per_engine, "CPU": group.cpus_per_engine}
            ] * group.replicas
            pg = placement_group(
                bundles,
                strategy=ray_placement_group_strategy(group_strategy),
            )
            ray.get(pg.ready())
            self._placement_groups.append(pg)

            resolved_engines = sorted(
                resolved_by_group[group.name],
                key=lambda engine: engine.replica_index,
            )
            for bundle_index, _engine in enumerate(resolved_engines):
                worker = remote_worker.options(
                    num_gpus=group.gpus_per_engine,
                    num_cpus=group.cpus_per_engine,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_index,
                    ),
                ).remote()
                self._workers.append(worker)

    def _build_worker_spec(self, engine) -> RolloutWorkerSpec:
        host = ray.get(self._workers[engine.engine_id].resolve_node_ip_address.remote())
        port = ray.get(
            self._workers[engine.engine_id].allocate_port.remote(
                start=30000 + (engine.engine_id * 10)
            )
        )
        bootstrap_port = None
        if self.cfg.rollout.engine_topology == "pd" and engine.role == "prefill":
            bootstrap_port = ray.get(
                self._workers[engine.engine_id].allocate_port.remote(start=port + 1)
            )
        return RolloutWorkerSpec(
            cfg=self.cfg,
            engine=engine,
            host=host,
            port=port,
            bootstrap_port=bootstrap_port,
            model_path=self.cfg.model.model_name_or_path,
            policy_version=None,
        )

    def _target_assignments(
        self,
        *,
        engine_ids: Sequence[int] | None,
        default_mode: str,
    ) -> list[tuple[ray.actor.ActorHandle, object]]:
        if engine_ids is not None:
            engine_id_set = set(engine_ids)
            assignments = [
                (worker, engine)
                for worker, engine in zip(
                    self._workers, self._resolved.engines, strict=True
                )
                if engine.engine_id in engine_id_set
            ]
        elif default_mode == "tensor":
            assignments = [
                (worker, engine)
                for worker, engine in zip(
                    self._workers, self._resolved.engines, strict=True
                )
                if engine.placement.node_pool == self.cfg.train.placement.node_pool
            ]
        else:
            assignments = [
                (worker, engine)
                for worker, engine in zip(
                    self._workers, self._resolved.engines, strict=True
                )
                if engine.placement.node_pool != self.cfg.train.placement.node_pool
            ]

        if not assignments:
            raise ValueError(f"no rollout engines selected for {default_mode} update")
        return assignments

    def _target_assignments_from_session(
        self,
        session: TensorUpdateSession | DistributedUpdateSession,
    ) -> list[tuple[ray.actor.ActorHandle, object]]:
        engine_id_set = set(session.engine_ids)
        return [
            (worker, engine)
            for worker, engine in zip(
                self._workers, self._resolved.engines, strict=True
            )
            if engine.engine_id in engine_id_set
        ]

    def _target_workers(
        self,
        engine_ids: Sequence[int] | None,
    ) -> list[ray.actor.ActorHandle]:
        if engine_ids is None:
            return list(self._workers)
        engine_id_set = set(engine_ids)
        return [
            worker
            for worker, engine in zip(
                self._workers, self._resolved.engines, strict=True
            )
            if engine.engine_id in engine_id_set
        ]

    def _initialize_router(self) -> RolloutEndpoint:
        engine_endpoints = self.engine_endpoints()
        prefill_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "prefill"
        ]
        decode_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "decode"
        ]

        remote_router = ray.remote(PDRouterWorker)
        self._router_worker = remote_router.options(num_cpus=0).remote()
        return ray.get(
            self._router_worker.initialize.remote(
                prefill_endpoints=prefill_endpoints,
                decode_endpoints=decode_endpoints,
                checkpoints_dir=self.cfg.train.checkpoints_dir,
            )
        )

    def _restart_router(self) -> RolloutEndpoint:
        if self._router_worker is None:
            return self._initialize_router()

        engine_endpoints = self.engine_endpoints()
        prefill_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "prefill"
        ]
        decode_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "decode"
        ]
        return ray.get(
            self._router_worker.restart.remote(
                prefill_endpoints=prefill_endpoints,
                decode_endpoints=decode_endpoints,
            )
        )

    def _refresh_after_update(self) -> None:
        self._engine_endpoints = self.engine_endpoints()
        if self.cfg.rollout.engine_topology == "pd":
            self._router_endpoint = self._restart_router()
