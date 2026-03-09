#!/usr/bin/env python3
# Copyright 2026 openforge

"""Focused unit tests for rollout SGLang modules."""

import argparse
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

import requests
import torch

import openforge.rollout.sglang.client as client_module
import openforge.rollout.sglang.engine_runtime as engine_runtime_module
import openforge.rollout.sglang.router as router_module
import openforge.rollout.sglang.runtime as runtime_module
import openforge.rollout.sglang.serialization as serialization_module
from openforge.configs.rollout import RolloutEndpoint
from openforge.policy.types import DistributedUpdateSession, PolicyArtifactRef, TensorUpdateSession
from openforge.rollout.runtime import RolloutRuntime
from openforge.rollout.sglang.client import SGLangControlClient
from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime
from openforge.rollout.sglang.router import PDRouterRuntime, PDRouterWorker
from openforge.rollout.sglang.runtime import SGLangRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from script_test_utils import assert_raises, run_named_tests


def make_spec(
    *,
    role: str = "regular",
    port: int = 31000,
    model_path: str = "/tmp/model",
    policy_version: int | None = None,
) -> SGLangEngineSpec:
    return SGLangEngineSpec(
        engine_id=0,
        name=f"{role}-0",
        role=role,
        host="127.0.0.1",
        port=port,
        bootstrap_port=port + 1 if role == "prefill" else None,
        model_path=model_path,
        tokenizer_path="/tmp/tokenizer",
        policy_version=policy_version,
        server_args={
            "host": "0.0.0.0",
            "port": port,
            "model_path": model_path,
            "weight_version": "default" if policy_version is None else str(policy_version),
        },
    )


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        raise_http_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._raise_http_error = raise_http_error

    def raise_for_status(self) -> None:
        if self._raise_http_error:
            raise requests.HTTPError("boom")


class FakeProcess:
    def __init__(self, *, alive: bool = True) -> None:
        self.pid = 12345
        self.alive = alive
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False


class StubbornProcess(FakeProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class FakeEngineRuntime:
    def __init__(self, spec: SGLangEngineSpec) -> None:
        self.spec = spec
        self.started = False
        self.stopped = False
        self.health = True
        self.flush_result = True
        self.request_timeout_seconds = 5.0
        self.pause_calls: list[str] = []
        self.continue_calls = 0
        self.init_group_calls: list[dict[str, object]] = []
        self.destroy_group_calls: list[str] = []
        self.tensor_update_calls: list[dict[str, object]] = []
        self.distributed_update_calls: list[dict[str, object]] = []

        self.client = mock.Mock(
            get_model_info=mock.Mock(return_value={"weight_version": spec.policy_version}),
        )

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def restart(self) -> None:
        self.stopped = True
        self.started = True

    def is_healthy(self) -> bool:
        return self.health

    def get_server_info(self) -> dict[str, object]:
        return {"name": self.spec.name}

    def get_weight_version(self) -> str | None:
        return None if self.spec.policy_version is None else str(self.spec.policy_version)

    def check_weights(self, *, action: str) -> dict[str, object]:
        return {"action": action}

    def flush_cache(self) -> bool:
        return self.flush_result

    def pause_generation(self, *, mode: str = "abort") -> str:
        self.pause_calls.append(mode)
        return mode

    def continue_generation(self) -> None:
        self.continue_calls += 1

    def init_weights_update_group(self, **kwargs) -> dict[str, object]:
        self.init_group_calls.append(kwargs)
        return {"ok": True}

    def destroy_weights_update_group(self, *, group_name: str) -> dict[str, object]:
        self.destroy_group_calls.append(group_name)
        return {"group_name": group_name}

    def update_weights_from_tensor(self, **kwargs) -> dict[str, object]:
        self.tensor_update_calls.append(kwargs)
        return {"updated": True}

    def update_weights_from_distributed(self, **kwargs) -> dict[str, object]:
        self.distributed_update_calls.append(kwargs)
        return {"updated": True}


class FakeFlattenedTensorBucket:
    supports_multi_dtypes = False

    def __init__(self, *, named_tensors):
        self.named_tensors = list(named_tensors)

    def get_flattened_tensor(self):
        return torch.tensor([len(self.named_tensors)], dtype=torch.float32)

    def get_metadata(self):
        return {"names": [name for name, _ in self.named_tensors]}


class FakePopen:
    def __init__(self, *_args, **_kwargs) -> None:
        self.pid = 9876
        self._returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self):
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -9


class FakeRouterRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.start_calls = 0
        self.stop_calls = 0
        self.restart_calls: list[dict[str, object]] = []
        self.endpoint_value = RolloutEndpoint(
            name="pd-router",
            role="router",
            host="127.0.0.1",
            port=39000,
            bootstrap_port=None,
            url="http://127.0.0.1:39000",
            healthy=True,
            policy_version=None,
            model_path=None,
        )

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def restart(self, **kwargs) -> None:
        self.restart_calls.append(kwargs)

    def endpoint(self) -> RolloutEndpoint:
        return self.endpoint_value

    def is_healthy(self) -> bool:
        return True


def test_rollout_runtime_is_abstract() -> None:
    assert_raises(TypeError, RolloutRuntime)


def test_sglang_engine_spec_url_property() -> None:
    assert make_spec(port=32000).url == "http://127.0.0.1:32000"


def test_sglang_control_client_request_and_headers() -> None:
    client = SGLangControlClient(
        "http://127.0.0.1:31000/",
        api_key="user-token",
        admin_api_key="admin-token",
    )

    assert client._headers("/health_generate") == {
        "Authorization": "Bearer user-token"
    }
    assert client._headers("/server_info") == {
        "Authorization": "Bearer admin-token"
    }
    assert client._decode_body("") is None
    assert client._decode_body("{\"ok\": true}") == {"ok": True}
    assert client._decode_body("plain-text") == "plain-text"

    with mock.patch.object(
        client_module.requests,
        "request",
        return_value=FakeResponse(status_code=200, text='{"ok": true}'),
    ) as request:
        status, payload = client._request("GET", "/server_info", timeout=3.0)

    assert status == 200
    assert payload == {"ok": True}
    request.assert_called_once()

    with mock.patch.object(
        client_module.requests,
        "request",
        return_value=FakeResponse(status_code=500, text="bad", raise_http_error=True),
    ):
        assert_raises(
            RuntimeError,
            client._request,
            "GET",
            "/server_info",
            timeout=1.0,
            match="failed with status 500",
        )

    with mock.patch.object(
        client_module.requests,
        "request",
        side_effect=requests.RequestException("network"),
    ):
        assert_raises(
            OSError,
            client._request,
            "GET",
            "/server_info",
            timeout=1.0,
            match="failed",
        )


def test_sglang_control_client_public_methods() -> None:
    client = SGLangControlClient("http://127.0.0.1:31000")

    with mock.patch.object(
        client,
        "_request",
        side_effect=[
            OSError("offline"),
            (503, None),
            (200, {"weight_version": 11}),
            (200, {"name": "server"}),
            (200, {"weight_version": 11}),
        ],
    ):
        assert client.health_generate(timeout=1.0) is False
        assert client.flush_cache(timeout=1.0) is False
        assert client.get_model_info() == {"weight_version": 11}
        assert client.get_server_info() == {"name": "server"}
        assert client.get_weight_version() == "11"


def test_sglang_control_client_payload_methods() -> None:
    test_cases = [
        ("check_weights", {"action": "snapshot"}, "/weights_checker", {"action": "snapshot"}),
        (
            "init_weights_update_group",
            {
                "master_address": "127.0.0.1",
                "master_port": 41000,
                "rank_offset": 1,
                "world_size": 4,
                "group_name": "sync",
                "backend": "nccl",
            },
            "/init_weights_update_group",
            {
                "master_address": "127.0.0.1",
                "master_port": 41000,
                "rank_offset": 1,
                "world_size": 4,
                "group_name": "sync",
                "backend": "nccl",
            },
        ),
        (
            "destroy_weights_update_group",
            {"group_name": "sync"},
            "/destroy_weights_update_group",
            {"group_name": "sync"},
        ),
        (
            "update_weights_from_distributed",
            {
                "names": ["w"],
                "dtypes": ["float16"],
                "shapes": [[2, 2]],
                "group_name": "sync",
                "weight_version": "7",
            },
            "/update_weights_from_distributed",
            {
                "names": ["w"],
                "dtypes": ["float16"],
                "shapes": [[2, 2]],
                "group_name": "sync",
                "flush_cache": True,
                "abort_all_requests": False,
                "weight_version": "7",
                "load_format": None,
            },
        ),
        (
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": ["payload"],
                "weight_version": "8",
            },
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": ["payload"],
                "load_format": None,
                "flush_cache": True,
                "abort_all_requests": False,
                "weight_version": "8",
            },
        ),
    ]

    for method_name, kwargs, expected_path, expected_payload in test_cases:
        client = SGLangControlClient("http://127.0.0.1:31000")
        seen: list[tuple[str, str, dict[str, object] | None]] = []

        def fake_request(method, path, *, payload=None, timeout, raise_for_status=True):
            _ = timeout, raise_for_status
            seen.append((method, path, payload))
            return 200, {"ok": True}

        with mock.patch.object(client, "_request", side_effect=fake_request):
            assert getattr(client, method_name)(**kwargs) == {"ok": True}

        assert seen == [("POST", expected_path, expected_payload)]


def test_sglang_control_client_dict_validations() -> None:
    client = SGLangControlClient("http://127.0.0.1:31000")
    methods = [
        ("check_weights", {"action": "snapshot"}),
        (
            "init_weights_update_group",
            {
                "master_address": "127.0.0.1",
                "master_port": 41000,
                "rank_offset": 1,
                "world_size": 4,
                "group_name": "sync",
                "backend": "nccl",
            },
        ),
        ("destroy_weights_update_group", {"group_name": "sync"}),
        (
            "update_weights_from_distributed",
            {
                "names": ["w"],
                "dtypes": ["float16"],
                "shapes": [[2, 2]],
                "group_name": "sync",
            },
        ),
        ("update_weights_from_tensor", {"serialized_named_tensors": ["payload"]}),
    ]
    for method_name, kwargs in methods:
        with mock.patch.object(client, "_request", return_value=(200, "not-a-dict")):
            assert_raises(
                RuntimeError,
                getattr(client, method_name),
                **kwargs,
                match="did not return a JSON object",
            )


def test_sglang_engine_runtime_lifecycle_and_delegates() -> None:
    spec = make_spec()
    fake_process = FakeProcess()
    process_launcher = mock.Mock(return_value=fake_process)
    runtime = SGLangEngineRuntime(spec, process_launcher=process_launcher)

    with mock.patch.object(runtime, "_wait_until_ready") as wait_until_ready:
        runtime.start()
        runtime.start()

    assert process_launcher.call_count == 1
    wait_until_ready.assert_called_once()

    runtime.client = mock.Mock(
        health_generate=mock.Mock(return_value=True),
        get_server_info=mock.Mock(return_value={"server": "ok"}),
        get_weight_version=mock.Mock(return_value="9"),
        check_weights=mock.Mock(return_value={"action": "snapshot"}),
        flush_cache=mock.Mock(return_value=True),
        pause_generation=mock.Mock(return_value="paused"),
        continue_generation=mock.Mock(return_value="continued"),
        init_weights_update_group=mock.Mock(return_value={"ok": True}),
        destroy_weights_update_group=mock.Mock(return_value={"ok": True}),
        update_weights_from_distributed=mock.Mock(return_value={"distributed": True}),
        update_weights_from_tensor=mock.Mock(return_value={"tensor": True}),
    )

    assert runtime.is_healthy() is True
    assert runtime.get_server_info() == {"server": "ok"}
    assert runtime.get_weight_version() == "9"
    assert runtime.check_weights(action="snapshot") == {"action": "snapshot"}
    assert runtime.flush_cache() is True
    assert runtime.pause_generation(mode="abort") == "paused"
    assert runtime.continue_generation() == "continued"
    assert runtime.init_weights_update_group(
        master_address="127.0.0.1",
        master_port=41000,
        rank_offset=1,
        world_size=4,
        group_name="sync",
        backend="nccl",
    ) == {"ok": True}
    assert runtime.destroy_weights_update_group(group_name="sync") == {"ok": True}
    assert runtime.update_weights_from_distributed(
        names=["w"],
        dtypes=["float16"],
        shapes=[[2, 2]],
        group_name="sync",
        weight_version="5",
    ) == {"distributed": True}
    assert spec.policy_version == 5
    assert runtime.update_weights_from_tensor(
        serialized_named_tensors=["payload"],
        weight_version="6",
    ) == {"tensor": True}
    assert spec.policy_version == 6

    runtime.process = fake_process
    with mock.patch.object(engine_runtime_module, "_kill_process_tree", return_value=False):
        runtime.stop()
    assert runtime.process is None
    assert fake_process.terminate_calls == 1


def test_sglang_engine_runtime_process_launch_helpers() -> None:
    fake_process = mock.Mock()
    fake_context = mock.Mock(
        Process=mock.Mock(return_value=fake_process),
    )

    with mock.patch.object(
        engine_runtime_module.multiprocessing,
        "get_context",
        return_value=fake_context,
    ):
        launched = engine_runtime_module._launch_sglang_process({"host": "0.0.0.0"})

    assert launched is fake_process
    fake_context.Process.assert_called_once()
    fake_process.start.assert_called_once()

    fake_http_server = ModuleType("sglang.srt.entrypoints.http_server")
    fake_http_server.launch_server = mock.Mock()
    fake_server_args = ModuleType("sglang.srt.server_args")
    fake_server_args.ServerArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_utils = ModuleType("sglang.srt.utils")
    fake_utils.kill_process_tree = mock.Mock()

    with (
        mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.entrypoints.http_server": fake_http_server,
                "sglang.srt.server_args": fake_server_args,
                "sglang.srt.utils": fake_utils,
            },
        ),
        mock.patch.object(engine_runtime_module.os, "getpid", return_value=55),
    ):
        engine_runtime_module._serve_sglang({"host": "[::1]", "port": 31000})

    fake_http_server.launch_server.assert_called_once()
    assert fake_http_server.launch_server.call_args.args[0].host == "::1"
    fake_utils.kill_process_tree.assert_called_once_with(55, include_parent=False)

    with mock.patch.dict(sys.modules, {"sglang.srt.utils": fake_utils}):
        assert engine_runtime_module._kill_process_tree(99) is True
    assert fake_utils.kill_process_tree.call_args_list[-1] == mock.call(
        99, include_parent=True
    )


def test_sglang_engine_runtime_wait_helpers_and_validation() -> None:
    spec = make_spec()
    runtime = SGLangEngineRuntime(spec, process_launcher=lambda payload: FakeProcess())
    runtime.process = FakeProcess()
    runtime.client = mock.Mock(
        health_generate=mock.Mock(side_effect=[False, True]),
        flush_cache=mock.Mock(side_effect=[False, True]),
    )

    with mock.patch.object(engine_runtime_module.time, "sleep"):
        runtime._wait_until_ready()

    assert runtime.client.health_generate.call_count == 2
    assert runtime.client.flush_cache.call_count == 2

    runtime.process = FakeProcess(alive=False)
    runtime.client = mock.Mock(health_generate=mock.Mock(return_value=False))
    assert_raises(
        RuntimeError,
        runtime._wait_until_ready,
        match="exited before becoming healthy",
    )

    runtime.process = FakeProcess()
    runtime.client = mock.Mock(flush_cache=mock.Mock(return_value=False))
    with mock.patch.object(engine_runtime_module, "HEALTHCHECK_TIMEOUT_SECONDS", 0.0):
        assert_raises(
            TimeoutError,
            runtime._wait_for_flush_cache,
            match="never acknowledged flush_cache",
        )

    assert_raises(
        ValueError,
        runtime._require_policy_version,
        None,
        match="weight_version is required",
    )
    assert_raises(
        ValueError,
        runtime._require_policy_version,
        "abc",
        match="must be an integer string",
    )

    stubborn = StubbornProcess()
    runtime.process = stubborn
    with mock.patch.object(engine_runtime_module, "_kill_process_tree", return_value=False):
        runtime.stop()
    assert stubborn.kill_calls == 1


def test_sglang_runtime_tensor_and_distributed_updates() -> None:
    spec = make_spec(policy_version=1)
    fake_runtime = FakeEngineRuntime(spec)

    with mock.patch.object(runtime_module, "SGLangEngineRuntime", return_value=fake_runtime):
        runtime = SGLangRuntime(spec)

    tensor_session = TensorUpdateSession(
        session_id="tensor-1",
        policy_version=5,
        load_format="flattened_bucket",
        engine_ids=[0],
    )
    distributed_session = DistributedUpdateSession(
        session_id="dist-1",
        policy_version=6,
        load_format="flattened_bucket",
        engine_ids=[0],
        master_addr="127.0.0.1",
        master_port=41000,
        group_name="sync-group",
        world_size=2,
        backend="nccl",
        rank_offsets={0: 1},
    )

    runtime.start()
    assert runtime.endpoint().policy_version == 1
    assert runtime.is_healthy() is True
    assert runtime.get_model_info() == {"weight_version": 1}
    assert runtime.get_server_info() == {"name": "regular-0"}
    assert runtime.get_weight_version() == "1"

    runtime.load_policy_artifact(
        PolicyArtifactRef(step=3, policy_version=4, path="/tmp/policy.pt")
    )
    assert spec.model_path == "/tmp/policy.pt"
    assert spec.policy_version == 4

    runtime.begin_tensor_update(tensor_session)
    runtime.apply_tensor_bucket(
        serialized_named_tensors=["payload"],
        load_format="flattened_bucket",
        policy_version=5,
    )
    runtime.finish_tensor_update(tensor_session)
    assert fake_runtime.pause_calls == ["abort"]
    assert fake_runtime.continue_calls == 1
    assert fake_runtime.tensor_update_calls[0]["weight_version"] == "5"

    runtime.begin_distributed_update(distributed_session)
    runtime.apply_distributed_bucket(
        bucket=SimpleNamespace(names=["w"], dtypes=["float16"], shapes=[[2, 2]]),
        load_format="flattened_bucket",
        policy_version=6,
        group_name="sync-group",
    )
    runtime.finish_distributed_update(distributed_session)
    assert fake_runtime.init_group_calls[0]["rank_offset"] == 1
    assert fake_runtime.distributed_update_calls[0]["group_name"] == "sync-group"
    assert fake_runtime.destroy_group_calls == ["sync-group"]

    runtime._active_update_group_name = "sync-group-2"
    runtime._active_session_id = "session-2"
    runtime._paused_for_weight_update = True
    runtime.abort_update(session_id="session-2")
    assert fake_runtime.destroy_group_calls[-1] == "sync-group-2"
    assert fake_runtime.continue_calls == 3

    runtime.stop()
    assert fake_runtime.stopped is True


def test_sglang_runtime_aborts_failed_begin_update() -> None:
    spec = make_spec()
    fake_runtime = FakeEngineRuntime(spec)
    fake_runtime.flush_result = False

    with mock.patch.object(runtime_module, "SGLangEngineRuntime", return_value=fake_runtime):
        runtime = SGLangRuntime(spec)

    tensor_session = TensorUpdateSession(
        session_id="tensor-fail",
        policy_version=5,
        load_format="flattened_bucket",
        engine_ids=[0],
    )
    assert_raises(
        RuntimeError,
        runtime.begin_tensor_update,
        tensor_session,
        match="did not flush cache before update",
    )
    assert fake_runtime.continue_calls == 1


def test_sglang_serialization_buckets_and_router_runtime() -> None:
    named_tensors = [
        ("w1", torch.ones(2, dtype=torch.float16)),
        ("w2", torch.ones(2, dtype=torch.float32)),
    ]

    with (
        mock.patch.object(serialization_module, "FlattenedTensorBucket", FakeFlattenedTensorBucket),
        mock.patch.object(
            serialization_module.MultiprocessingSerializer,
            "serialize",
            side_effect=lambda payload, output_str: f"{payload['metadata']['names']}-{output_str}",
        ),
    ):
        grouped = serialization_module.bucket_named_tensors_for_sglang(named_tensors)
        serialized = serialization_module.serialize_named_tensors_for_sglang(named_tensors)

    assert len(grouped) == 2
    assert serialized == ["['w1']-True", "['w2']-True"]

    FakeFlattenedTensorBucket.supports_multi_dtypes = True
    try:
        with mock.patch.object(serialization_module, "FlattenedTensorBucket", FakeFlattenedTensorBucket):
            assert serialization_module.bucket_named_tensors_for_sglang(named_tensors) == [named_tensors]
    finally:
        FakeFlattenedTensorBucket.supports_multi_dtypes = False

    prefill = RolloutEndpoint(
        name="prefill-0",
        role="prefill",
        host="127.0.0.1",
        port=30000,
        bootstrap_port=30001,
        url="http://127.0.0.1:30000",
        healthy=True,
        policy_version=None,
        model_path="model",
    )
    decode = RolloutEndpoint(
        name="decode-0",
        role="decode",
        host="127.0.0.1",
        port=30010,
        bootstrap_port=None,
        url="http://127.0.0.1:30010",
        healthy=True,
        policy_version=None,
        model_path="model",
    )

    with TemporaryDirectory(prefix="openforge_pd_router_") as temp_dir:
        runtime = PDRouterRuntime(
            prefill_endpoints=[prefill],
            decode_endpoints=[decode],
            checkpoints_dir=temp_dir,
        )
        with (
            mock.patch.object(router_module.subprocess, "Popen", return_value=FakePopen()) as popen,
            mock.patch.object(PDRouterRuntime, "_wait_until_healthy"),
            mock.patch.object(router_module, "_check_health", return_value=True),
        ):
            runtime.start()
            command = popen.call_args.args[0]
            assert "--prefill" in command
            assert "http://127.0.0.1:30000" in command
            assert runtime.endpoint().role == "router"
            assert runtime.is_healthy() is True
            runtime.restart(prefill_endpoints=[prefill], decode_endpoints=[decode])
            runtime.stop()

    with (
        mock.patch.object(router_module.requests, "get", return_value=mock.Mock(status_code=200)),
        mock.patch.object(router_module.ray, "is_initialized", return_value=False),
        mock.patch.object(router_module.socket, "gethostbyname", return_value="127.0.0.2"),
        mock.patch.object(router_module.socket, "gethostname", return_value="localhost"),
    ):
        assert router_module._check_health(39000) is True
        assert router_module._resolve_node_ip_address() == "127.0.0.2"

    with (
        mock.patch.object(
            router_module.requests,
            "get",
            side_effect=requests.RequestException("offline"),
        ),
        mock.patch.object(router_module.ray, "is_initialized", return_value=True),
        mock.patch.object(
            router_module,
            "get_current_ray_node_ip_address",
            return_value="10.0.0.3",
        ),
    ):
        assert router_module._check_health(39000) is False
        assert router_module._resolve_node_ip_address() == "10.0.0.3"


def test_pd_router_worker_wraps_runtime() -> None:
    with mock.patch.object(router_module, "PDRouterRuntime", FakeRouterRuntime):
        worker = PDRouterWorker()
        endpoint = worker.initialize(
            prefill_endpoints=[],
            decode_endpoints=[],
            checkpoints_dir="/tmp",
        )
        assert endpoint.role == "router"
        assert worker.endpoint().url == "http://127.0.0.1:39000"
        assert worker.is_healthy() is True
        worker.restart(prefill_endpoints=[], decode_endpoints=[])
        worker.shutdown()
        assert_raises(
            RuntimeError,
            worker._runtime,
            match="has not been initialized",
        )


TESTS = [
    ("rollout_runtime_is_abstract", test_rollout_runtime_is_abstract),
    ("sglang_engine_spec_url_property", test_sglang_engine_spec_url_property),
    (
        "sglang_control_client_request_and_headers",
        test_sglang_control_client_request_and_headers,
    ),
    (
        "sglang_control_client_public_methods",
        test_sglang_control_client_public_methods,
    ),
    (
        "sglang_control_client_payload_methods",
        test_sglang_control_client_payload_methods,
    ),
    (
        "sglang_control_client_dict_validations",
        test_sglang_control_client_dict_validations,
    ),
    (
        "sglang_engine_runtime_lifecycle_and_delegates",
        test_sglang_engine_runtime_lifecycle_and_delegates,
    ),
    (
        "sglang_engine_runtime_process_launch_helpers",
        test_sglang_engine_runtime_process_launch_helpers,
    ),
    (
        "sglang_engine_runtime_wait_helpers_and_validation",
        test_sglang_engine_runtime_wait_helpers_and_validation,
    ),
    (
        "sglang_runtime_tensor_and_distributed_updates",
        test_sglang_runtime_tensor_and_distributed_updates,
    ),
    (
        "sglang_runtime_aborts_failed_begin_update",
        test_sglang_runtime_aborts_failed_begin_update,
    ),
    (
        "sglang_serialization_buckets_and_router_runtime",
        test_sglang_serialization_buckets_and_router_runtime,
    ),
    ("pd_router_worker_wraps_runtime", test_pd_router_worker_wraps_runtime),
]


def run_suite() -> int:
    return run_named_tests(TESTS, summary_label="SGLang unit")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone tests for rollout SGLang modules."
    )
    parser.parse_args()
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
