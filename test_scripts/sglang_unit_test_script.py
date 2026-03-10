#!/usr/bin/env python3
# Copyright 2026 openforge

"""Focused unit tests for rollout SGLang modules."""

import argparse
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import mock

import requests
import torch

import openforge.rollout.sglang.client as client_module
import openforge.rollout.sglang.engine_runtime as engine_runtime_module
import openforge.rollout.sglang.runtime as runtime_module
import openforge.rollout.sglang.serialization as serialization_module
from openforge.policy.types import PolicyArtifactRef
from openforge.rollout.runtime import RolloutRuntime
from openforge.rollout.sglang.client import SGLangControlClient
from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime
from openforge.rollout.sglang.runtime import SGLangRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from script_test_utils import assert_raises, run_named_tests


def make_spec(
    *,
    port: int = 31000,
    model_path: str = "/tmp/model",
    policy_version: int | None = None,
) -> SGLangEngineSpec:
    return SGLangEngineSpec(
        engine_id=0,
        name="regular-0",
        role="regular",
        host="127.0.0.1",
        port=port,
        bootstrap_port=None,
        model_path=model_path,
        tokenizer_path="/tmp/tokenizer",
        policy_version=policy_version,
        server_args={
            "host": "0.0.0.0",
            "port": port,
            "model_path": model_path,
            "weight_version": "default"
            if policy_version is None
            else str(policy_version),
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
        self.client = mock.Mock(
            get_model_info=mock.Mock(
                return_value={"weight_version": spec.policy_version}
            )
        )
        self.disk_update_calls: list[dict[str, object]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def restart(self) -> None:
        self.started = True
        self.stopped = True

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

    def update_weights_from_disk(self, **kwargs) -> dict[str, object]:
        self.disk_update_calls.append(kwargs)
        self.spec.model_path = str(kwargs["model_path"])
        weight_version = kwargs.get("weight_version")
        if weight_version is not None:
            self.spec.policy_version = int(weight_version)
        return {"success": True}


class FakeFlattenedTensorBucket:
    supports_multi_dtypes = False

    def __init__(self, *, named_tensors):
        self.named_tensors = list(named_tensors)

    def get_flattened_tensor(self):
        return torch.tensor([len(self.named_tensors)], dtype=torch.float32)

    def get_metadata(self):
        return {"names": [name for name, _ in self.named_tensors]}


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
            (200, {"success": True}),
            (200, {"action": "snapshot"}),
            (200, {"paused": True}),
            (200, {"continued": True}),
        ],
    ):
        assert client.health_generate(timeout=1.0) is False
        assert client.flush_cache(timeout=1.0) is False
        assert client.get_model_info() == {"weight_version": 11}
        assert client.get_server_info() == {"name": "server"}
        assert client.get_weight_version() == "11"
        assert client.update_weights_from_disk(model_path="/tmp/model") == {
            "success": True
        }
        assert client.check_weights(action="snapshot") == {"action": "snapshot"}
        assert client.pause_generation(mode="abort") == {"paused": True}
        assert client.continue_generation() == {"continued": True}


def test_sglang_control_client_payload_methods() -> None:
    test_cases = [
        (
            "update_weights_from_disk",
            {"model_path": "/tmp/model", "weight_version": "7"},
            "/update_weights_from_disk",
            {
                "model_path": "/tmp/model",
                "load_format": None,
                "flush_cache": True,
                "abort_all_requests": False,
                "weight_version": "7",
            },
        ),
        ("check_weights", {"action": "snapshot"}, "/weights_checker", {"action": "snapshot"}),
        ("pause_generation", {"mode": "abort"}, "/pause_generation", {"mode": "abort"}),
        ("continue_generation", {}, "/continue_generation", {}),
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
        ("update_weights_from_disk", {"model_path": "/tmp/model"}),
        ("check_weights", {"action": "snapshot"}),
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
        update_weights_from_disk=mock.Mock(return_value={"success": True}),
        check_weights=mock.Mock(return_value={"action": "snapshot"}),
        flush_cache=mock.Mock(return_value=True),
        pause_generation=mock.Mock(return_value="paused"),
        continue_generation=mock.Mock(return_value="continued"),
    )

    assert runtime.is_healthy() is True
    assert runtime.get_server_info() == {"server": "ok"}
    assert runtime.get_weight_version() == "9"
    assert runtime.update_weights_from_disk(
        model_path="/tmp/model",
        weight_version="10",
    ) == {"success": True}
    assert spec.model_path == "/tmp/model"
    assert spec.policy_version == 10
    assert runtime.check_weights(action="snapshot") == {"action": "snapshot"}
    assert runtime.flush_cache() is True
    assert runtime.pause_generation(mode="abort") == "paused"
    assert runtime.continue_generation() == "continued"

    runtime.process = fake_process
    with mock.patch.object(engine_runtime_module, "_kill_process_tree", return_value=False):
        runtime.stop()
    assert runtime.process is None
    assert fake_process.terminate_calls == 1


def test_sglang_engine_runtime_process_launch_helpers() -> None:
    fake_process = mock.Mock()
    fake_context = mock.Mock(Process=mock.Mock(return_value=fake_process))

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


def test_sglang_engine_runtime_wait_helpers() -> None:
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

    stubborn = StubbornProcess()
    runtime.process = stubborn
    with mock.patch.object(engine_runtime_module, "_kill_process_tree", return_value=False):
        runtime.stop()
    assert stubborn.kill_calls == 1


def test_sglang_runtime_regular_engine_flow() -> None:
    spec = make_spec(policy_version=1)
    fake_runtime = FakeEngineRuntime(spec)

    with mock.patch.object(runtime_module, "SGLangEngineRuntime", return_value=fake_runtime):
        runtime = SGLangRuntime(spec)

    runtime.start()
    assert runtime.endpoint().policy_version == 1
    assert runtime.is_healthy() is True
    assert runtime.get_model_info() == {"weight_version": 1}
    assert runtime.get_server_info() == {"name": "regular-0"}
    assert runtime.get_weight_version() == "1"
    assert runtime.flush_cache() is True
    runtime.pause_generation(mode="abort")
    runtime.continue_generation()

    runtime.load_policy_artifact(
        PolicyArtifactRef(step=3, policy_version=4, path="/tmp/policy_dir")
    )
    assert fake_runtime.disk_update_calls[0]["model_path"] == "/tmp/policy_dir"
    assert spec.model_path == "/tmp/policy_dir"
    assert spec.policy_version == 4

    runtime.stop()
    assert fake_runtime.stopped is True


def test_sglang_serialization_buckets() -> None:
    named_tensors = [
        ("w1", torch.ones(2, dtype=torch.float16)),
        ("w2", torch.ones(2, dtype=torch.float32)),
    ]

    with (
        mock.patch.object(
            serialization_module,
            "FlattenedTensorBucket",
            FakeFlattenedTensorBucket,
        ),
        mock.patch.object(
            serialization_module.MultiprocessingSerializer,
            "serialize",
            side_effect=lambda payload, output_str: f"{payload['metadata']['names']}-{output_str}",
        ),
    ):
        grouped = serialization_module.bucket_named_tensors_for_sglang(named_tensors)
        serialized = serialization_module.serialize_named_tensors_for_sglang(
            named_tensors
        )

    assert len(grouped) == 2
    assert serialized == ["['w1']-True", "['w2']-True"]

    FakeFlattenedTensorBucket.supports_multi_dtypes = True
    try:
        with mock.patch.object(
            serialization_module,
            "FlattenedTensorBucket",
            FakeFlattenedTensorBucket,
        ):
            assert serialization_module.bucket_named_tensors_for_sglang(
                named_tensors
            ) == [named_tensors]
    finally:
        FakeFlattenedTensorBucket.supports_multi_dtypes = False


def run_suite() -> int:
    return run_named_tests(
        [
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
                "sglang_engine_runtime_wait_helpers",
                test_sglang_engine_runtime_wait_helpers,
            ),
            ("sglang_runtime_regular_engine_flow", test_sglang_runtime_regular_engine_flow),
            ("sglang_serialization_buckets", test_sglang_serialization_buckets),
        ],
        summary_label="SGLang",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Focused unit tests for rollout SGLang modules."
    )
    parser.parse_args()

    with TemporaryDirectory(prefix="openforge_sglang_unit_"):
        return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
