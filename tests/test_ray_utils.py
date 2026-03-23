# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

from unittest.mock import patch

from _script_test_utils import expect_raises, install_test_stubs, run_tests

install_test_stubs()

import ray

from openforge.gateway.runtime import Runtime
from openforge.utils import ray as ray_utils
from test_gateway_runtime import _runtime_config, _server_config


def test_create_placement_groups_fails_fast_when_ray_sees_too_few_gpus() -> None:
    cfg = Runtime(cfg=_server_config())._build_config(runtime_config=_runtime_config())
    with patch.object(ray, "cluster_resources", lambda: {"GPU": 0.0}):
        with patch.object(ray, "available_resources", lambda: {"GPU": 0.0}):
            with expect_raises(RuntimeError, "requested 2, cluster reported 0.0"):
                ray_utils.create_placement_groups(cfg)


def main() -> int:
    return run_tests([test_create_placement_groups_fails_fast_when_ray_sees_too_few_gpus])


if __name__ == "__main__":
    raise SystemExit(main())
