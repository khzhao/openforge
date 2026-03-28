# Copyright 2026 openforge

from types import SimpleNamespace

from openforge.rollout.sglang.engine import Engine


class _HealthyClient:
    def __init__(self) -> None:
        self.flush_calls = 0
        self.health_calls = 0

    def health_generate(self, *, timeout: float) -> bool:
        self.health_calls += 1
        return True

    def flush_cache(self, *, timeout: float) -> bool:
        self.flush_calls += 1
        return False


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


def test_engine_ready_does_not_wait_for_flush_cache() -> None:
    engine = Engine()
    engine.spec = SimpleNamespace(engine_name="regular-0")
    engine.client = _HealthyClient()
    engine.process = _AliveProcess()

    engine._wait_until_ready()

    assert engine.client.health_calls == 1
    assert engine.client.flush_calls == 0
