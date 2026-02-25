# Copyright 2026 openforge

import asyncio
from abc import ABC, abstractmethod
from collections import deque

from openforge.configs import Reward, RolloutDatum


class DataPool(ABC):
    """Abstract base class for data pools."""

    @abstractmethod
    async def put(self, datum: RolloutDatum) -> None: ...

    @abstractmethod
    async def set_reward(self, sample_id: str, reward: Reward) -> bool: ...

    @abstractmethod
    async def pull(self, n: int) -> list[RolloutDatum]: ...

    @abstractmethod
    async def done(self, ids: list[str]) -> None: ...


class InMemoryDataPool(DataPool):
    """In-memory DataPool for local development."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[str, RolloutDatum] = {}
        self._ready: deque[str] = deque()
        self._queued: set[str] = set()

    async def put(self, datum: RolloutDatum) -> None:
        async with self._lock:
            self._records[datum.sample_id] = datum
            self._enqueue(datum.sample_id)

    async def set_reward(self, sample_id: str, reward: Reward) -> bool:
        async with self._lock:
            record = self._records.get(sample_id)
            if record is None:
                return False
            record.reward = reward
            self._enqueue(sample_id)
            return True

    async def pull(self, n: int) -> list[RolloutDatum]:
        if n <= 0:
            return []

        async with self._lock:
            items: list[RolloutDatum] = []
            while len(items) < n and self._ready:
                sample_id = self._ready.popleft()
                self._queued.discard(sample_id)
                record = self._records.get(sample_id)
                if record is None or record.consumed or record.reward is None:
                    continue
                items.append(record)
            return items

    async def done(self, ids: list[str]) -> None:
        async with self._lock:
            for sample_id in ids:
                record = self._records.get(sample_id)
                if record is None:
                    continue
                record.consumed = True
                self._queued.discard(sample_id)

    def _enqueue(self, sample_id: str) -> None:
        record = self._records.get(sample_id)
        if record is None:
            return
        if record.consumed or record.reward is None or sample_id in self._queued:
            return
        self._ready.append(sample_id)
        self._queued.add(sample_id)
