# Copied from SLIME/verl sequence-length balancing utilities.
# Copyright 2024 Bytedance Ltd. and/or its affiliates

import copy
import heapq

__all__ = [
    "get_minimum_num_micro_batches",
    "get_seqlen_balanced_partitions",
]


def get_minimum_num_micro_batches(
    lengths: list[int],
    max_tokens_per_partition: int,
) -> int:
    """Greedily estimate how many microbatches fit under a token cap."""
    batches: list[int] = []
    for length in lengths:
        for index, total in enumerate(batches):
            if total + length <= max_tokens_per_partition:
                batches[index] += length
                break
        else:
            batches.append(length)
    return len(batches)


def _karmarkar_karp(
    seqlen_list: list[int],
    k_partitions: int,
    *,
    equal_size: bool,
) -> list[list[int]]:
    class _Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int) -> None:
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other: "_Set") -> None:
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other: "_Set") -> bool:
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class _State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            self.sets = [_Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for index, (item_idx, seqlen) in enumerate(items):
                self.sets[index].add(idx=item_idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self) -> list[list[int]]:
            partitions: list[list[int]] = []
            for subset in self.sets:
                partitions.append([idx for idx, _ in subset.items])
            return partitions

        def merge(self, other: "_State") -> None:
            for index in range(self.k):
                self.sets[index].merge(other.sets[self.k - 1 - index])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other: "_State") -> bool:
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    sorted_seqlens = sorted((seqlen, index) for index, seqlen in enumerate(seqlen_list))
    states: list[_State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0
        for offset in range(0, len(sorted_seqlens), k_partitions):
            items = []
            for index in range(k_partitions):
                seqlen, item_idx = sorted_seqlens[offset + index]
                items.append((item_idx, seqlen))
            heapq.heappush(states, _State(items=items, k=k_partitions))
    else:
        for seqlen, item_idx in sorted_seqlens:
            heapq.heappush(states, _State(items=[(item_idx, seqlen)], k=k_partitions))

    while len(states) > 1:
        state0 = heapq.heappop(states)
        state1 = heapq.heappop(states)
        state0.merge(state1)
        heapq.heappush(states, state0)

    return states[0].get_partitions()


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    *,
    equal_size: bool,
) -> list[list[int]]:
    """Partition item indexes into groups with balanced total sequence length."""
    assert len(seqlen_list) >= k_partitions, (
        f"number of items [{len(seqlen_list)}] < partitions [{k_partitions}]"
    )

    def _check_and_sort(partitions: list[list[int]]) -> list[list[int]]:
        assert len(partitions) == k_partitions
        seen = set()
        sorted_partitions: list[list[int]] = []
        for partition in partitions:
            assert partition, "empty partition"
            sorted_partition = sorted(partition)
            sorted_partitions.append(sorted_partition)
            seen.update(sorted_partition)
        assert seen == set(range(len(seqlen_list)))
        return sorted_partitions

    return _check_and_sort(
        _karmarkar_karp(
            seqlen_list=seqlen_list,
            k_partitions=k_partitions,
            equal_size=equal_size,
        )
    )


def get_reverse_idx(idx_map: list[int]) -> list[int]:
    reverse_idx_map = copy.deepcopy(idx_map)
    for index, idx in enumerate(idx_map):
        reverse_idx_map[idx] = index
    return reverse_idx_map
