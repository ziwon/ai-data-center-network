from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[["KvWorkload"], "HandoffStats"]


@dataclass(frozen=True)
class KvWorkload:
    source: list[bytearray]
    selected_blocks: list[int]
    block_size: int


@dataclass(frozen=True)
class HandoffStats:
    checksum: int
    operations: int
    bytes_transferred: int
    elapsed_ms: float


@dataclass(frozen=True)
class Result:
    name: str
    stats: HandoffStats
    median_ms: float


def make_workload(blocks: int = 256, block_size: int = 16 * 1024) -> KvWorkload:
    source: list[bytearray] = []
    for block_id in range(blocks):
        source.append(bytearray((block_id + offset) % 251 for offset in range(block_size)))

    selected = [
        block_id
        for block_id in range(blocks)
        if block_id % 7 in (0, 3) or block_id % 31 == 0
    ]
    return KvWorkload(source=source, selected_blocks=selected, block_size=block_size)


def checksum_blocks(blocks: list[bytearray]) -> int:
    return sum(sum(block) for block in blocks)


def measure(name: str, fn: RunFn, workload: KvWorkload, iterations: int = 10) -> Result:
    timings: list[float] = []
    stats = fn(workload)
    for _ in range(iterations):
        stats = fn(workload)
        timings.append(stats.elapsed_ms)
    return Result(name=name, stats=stats, median_ms=statistics.median(timings))


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.stats.checksum != optimized.stats.checksum:
        raise AssertionError("Selected KV payload changed between baseline and optimized paths.")

    print(f"{'version':<12} {'median_ms':>12} {'operations':>12} {'MiB':>10} {'checksum':>14}")
    print("-" * 68)
    for result in (baseline, optimized):
        mib = result.stats.bytes_transferred / (1024 * 1024)
        print(
            f"{result.name:<12} {result.median_ms:>12.3f} "
            f"{result.stats.operations:>12} {mib:>10.2f} {result.stats.checksum:>14}"
        )

    print(f"\ntier-handoff speedup: {baseline.median_ms / optimized.median_ms:.2f}x")
