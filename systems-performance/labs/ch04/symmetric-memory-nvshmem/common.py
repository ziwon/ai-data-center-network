from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[["ExchangeWorkload"], "Result"]


@dataclass(frozen=True)
class ExchangeWorkload:
    operations: int = 64
    block_size: int = 8 * 1024


@dataclass(frozen=True)
class Result:
    name: str
    checksum: int
    median_op_ms: float
    setup_count: int


def register_buffer() -> bytearray:
    time.sleep(0.0007)
    return bytearray(8 * 1024)


def exchange(buffer: bytearray, token: int) -> int:
    time.sleep(0.00025)
    buffer[0] = token % 251
    buffer[-1] = (token * 3) % 251
    return buffer[0] + buffer[-1]


def measure(name: str, fn: RunFn, workload: ExchangeWorkload, iterations: int = 5) -> Result:
    results = [fn(workload) for _ in range(iterations)]
    latest = results[-1]
    return Result(
        name,
        latest.checksum,
        statistics.median(result.median_op_ms for result in results),
        latest.setup_count,
    )


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.checksum != optimized.checksum:
        raise AssertionError("Symmetric-memory variant changed payload checksum.")
    print(f"{'version':<12} {'op_ms':>10} {'setups':>8} {'checksum':>10}")
    print("-" * 46)
    for result in (baseline, optimized):
        print(f"{result.name:<12} {result.median_op_ms:>10.3f} {result.setup_count:>8} {result.checksum:>10}")
    print(f"\nsymmetric-buffer speedup: {baseline.median_op_ms / optimized.median_op_ms:.2f}x")
