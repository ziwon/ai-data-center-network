from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[[int], "Result"]


@dataclass(frozen=True)
class Result:
    name: str
    checksum: int
    median_step_ms: float
    setup_count: int


def create_communicator() -> dict[str, int]:
    time.sleep(0.004)
    return {"world_size": 4, "channels": 8}


def all_reduce_step(comm: dict[str, int], step: int) -> int:
    time.sleep(0.001)
    return (step + 1) * comm["world_size"] * comm["channels"]


def measure(name: str, fn: RunFn, steps: int = 16, iterations: int = 5) -> Result:
    results = [fn(steps) for _ in range(iterations)]
    latest = results[-1]
    return Result(
        name=name,
        checksum=latest.checksum,
        median_step_ms=statistics.median(result.median_step_ms for result in results),
        setup_count=latest.setup_count,
    )


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.checksum != optimized.checksum:
        raise AssertionError("Communicator lifecycle changed collective result.")
    print(f"{'version':<12} {'step_ms':>10} {'setups':>8} {'checksum':>10}")
    print("-" * 46)
    for result in (baseline, optimized):
        print(f"{result.name:<12} {result.median_step_ms:>10.3f} {result.setup_count:>8} {result.checksum:>10}")
    print(f"\nreuse speedup: {baseline.median_step_ms / optimized.median_step_ms:.2f}x")
