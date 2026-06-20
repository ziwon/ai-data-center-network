from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[["ParallelShape"], "Result"]


@dataclass(frozen=True)
class ParallelShape:
    stages: int = 4
    microbatches: int = 8
    stage_compute_ms: float = 6.0
    tp_allgather_ms: float = 2.0


@dataclass(frozen=True)
class Result:
    name: str
    modelled_ms: float
    bubble_ms: float
    exposed_tp_ms: float
    checksum: int


def measure(name: str, fn: RunFn, shape: ParallelShape, iterations: int = 8) -> Result:
    results = [fn(shape) for _ in range(iterations)]
    latest = results[-1]
    return Result(
        name,
        statistics.median(result.modelled_ms for result in results),
        latest.bubble_ms,
        latest.exposed_tp_ms,
        latest.checksum,
    )


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.checksum != optimized.checksum:
        raise AssertionError("Parallel schedule changed logical microbatch output.")
    print(f"{'version':<12} {'model_ms':>10} {'bubble_ms':>10} {'tp_exposed':>11} {'checksum':>10}")
    print("-" * 62)
    for result in (baseline, optimized):
        print(
            f"{result.name:<12} {result.modelled_ms:>10.3f} "
            f"{result.bubble_ms:>10.3f} {result.exposed_tp_ms:>11.3f} {result.checksum:>10}"
        )
    print(f"\nschedule speedup: {baseline.modelled_ms / optimized.modelled_ms:.2f}x")
