from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[["TrainingShape"], "Result"]


@dataclass(frozen=True)
class TrainingShape:
    gpus: int = 4
    batch_size: int = 1024
    model_mb: float = 64
    activation_mb: float = 256


@dataclass(frozen=True)
class Result:
    name: str
    modelled_ms: float
    host_ms: float
    gpu_sync_ms: float
    checksum: int


def measure(name: str, fn: RunFn, shape: TrainingShape, iterations: int = 8) -> Result:
    results = [fn(shape) for _ in range(iterations)]
    latest = results[-1]
    return Result(
        name=name,
        modelled_ms=statistics.median(result.modelled_ms for result in results),
        host_ms=latest.host_ms,
        gpu_sync_ms=latest.gpu_sync_ms,
        checksum=latest.checksum,
    )


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.checksum != optimized.checksum:
        raise AssertionError("Training output contract changed.")
    print(f"{'version':<12} {'model_ms':>10} {'host_ms':>10} {'sync_ms':>10} {'checksum':>10}")
    print("-" * 58)
    for result in (baseline, optimized):
        print(
            f"{result.name:<12} {result.modelled_ms:>10.3f} "
            f"{result.host_ms:>10.3f} {result.gpu_sync_ms:>10.3f} {result.checksum:>10}"
        )
    print(f"\nDDP-shaped speedup: {baseline.modelled_ms / optimized.modelled_ms:.2f}x")
