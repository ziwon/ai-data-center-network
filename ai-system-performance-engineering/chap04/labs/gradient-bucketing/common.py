from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[[list[list[float]]], "BucketStats"]


@dataclass(frozen=True)
class BucketStats:
    checksum: float
    launches: int
    bytes_transferred: int
    modelled_comm_ms: float
    elapsed_ms: float


@dataclass(frozen=True)
class Result:
    name: str
    stats: BucketStats
    median_ms: float


def make_gradients(num_buckets: int = 96, bucket_elems: int = 2_048) -> list[list[float]]:
    gradients: list[list[float]] = []
    for bucket_id in range(num_buckets):
        bucket = [
            math.sin((bucket_id + 1) * (idx + 3) * 0.001) * 0.5
            for idx in range(bucket_elems)
        ]
        gradients.append(bucket)
    return gradients


def model_comm_ms(launches: int, bytes_transferred: int) -> float:
    launch_latency_ms = 0.035
    bandwidth_bytes_per_ms = 24 * 1024 * 1024
    return launches * launch_latency_ms + bytes_transferred / bandwidth_bytes_per_ms


def fp16_like(value: float) -> float:
    return round(value, 3)


def measure(name: str, fn: RunFn, gradients: list[list[float]], iterations: int = 8) -> Result:
    timings: list[float] = []
    stats = fn(gradients)
    for _ in range(iterations):
        start = time.perf_counter()
        stats = fn(gradients)
        timings.append((time.perf_counter() - start) * 1000)
    return Result(name=name, stats=stats, median_ms=statistics.median(timings))


def print_report(baseline: Result, optimized: Result) -> None:
    checksum_delta = abs(baseline.stats.checksum - optimized.stats.checksum)
    relative_delta = checksum_delta / max(abs(baseline.stats.checksum), 1.0)
    if relative_delta > 0.01:
        raise AssertionError(f"Optimized checksum drift is too large: {relative_delta:.3%}")

    print(
        f"{'version':<12} {'py_ms':>10} {'comm_ms':>10} "
        f"{'launches':>10} {'MiB':>10} {'checksum':>14}"
    )
    print("-" * 74)
    for result in (baseline, optimized):
        mib = result.stats.bytes_transferred / (1024 * 1024)
        print(
            f"{result.name:<12} {result.median_ms:>10.3f} "
            f"{result.stats.modelled_comm_ms:>10.3f} {result.stats.launches:>10} "
            f"{mib:>10.2f} {result.stats.checksum:>14.3f}"
        )

    print(f"\nmodelled communication speedup: {baseline.stats.modelled_comm_ms / optimized.stats.modelled_comm_ms:.2f}x")
    print(f"relative checksum drift: {relative_delta:.4%}")
