from __future__ import annotations

import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[[list["Bucket"]], "StepStats"]


@dataclass(frozen=True)
class Bucket:
    layer: str
    compute_ms: float
    comm_ms: float
    elements: int


@dataclass(frozen=True)
class StepStats:
    checksum: int
    compute_ms: float
    comm_ms: float
    step_ms: float

    @property
    def hidden_comm_ms(self) -> float:
        exposed = max(0.0, self.step_ms - self.compute_ms)
        return max(0.0, self.comm_ms - exposed)

    @property
    def overlap_ratio(self) -> float:
        if self.comm_ms <= 0:
            return 0.0
        return self.hidden_comm_ms / self.comm_ms


@dataclass(frozen=True)
class Result:
    name: str
    stats: StepStats
    median_ms: float


def make_buckets() -> list[Bucket]:
    return [
        Bucket("mlp.5", compute_ms=8.0, comm_ms=14.0, elements=32_768),
        Bucket("mlp.4", compute_ms=9.5, comm_ms=13.0, elements=32_768),
        Bucket("mlp.3", compute_ms=11.0, comm_ms=11.0, elements=24_576),
        Bucket("mlp.2", compute_ms=8.5, comm_ms=9.0, elements=24_576),
        Bucket("mlp.1", compute_ms=7.0, comm_ms=8.0, elements=16_384),
    ]


def sleep_ms(milliseconds: float) -> None:
    time.sleep(milliseconds / 1000)


def compute_bucket(bucket: Bucket) -> int:
    sleep_ms(bucket.compute_ms)
    return bucket.elements * (len(bucket.layer) + 17)


def communicate_bucket(bucket: Bucket) -> int:
    sleep_ms(bucket.comm_ms)
    return bucket.elements


def wait_for_all(futures: list[Future[int]]) -> int:
    return sum(future.result() for future in futures)


def run_overlapped(buckets: list[Bucket]) -> StepStats:
    start = time.perf_counter()
    checksum = 0
    futures: list[Future[int]] = []

    with ThreadPoolExecutor(max_workers=1) as comm_stream:
        for bucket in buckets:
            checksum += compute_bucket(bucket)
            futures.append(comm_stream.submit(communicate_bucket, bucket))
        checksum += wait_for_all(futures)

    step_ms = (time.perf_counter() - start) * 1000
    return StepStats(
        checksum=checksum,
        compute_ms=sum(bucket.compute_ms for bucket in buckets),
        comm_ms=sum(bucket.comm_ms for bucket in buckets),
        step_ms=step_ms,
    )


def measure(name: str, fn: RunFn, buckets: list[Bucket], iterations: int = 8) -> Result:
    timings: list[float] = []
    stats = fn(buckets)
    for _ in range(iterations):
        stats = fn(buckets)
        timings.append(stats.step_ms)
    return Result(name=name, stats=stats, median_ms=statistics.median(timings))


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.stats.checksum != optimized.stats.checksum:
        raise AssertionError("Output checksum changed between baseline and optimized paths.")

    print(f"{'version':<12} {'median_ms':>12} {'compute_ms':>12} {'comm_ms':>10} {'overlap':>10}")
    print("-" * 62)
    for result in (baseline, optimized):
        print(
            f"{result.name:<12} {result.median_ms:>12.3f} "
            f"{result.stats.compute_ms:>12.1f} {result.stats.comm_ms:>10.1f} "
            f"{result.stats.overlap_ratio:>9.1%}"
        )

    print(f"\nstep-time speedup: {baseline.median_ms / optimized.median_ms:.2f}x")
