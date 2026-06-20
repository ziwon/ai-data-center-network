from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable


RunFn = Callable[["Topology"], "Result"]


@dataclass(frozen=True)
class Link:
    a: int
    b: int
    bandwidth_gbps: float
    latency_us: float
    name: str


@dataclass(frozen=True)
class Topology:
    links: dict[tuple[int, int], Link]
    gradient_mb: float


@dataclass(frozen=True)
class Result:
    name: str
    bytes_moved: float
    modelled_ms: float
    slow_edges: int
    checksum: int


def make_topology() -> Topology:
    links: dict[tuple[int, int], Link] = {}

    def add(a: int, b: int, bandwidth_gbps: float, latency_us: float, name: str) -> None:
        link = Link(a, b, bandwidth_gbps, latency_us, name)
        links[(min(a, b), max(a, b))] = link

    add(0, 1, 900, 2.0, "nvlink")
    add(2, 3, 900, 2.0, "nvlink")
    add(0, 2, 64, 9.0, "pcie-cross")
    add(1, 3, 64, 9.0, "pcie-cross")
    add(0, 3, 48, 11.0, "pcie-cross")
    add(1, 2, 48, 11.0, "pcie-cross")
    return Topology(links=links, gradient_mb=512)


def edge_cost_ms(topology: Topology, a: int, b: int, traffic_mb: float) -> tuple[float, bool]:
    link = topology.links[(min(a, b), max(a, b))]
    transfer_ms = (traffic_mb / 1024) / link.bandwidth_gbps * 1000
    return transfer_ms + link.latency_us / 1000, link.name != "nvlink"


def measure(name: str, fn: RunFn, topology: Topology, iterations: int = 8) -> Result:
    results = [fn(topology) for _ in range(iterations)]
    median_ms = statistics.median(result.modelled_ms for result in results)
    latest = results[-1]
    return Result(name, latest.bytes_moved, median_ms, latest.slow_edges, latest.checksum)


def print_report(baseline: Result, optimized: Result) -> None:
    if baseline.checksum != optimized.checksum:
        raise AssertionError("Topology variants changed logical collective output.")
    print(f"{'version':<12} {'model_ms':>10} {'traffic_MiB':>12} {'slow_edges':>11} {'checksum':>10}")
    print("-" * 62)
    for result in (baseline, optimized):
        print(
            f"{result.name:<12} {result.modelled_ms:>10.3f} "
            f"{result.bytes_moved / (1024 * 1024):>12.1f} {result.slow_edges:>11} {result.checksum:>10}"
        )
    print(f"\ntopology-aware speedup: {baseline.modelled_ms / optimized.modelled_ms:.2f}x")
