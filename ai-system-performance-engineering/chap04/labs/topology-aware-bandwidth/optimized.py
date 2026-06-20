from __future__ import annotations

from common import Result, Topology, edge_cost_ms


def run(topology: Topology) -> Result:
    local_edges = [(0, 1), (2, 3)]
    cross_edges = [(0, 2), (1, 3)]
    total_ms = 0.0
    slow_edges = 0

    for left, right in local_edges:
        cost, slow = edge_cost_ms(topology, left, right, topology.gradient_mb)
        total_ms += cost
        slow_edges += int(slow)

    for left, right in cross_edges:
        cost, slow = edge_cost_ms(topology, left, right, topology.gradient_mb / 2)
        total_ms += cost
        slow_edges += int(slow)

    return Result("optimized", topology.gradient_mb * 1024 * 1024 * 4, total_ms, slow_edges, checksum=sum(range(4)))
