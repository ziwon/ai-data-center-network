from __future__ import annotations

from common import Result, Topology, edge_cost_ms


def run(topology: Topology) -> Result:
    rank_order = [0, 2, 1, 3]
    traffic_mb = topology.gradient_mb
    total_ms = 0.0
    slow_edges = 0
    for left, right in zip(rank_order, rank_order[1:] + rank_order[:1]):
        cost, slow = edge_cost_ms(topology, left, right, traffic_mb)
        total_ms += cost
        slow_edges += int(slow)
    return Result("baseline", traffic_mb * 1024 * 1024 * 4, total_ms, slow_edges, checksum=sum(rank_order))
