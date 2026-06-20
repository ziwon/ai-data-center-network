from __future__ import annotations

import time

from common import ExchangeWorkload, Result, exchange, register_buffer


def run(workload: ExchangeWorkload) -> Result:
    buffer = register_buffer()
    timings: list[float] = []
    checksum = 0
    for op in range(workload.operations):
        start = time.perf_counter()
        checksum += exchange(buffer, op)
        timings.append((time.perf_counter() - start) * 1000)
    return Result("optimized", checksum, sum(timings) / len(timings), setup_count=1)
