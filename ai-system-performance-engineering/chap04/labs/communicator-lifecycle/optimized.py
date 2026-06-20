from __future__ import annotations

import time

from common import Result, all_reduce_step, create_communicator


def run(steps: int) -> Result:
    timings: list[float] = []
    checksum = 0
    comm = create_communicator()
    for step in range(steps):
        start = time.perf_counter()
        checksum += all_reduce_step(comm, step)
        timings.append((time.perf_counter() - start) * 1000)
    return Result("optimized", checksum, sum(timings) / len(timings), setup_count=1)
