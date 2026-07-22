from __future__ import annotations

import time

from common import Result, all_reduce_step, create_communicator


def run(steps: int) -> Result:
    timings: list[float] = []
    checksum = 0
    for step in range(steps):
        start = time.perf_counter()
        comm = create_communicator()
        checksum += all_reduce_step(comm, step)
        timings.append((time.perf_counter() - start) * 1000)
    return Result("baseline", checksum, sum(timings) / len(timings), setup_count=steps)
