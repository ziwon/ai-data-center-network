from __future__ import annotations

import time

from common import HandoffStats, KvWorkload, checksum_blocks


def run(workload: KvWorkload) -> HandoffStats:
    start = time.perf_counter()
    target: list[bytearray] = []
    scratch = bytearray(workload.block_size)

    for block_id in workload.selected_blocks:
        scratch[:] = workload.source[block_id]
        target.append(bytearray(scratch))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return HandoffStats(
        checksum=checksum_blocks(target),
        operations=len(workload.selected_blocks) * 2,
        bytes_transferred=len(workload.selected_blocks) * workload.block_size,
        elapsed_ms=elapsed_ms,
    )
