from __future__ import annotations

import time

from common import HandoffStats, KvWorkload


def run(workload: KvWorkload) -> HandoffStats:
    start = time.perf_counter()
    payload = bytearray(len(workload.selected_blocks) * workload.block_size)

    for offset, block_id in enumerate(workload.selected_blocks):
        begin = offset * workload.block_size
        end = begin + workload.block_size
        payload[begin:end] = workload.source[block_id]

    elapsed_ms = (time.perf_counter() - start) * 1000
    return HandoffStats(
        checksum=sum(payload),
        operations=1,
        bytes_transferred=len(payload),
        elapsed_ms=elapsed_ms,
    )
