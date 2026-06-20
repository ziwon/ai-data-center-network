from __future__ import annotations

from common import BucketStats, model_comm_ms


def run(gradients: list[list[float]]) -> BucketStats:
    checksum = 0.0
    bytes_transferred = 0

    for bucket in gradients:
        checksum += sum(bucket)
        bytes_transferred += len(bucket) * 4

    launches = len(gradients)
    return BucketStats(
        checksum=checksum,
        launches=launches,
        bytes_transferred=bytes_transferred,
        modelled_comm_ms=model_comm_ms(launches, bytes_transferred),
        elapsed_ms=0.0,
    )
