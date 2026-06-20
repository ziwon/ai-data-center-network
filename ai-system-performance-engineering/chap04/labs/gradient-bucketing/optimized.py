from __future__ import annotations

from common import BucketStats, fp16_like, model_comm_ms


def run(gradients: list[list[float]]) -> BucketStats:
    fused = [fp16_like(value) for bucket in gradients for value in bucket]
    checksum = sum(fused)
    launches = 1
    bytes_transferred = len(fused) * 2

    return BucketStats(
        checksum=checksum,
        launches=launches,
        bytes_transferred=bytes_transferred,
        modelled_comm_ms=model_comm_ms(launches, bytes_transferred),
        elapsed_ms=0.0,
    )
