from __future__ import annotations

from common import Bucket, StepStats, communicate_bucket, compute_bucket


def run(buckets: list[Bucket]) -> StepStats:
    checksum = 0
    for bucket in buckets:
        checksum += compute_bucket(bucket)

    for bucket in buckets:
        checksum += communicate_bucket(bucket)

    return StepStats(
        checksum=checksum,
        compute_ms=sum(bucket.compute_ms for bucket in buckets),
        comm_ms=sum(bucket.comm_ms for bucket in buckets),
        step_ms=sum(bucket.compute_ms + bucket.comm_ms for bucket in buckets),
    )
