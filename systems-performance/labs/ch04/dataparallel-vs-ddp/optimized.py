from __future__ import annotations

from common import Result, TrainingShape


def run(shape: TrainingShape) -> Result:
    host_ms = 0.7
    compute_ms = 18.0
    allreduce_ms = shape.model_mb / 55
    exposed_sync_ms = allreduce_ms * 0.35
    return Result("optimized", host_ms + compute_ms + exposed_sync_ms, host_ms, exposed_sync_ms, checksum=shape.batch_size)
