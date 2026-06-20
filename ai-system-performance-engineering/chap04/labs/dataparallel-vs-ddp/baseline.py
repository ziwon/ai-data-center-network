from __future__ import annotations

from common import Result, TrainingShape


def run(shape: TrainingShape) -> Result:
    scatter_ms = shape.activation_mb / 28
    gather_ms = shape.activation_mb / 22
    python_replica_ms = shape.gpus * 1.1
    primary_reduce_ms = shape.model_mb / 18
    compute_ms = 18.0
    host_ms = scatter_ms + gather_ms + python_replica_ms
    sync_ms = primary_reduce_ms
    return Result("baseline", host_ms + compute_ms + sync_ms, host_ms, sync_ms, checksum=shape.batch_size)
