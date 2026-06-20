from __future__ import annotations

from common import ParallelShape, Result


def run(shape: ParallelShape) -> Result:
    useful_compute = shape.microbatches * shape.stages * shape.stage_compute_ms
    bubble = (shape.stages - 1) * shape.stage_compute_ms
    exposed_tp = shape.microbatches * shape.stages * shape.tp_allgather_ms * 0.35
    return Result("optimized", useful_compute + bubble + exposed_tp, bubble, exposed_tp, checksum=shape.microbatches)
