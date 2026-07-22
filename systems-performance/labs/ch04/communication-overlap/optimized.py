from __future__ import annotations

from common import Bucket, StepStats, run_overlapped


def run(buckets: list[Bucket]) -> StepStats:
    return run_overlapped(buckets)
