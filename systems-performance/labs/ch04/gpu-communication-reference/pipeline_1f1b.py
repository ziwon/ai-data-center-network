from __future__ import annotations

import argparse
import sys
import time

import torch

from common import positive_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-side pipeline scheduling sketch.")
    parser.add_argument("--microbatches", type=positive_int, default=8)
    parser.add_argument("--work", type=positive_int, default=8_000_000)
    return parser.parse_args()


def cuda_work(units: int) -> None:
    torch.cuda._sleep(units)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("SKIPPED: pipeline_1f1b requires CUDA", file=sys.stderr)
        raise SystemExit(0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    compute = torch.cuda.Stream(device)
    comm = torch.cuda.Stream(device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.cuda.stream(compute):
        for _ in range(args.microbatches):
            cuda_work(args.work)
    compute.synchronize()
    fill_drain_ms = (time.perf_counter() - start) * 1000

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    events: list[torch.cuda.Event] = []
    for _ in range(args.microbatches):
        with torch.cuda.stream(compute):
            cuda_work(args.work)
            event = torch.cuda.Event()
            event.record(compute)
            events.append(event)
        with torch.cuda.stream(comm):
            comm.wait_event(events[-1])
            cuda_work(args.work // 4)
    torch.cuda.synchronize(device)
    overlapped_ms = (time.perf_counter() - start) * 1000

    print(f"fill_drain_ms={fill_drain_ms:.3f}")
    print(f"overlapped_1f1b_ms={overlapped_ms:.3f}")


if __name__ == "__main__":
    main()
