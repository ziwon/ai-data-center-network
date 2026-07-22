from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.distributed as dist

from common import cleanup_distributed, positive_int, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure process-group reuse vs reinit overhead.")
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--tensor-mb", type=positive_int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = setup_distributed("communicator_reuse")
    numel = args.tensor_mb * 1024 * 1024 // 4
    tensor = torch.ones(numel, device=info.device)

    try:
        reinit_times: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            group = dist.new_group(backend="nccl")
            dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize(info.device)
            dist.destroy_process_group(group)
            reinit_times.append((time.perf_counter() - start) * 1000)

        group = dist.new_group(backend="nccl")
        reuse_times: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize(info.device)
            reuse_times.append((time.perf_counter() - start) * 1000)
        dist.destroy_process_group(group)

        if info.rank == 0:
            print(f"reinit_median_ms={statistics.median(reinit_times):.3f}")
            print(f"reuse_median_ms={statistics.median(reuse_times):.3f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
