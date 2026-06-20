from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.distributed as dist

from common import cleanup_distributed, positive_int, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCCL all-reduce bucket size sweep.")
    parser.add_argument("--min-kb", type=positive_int, default=4)
    parser.add_argument("--max-mb", type=positive_int, default=256)
    parser.add_argument("--iterations", type=positive_int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def bucket_sizes(min_kb: int, max_mb: int) -> list[int]:
    sizes: list[int] = []
    size = min_kb * 1024
    max_bytes = max_mb * 1024 * 1024
    while size <= max_bytes:
        sizes.append(size)
        size *= 2
    return sizes


def main() -> None:
    args = parse_args()
    info = setup_distributed("allreduce_bucket_sweep")
    dtype = dtype_from_name(args.dtype)

    try:
        if info.rank == 0:
            print(f"{'bytes':>12} {'dtype':>8} {'median_ms':>12} {'busbw_GBps':>12}")
            print("-" * 50)

        for size_bytes in bucket_sizes(args.min_kb, args.max_mb):
            element_size = torch.empty((), dtype=dtype).element_size()
            numel = max(1, size_bytes // element_size)
            tensor = torch.randn(numel, device=info.device, dtype=dtype)

            for _ in range(max(args.warmup, 0)):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(info.device)

            timings: list[float] = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(info.device)
                timings.append((time.perf_counter() - start) * 1000)

            median_ms = statistics.median(timings)
            algorithm_bytes = size_bytes * 2 * (info.world_size - 1) / info.world_size
            busbw_gbps = algorithm_bytes / (median_ms / 1000) / 1e9

            if info.rank == 0:
                print(f"{size_bytes:>12} {args.dtype:>8} {median_ms:>12.3f} {busbw_gbps:>12.2f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
