from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from common import positive_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU peer-copy topology sweep.")
    parser.add_argument("--size-mb", type=positive_int, default=256)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIPPED: topology_sweep requires at least 2 CUDA GPUs", file=sys.stderr)
        raise SystemExit(0)

    numel = args.size_mb * 1024 * 1024 // 4
    print(f"{'src':>4} {'dst':>4} {'p2p':>5} {'median_ms':>12} {'GBps':>10}")
    print("-" * 44)

    for src in range(torch.cuda.device_count()):
        for dst in range(torch.cuda.device_count()):
            if src == dst:
                continue
            source = torch.randn(numel, device=f"cuda:{src}")
            target = torch.empty(numel, device=f"cuda:{dst}")
            p2p = torch.cuda.can_device_access_peer(dst, src)

            for _ in range(max(args.warmup, 0)):
                target.copy_(source, non_blocking=True)
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)

            timings: list[float] = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                target.copy_(source, non_blocking=True)
                torch.cuda.synchronize(dst)
                timings.append((time.perf_counter() - start) * 1000)

            median_ms = statistics.median(timings)
            gbps = (args.size_mb / 1024) / (median_ms / 1000)
            print(f"{src:>4} {dst:>4} {str(p2p):>5} {median_ms:>12.3f} {gbps:>10.2f}")


if __name__ == "__main__":
    main()
