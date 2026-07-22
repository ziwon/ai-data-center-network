from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP

from common import TinyStack, add_training_args, cleanup_distributed, make_batch, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DataParallel shape with torchrun DDP shape.")
    add_training_args(parser)
    return parser.parse_args()


def run_dataparallel(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIPPED: dataparallel_vs_ddp requires at least 2 CUDA GPUs", file=sys.stderr)
        raise SystemExit(0)
    device = torch.device("cuda:0")
    model = DataParallel(TinyStack(args.hidden_size, args.layers).to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs, targets = make_batch(args.batch_size, args.hidden_size, device)

    timings: list[float] = []
    for step in range(args.warmup + args.iterations):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step >= args.warmup:
            timings.append((time.perf_counter() - start) * 1000)
    print(f"DataParallel median_ms={statistics.median(timings):.3f}")


def run_ddp(args: argparse.Namespace) -> None:
    info = setup_distributed("dataparallel_vs_ddp DDP")
    try:
        model = DDP(TinyStack(args.hidden_size, args.layers).to(info.device), device_ids=[info.local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs, targets = make_batch(args.batch_size, args.hidden_size, info.device)
        timings: list[float] = []
        for step in range(args.warmup + args.iterations):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(info.device)
            if step >= args.warmup:
                timings.append((time.perf_counter() - start) * 1000)
        value = torch.tensor([statistics.median(timings)], device=info.device)
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
        if info.rank == 0:
            print(f"DDP median_ms_avg={value.item():.3f}")
    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_ddp(args)
    else:
        run_dataparallel(args)
        print("Run with torchrun --nproc_per_node=2 dataparallel_vs_ddp.py to measure DDP.")


if __name__ == "__main__":
    main()
