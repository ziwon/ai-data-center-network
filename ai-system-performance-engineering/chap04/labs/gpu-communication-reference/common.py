from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


class TinyStack(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 6) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def setup_distributed(name: str) -> DistInfo:
    if not torch.cuda.is_available():
        print(f"SKIPPED: {name} requires CUDA", file=sys.stderr)
        raise SystemExit(0)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        print(f"SKIPPED: {name} requires torchrun with at least 2 ranks", file=sys.stderr)
        raise SystemExit(0)

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 2:
        print(f"SKIPPED: {name} requires at least 2 visible GPUs", file=sys.stderr)
        raise SystemExit(0)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return DistInfo(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def make_batch(batch_size: int, hidden_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(batch_size, hidden_size, device=device)
    targets = torch.randn(batch_size, 1, device=device)
    return inputs, targets


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def benchmark_steps(
    *,
    name: str,
    info: DistInfo,
    warmup: int,
    iterations: int,
    step_fn: Callable[[], torch.Tensor],
) -> None:
    for _ in range(warmup):
        loss = step_fn()
        del loss

    torch.cuda.synchronize(info.device)
    timings: list[float] = []
    last_loss = torch.zeros((), device=info.device)

    for _ in range(iterations):
        start = time.perf_counter()
        last_loss = step_fn()
        torch.cuda.synchronize(info.device)
        timings.append((time.perf_counter() - start) * 1000)

    loss_value = last_loss.detach().float().clone()
    dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)

    if info.rank == 0:
        print(f"{name}")
        print(f"{'metric':<16} {'value':>12}")
        print("-" * 30)
        print(f"{'world_size':<16} {info.world_size:>12}")
        print(f"{'median_ms':<16} {statistics.median(timings):>12.3f}")
        print(f"{'p95_ms':<16} {percentile(timings, 95):>12.3f}")
        print(f"{'loss_avg':<16} {loss_value.item():>12.6f}")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--hidden-size", type=positive_int, default=1024)
    parser.add_argument("--layers", type=positive_int, default=6)
