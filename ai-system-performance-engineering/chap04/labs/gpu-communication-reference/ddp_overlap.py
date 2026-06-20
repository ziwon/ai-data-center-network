from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from common import (
    TinyStack,
    add_training_args,
    benchmark_steps,
    cleanup_distributed,
    make_batch,
    setup_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP path that can overlap gradient buckets with backward.")
    add_training_args(parser)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = setup_distributed("ddp_overlap")

    try:
        torch.manual_seed(42 + info.rank)
        model = TinyStack(args.hidden_size, args.layers).to(info.device)
        model = DDP(
            model,
            device_ids=[info.local_rank],
            output_device=info.local_rank,
            bucket_cap_mb=args.bucket_cap_mb,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs, targets = make_batch(args.batch_size, args.hidden_size, info.device)

        def step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.detach()

        benchmark_steps(
            name="DDP gradient bucket overlap",
            info=info,
            warmup=max(args.warmup, 0),
            iterations=args.iterations,
            step_fn=step,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
