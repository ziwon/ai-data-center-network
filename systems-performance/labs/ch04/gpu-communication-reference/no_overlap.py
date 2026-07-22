from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import (
    TinyStack,
    add_training_args,
    benchmark_steps,
    cleanup_distributed,
    make_batch,
    setup_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP-style baseline with communication exposed after backward.")
    add_training_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = setup_distributed("no_overlap")

    try:
        torch.manual_seed(42 + info.rank)
        model = TinyStack(args.hidden_size, args.layers).to(info.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs, targets = make_batch(args.batch_size, args.hidden_size, info.device)

        def step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()

            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.mul_(1.0 / info.world_size)

            optimizer.step()
            return loss.detach()

        benchmark_steps(
            name="manual all-reduce after backward",
            info=info,
            warmup=max(args.warmup, 0),
            iterations=args.iterations,
            step_fn=step,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
