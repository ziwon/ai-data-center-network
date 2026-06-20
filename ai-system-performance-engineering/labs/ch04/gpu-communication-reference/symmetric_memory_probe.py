from __future__ import annotations

import argparse
import sys

import torch

from common import positive_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe PyTorch symmetric-memory availability.")
    parser.add_argument("--size-mb", type=positive_int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("SKIPPED: symmetric_memory_probe requires CUDA", file=sys.stderr)
        raise SystemExit(0)

    symmetric_memory = getattr(torch.distributed, "_symmetric_memory", None)
    if symmetric_memory is None:
        print("SKIPPED: torch.distributed._symmetric_memory is not available in this PyTorch build")
        raise SystemExit(0)

    tensor = torch.empty(args.size_mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    print(f"allocated_cuda_tensor_mb={args.size_mb}")
    print("symmetric_memory_module_available=True")
    print("Use the PyTorch/NVSHMEM build instructions for real symmetric allocation and rendezvous.")
    del tensor


if __name__ == "__main__":
    main()
