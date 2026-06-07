"""Pinned-prefetch MLP optimization: pinned-memory prefetch + overlap.

This benchmark demonstrates efficient host-to-device (H2D) data loading using:

- pinned host memory
- non-blocking H2D copies
- a dedicated CUDA copy stream
- double-buffered device-side staging buffers
- stream/event-based synchronization

The model architecture intentionally matches the baseline MLP so that performance
differences primarily come from the data-loading path rather than the model itself.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.optimization.allocator_tuning import log_allocator_guidance
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class Prefetcher:
    """Double-buffered prefetcher from pinned host memory to device.

    The core idea is:

        copy_stream:     H2D copy for batch N+1 --------------->
        compute stream:          forward/backward for batch N --->

    This hides part of the H2D copy latency behind GPU compute when the workload
    has enough compute to overlap with transfer.
    """

    def __init__(
        self,
        device: torch.device,
        host_batches: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Prefetcher requires CUDA")

        if not host_batches:
            raise ValueError("host_batches must be non-empty")
        if not targets:
            raise ValueError("targets must be non-empty")
        if len(host_batches) != len(targets):
            raise ValueError(
                f"host_batches and targets must have the same length: "
                f"{len(host_batches)} != {len(targets)}"
            )

        first_input = host_batches[0]
        first_target = targets[0]

        for i, batch in enumerate(host_batches):
            if batch.shape != first_input.shape:
                raise ValueError(
                    f"All host_batches must have the same shape. "
                    f"batch[0]={first_input.shape}, batch[{i}]={batch.shape}"
                )
            if batch.dtype != first_input.dtype:
                raise ValueError(
                    f"All host_batches must have the same dtype. "
                    f"batch[0]={first_input.dtype}, batch[{i}]={batch.dtype}"
                )
            if not batch.is_pinned():
                raise ValueError(
                    "All host_batches must be allocated with pin_memory=True "
                    "for non_blocking H2D copies to be effective"
                )

        for i, target in enumerate(targets):
            if target.shape != first_target.shape:
                raise ValueError(
                    f"All targets must have the same shape. "
                    f"target[0]={first_target.shape}, target[{i}]={target.shape}"
                )
            if target.dtype != first_target.dtype:
                raise ValueError(
                    f"All targets must have the same dtype. "
                    f"target[0]={first_target.dtype}, target[{i}]={target.dtype}"
                )
            if not target.is_pinned():
                raise ValueError(
                    "All targets must be allocated with pin_memory=True "
                    "for non_blocking H2D copies to be effective"
                )

        self.device = device
        self.host_batches = host_batches
        self.targets = targets
        self.num_batches = len(host_batches)

        self.copy_stream = torch.cuda.Stream(device=device)

        # Two device-side slots:
        # - one slot is consumed by the compute stream
        # - the other slot is refilled by the copy stream
        self.input_bufs = [
            torch.empty_like(first_input, device=device),
            torch.empty_like(first_input, device=device),
        ]
        self.target_bufs = [
            torch.empty_like(first_target, device=device),
            torch.empty_like(first_target, device=device),
        ]

        # One event per slot. Each event marks when that slot's H2D copy is ready.
        self.ready_events = [
            torch.cuda.Event(blocking=False),
            torch.cuda.Event(blocking=False),
        ]

        self.consume_slot = 0
        self.batch_idx = 0

        # Prime both slots before benchmark iterations begin.
        self._enqueue_prefetch(slot=0)
        self._enqueue_prefetch(slot=1)

    def _enqueue_prefetch(self, slot: int) -> None:
        """Enqueue H2D copies into a device-side buffer slot."""
        host_idx = self.batch_idx % self.num_batches
        self.batch_idx += 1

        with torch.cuda.stream(self.copy_stream):
            self.input_bufs[slot].copy_(
                self.host_batches[host_idx],
                non_blocking=True,
            )
            self.target_bufs[slot].copy_(
                self.targets[host_idx],
                non_blocking=True,
            )
            self.ready_events[slot].record(self.copy_stream)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next prefetched device batch.

        The current compute stream waits only for the buffer slot it is about to
        consume. It does not globally synchronize the GPU.
        """
        current_stream = torch.cuda.current_stream(self.device)

        slot = self.consume_slot
        current_stream.wait_event(self.ready_events[slot])

        inputs = self.input_bufs[slot]
        targets = self.target_bufs[slot]

        # Alternate between slot 0 and slot 1.
        self.consume_slot = 1 - self.consume_slot

        # Refill the slot we just handed to the caller.
        #
        # Because this refill is enqueued on copy_stream, it can overlap with
        # compute on the current stream. The caller must finish using this slot
        # before it is reused in a later iteration. With two slots and sequential
        # benchmark iteration execution, this is safe for this workload pattern.
        self._enqueue_prefetch(slot=slot)

        return inputs, targets


class OptimizedPinnedPrefetchMLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pinned-memory prefetch benchmark using the baseline MLP workload."""

    def __init__(self) -> None:
        super().__init__()

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.prefetcher: Optional[Prefetcher] = None

        self.output: Optional[torch.Tensor] = None
        self._payload_inputs: Optional[torch.Tensor] = None
        self._payload_targets: Optional[torch.Tensor] = None

        # Training benchmarks do not support strict jitter checks because
        # weights change after every optimizer step.
        #
        # Larger transfers make H2D optimization measurable on high-bandwidth
        # GPUs. The prefetch benefit is roughly proportional to:
        #
        #     H2D copy time / compute time
        from core.benchmark.smoke import is_smoke_mode

        low_mem = is_smoke_mode()

        self.input_dim = 2048 if low_mem else 4096
        self.hidden_dim = 2048 if low_mem else 4096
        self.output_dim = 1024 if low_mem else 2048
        self.batch_size = 512 if low_mem else 1024
        self.num_batches = 4 if low_mem else 8

        # H2D bytes include both input and target tensors.
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(
                self.batch_size * (self.input_dim + self.output_dim) * 4
            ),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        log_allocator_guidance(
            "ch03/optimized_pinned_prefetch_mlp",
            optimized=True,
        )

        # Same model architecture as the baseline for fair comparison.
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

        # Pinned host memory is required for effective async H2D copy with
        # non_blocking=True.
        for _ in range(self.num_batches):
            self.host_batches.append(
                torch.randn(
                    self.batch_size,
                    self.input_dim,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            )
            self.targets.append(
                torch.randn(
                    self.batch_size,
                    self.output_dim,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            )

        self.prefetcher = Prefetcher(
            device=self.device,
            host_batches=self.host_batches,
            targets=self.targets,
        )

        # One setup-time synchronization is fine. It keeps warmup/benchmark
        # measurements from including initial buffer priming.
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.optimizer is not None
        assert self.prefetcher is not None

        inputs, targets = self.prefetcher.next()

        with nvtx_range("optimized_pinned_prefetch_mlp", enable=enable_nvtx):
            out = self.model(inputs)
            loss = torch.nn.functional.mse_loss(out, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # Keep a detached output reference for verification.
        #
        # Do not clone here by default because clone would add extra GPU work to
        # the benchmark timing. Inputs/targets are cloned later in
        # capture_verification_payload() where correctness artifacts are captured.
        self.output = out.detach()
        self._payload_inputs = inputs.detach()
        self._payload_targets = targets.detach()

    def capture_verification_payload(self) -> None:
        if self.model is None:
            raise RuntimeError("model is not initialized")
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before capture_verification_payload()")
        if self._payload_inputs is None or self._payload_targets is None:
            raise RuntimeError("benchmark_fn() must be called before capture_verification_payload()")

        # Clone here, not inside benchmark_fn(), so verification payload capture
        # does not distort benchmark timing.
        #
        # This matters because the prefetcher reuses double-buffered device memory.
        # Without cloning, a later prefetch can overwrite the tensor storage
        # referenced by the payload.
        payload_inputs = self._payload_inputs.detach().clone()
        payload_targets = self._payload_targets.detach().clone()
        payload_output = self.output.detach().clone()

        self._set_verification_payload(
            inputs={
                "data": payload_inputs,
                "target": payload_targets,
            },
            output=payload_output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                    if torch.cuda.is_available()
                    else False
                ),
            },
            # Training updates weights every iteration, so tolerate wider output
            # drift than pure inference benchmarks.
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None

        self.host_batches = []
        self.targets = []
        self.prefetcher = None

        self.output = None
        self._payload_inputs = None
        self._payload_targets = None

        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        from core.benchmark.smoke import is_smoke_mode

        low_mem = is_smoke_mode()
        return BenchmarkConfig(
            iterations=5 if low_mem else 20,
            warmup=5 if low_mem else 10,
        )

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        if self.prefetcher is None:
            return []
        return [self.prefetcher.copy_stream]

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics

        return compute_system_config_metrics(
            numa_nodes=getattr(self, "numa_nodes", 0),
            cpu_cores=getattr(self, "cpu_cores", 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.prefetcher is None:
            return "Prefetcher not initialized"

        if not self.host_batches:
            return "host_batches is empty"

        if not self.targets:
            return "targets is empty"

        if len(self.host_batches) != len(self.targets):
            return "host_batches and targets length mismatch"

        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPinnedPrefetchMLPBenchmark()
