"""Compiled comparison GEMM benchmark using torch.compile.

This benchmark is a supplementary host/runtime comparison workload.

The goal is not to claim a NUMA, OS, or math-kernel optimization. Instead, this
variant measures a stable-shape GEMM path wrapped with
`torch.compile(mode="reduce-overhead")` so the benchmark can compare runtime
and launch-overhead behavior against a baseline launch pattern while keeping the
mathematics fixed.

Important interpretation:
- A single large `torch.matmul` usually dispatches to cuBLAS/cuBLASLt already.
- Therefore, large speedups should not be interpreted as "cuBLAS was optimized."
- If the paired baseline uses many fragmented GEMM calls, most of the gain may
  come from changing the execution pattern from fragmented GEMMs to one
  monolithic GEMM call, not from torch.compile alone.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


CompiledMatmulFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class OptimizedGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Supplementary comparison workload with one compiled stable-shape GEMM."""

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "optimized",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "Supplementary comparison for host/runtime overhead. This does not "
            "claim a NUMA-local, OS-level, or math-kernel optimization; it "
            "isolates a stable-shape compiled GEMM path against the paired "
            "baseline launch pattern."
        ),
        "comparison_axis": "fragmented_vs_stable_compiled_gemm_path",
        "execution_pattern": "compiled_single_gemm_call",
        "optimization_mechanism": (
            'wrap a stable-shape torch.matmul with torch.compile(mode="reduce-overhead") '
            "to observe compiler/runtime launch-overhead behavior while keeping "
            "the GEMM math fixed"
        ),
        "interpretation_warning": (
            "A single large torch.matmul normally dispatches to cuBLAS/cuBLASLt. "
            "Large improvements over a fragmented baseline may primarily reflect "
            "fewer GEMM calls and fewer launches rather than a faster GEMM kernel."
        ),
        "chapter_native_targets": [
            "pageable_copy",
            "rack_prep",
            "pinned_prefetch_mlp",
            "double_buffered_batch_provisioning",
        ],
    }

    def __init__(self) -> None:
        super().__init__()

        # Matrix dimensions. These must match the paired baseline for verification.
        self.m = 2048
        self.n = 2048
        self.k = 2048

        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.fn: Optional[CompiledMatmulFn] = None

        # Preserve global PyTorch matmul settings so teardown can restore them.
        self._previous_allow_tf32: Optional[bool] = None
        self._previous_matmul_precision: Optional[str] = None

        # Register workload metadata for compliance checks.
        # Here "tokens" is used as a generic amount-of-work proxy.
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        """Initialize inputs and compile the stable-shape GEMM function."""
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Make TF32 behavior explicit for reproducibility.
        # This is appropriate for a performance-oriented FP32 GEMM benchmark on
        # NVIDIA Ampere+ GPUs. Verification tolerance is set accordingly.
        self._previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self._previous_matmul_precision = torch.get_float32_matmul_precision()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.left = torch.randn(
            self.m,
            self.k,
            device=self.device,
            dtype=torch.float32,
        )
        self.right = torch.randn(
            self.k,
            self.n,
            device=self.device,
            dtype=torch.float32,
        )
        self.output = None

        def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            raise RuntimeError("torch.compile is required for this benchmark")

        # fullgraph=True is intentional here. The function is deliberately simple
        # and should compile as one graph. If this fails, the benchmark should fail
        # loudly instead of silently falling back to a less controlled path.
        self.fn = compile_fn(
            matmul_fn,
            mode="reduce-overhead",
            fullgraph=True,
        )

        assert self.fn is not None
        assert self.left is not None and self.right is not None

        # Warm up compilation, CUDA Graph capture/replay if eligible, cuBLAS/cuBLASLt
        # algorithm selection, and allocator state.
        for _ in range(10):
            self.output = self.fn(self.left, self.right)

        self._synchronize()

    def benchmark_fn(self) -> None:
        """Compute C = A @ B using the compiled stable-shape GEMM path."""
        if self.left is None or self.right is None:
            raise RuntimeError("Input tensors are not initialized")
        if self.fn is None:
            raise RuntimeError("Compiled function is not initialized")

        with self._nvtx_range("optimized_gemm_compiled_single_call"):
            self.output = self.fn(self.left, self.right)

    def capture_verification_payload(self) -> None:
        """Capture inputs and output for numerical verification."""
        if self.left is None or self.right is None or self.output is None:
            raise RuntimeError("Benchmark tensors/output are not initialized")

        self._set_verification_payload(
            inputs={
                "left": self.left,
                "right": self.right,
            },
            output=self.output.detach().clone(),
            batch_size=self.left.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": True,
            },
            # TF32 may introduce small numerical differences versus strict FP32.
            output_tolerance=(1e-4, 1e-3),
        )

    def teardown(self) -> None:
        """Release tensors and restore global PyTorch matmul settings."""
        self.left = None
        self.right = None
        self.output = None
        self.fn = None

        if self._previous_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._previous_allow_tf32

        if self._previous_matmul_precision is not None:
            torch.set_float32_matmul_precision(self._previous_matmul_precision)

        self._previous_allow_tf32 = None
        self._previous_matmul_precision = None

        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics

        metrics = compute_system_config_metrics(
            numa_nodes=getattr(self, "numa_nodes", 0),
            cpu_cores=getattr(self, "cpu_cores", 64),
        )

        metrics.update(
            {
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
                "launch.gemm_calls_per_iteration": 1.0,
                "launch.block_k": float(self.k),
                "compile.enabled": 1.0,
                "compile.mode_reduce_overhead": 1.0,
                "compile.fullgraph": 1.0,
                "compile.static_shapes": 1.0,
                "compile.expected_cudagraph_eligible": 1.0,
                "math.m": float(self.m),
                "math.n": float(self.n),
                "math.k": float(self.k),
                "math.dtype_fp32": 1.0,
                "math.tf32_allowed": 1.0,
            }
        )

        return metrics

    def get_optimization_goal(self) -> str:
        """Keep this as a supplementary comparison workload."""
        return "comparison"

    def validate_result(self) -> Optional[str]:
        if self.fn is None:
            return "Compiled function not initialized"
        if self.left is None or self.right is None:
            return "Input tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmBenchmark()
