#!/usr/bin/env python3
"""CUDA Green Context microbenchmark using cuda-python Driver API.

This benchmark compares the same simple CUDA kernel under:

1. a normal CUDA context that can use the full GPU, and
2. a CUDA Green Context that is restricted to a selected number of SMs.

The goal is not to benchmark a production LLM kernel. The goal is to make SM
partitioning visible and measurable.

Requirements:
  - CUDA driver/runtime with Green Context support
  - cuda-python bindings that expose Green Context Driver APIs
  - numpy

Example:
  python cuda_green_context_bench.py --sm-fraction 0.5 --iterations 1000 --verify

Notes:
  - CUDA event timing is used by default for GPU-side elapsed time.
  - Wall-clock timing is also available for observing Python/driver launch overhead.
  - The kernel is intentionally simple and mostly memory-bound.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from cuda.bindings import driver, nvrtc


KERNEL_CODE = r"""
extern "C" __global__ void scale(float* data, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= alpha;
    }
}
"""


@dataclass(frozen=True)
class BenchConfig:
    device_index: int
    elements: int
    iterations: int
    warmup: int
    block_size: int
    alpha: float
    sm_fraction: float
    repeats: int
    timing: str
    verify: bool
    arch: Optional[str]


@dataclass(frozen=True)
class BenchResult:
    label: str
    elapsed_s_mean: float
    elapsed_s_stdev: float
    elapsed_s_runs: tuple[float, ...]
    sample: float
    verified: Optional[bool]


@dataclass(frozen=True)
class GreenContextInfo:
    green: driver.CUgreenCtx
    ctx: driver.CUcontext
    stream: driver.CUstream
    requested_sms: int
    allocated_sms: int
    total_sms: int
    min_partition: int
    alignment: int


def _cuda_check(err: driver.CUresult, msg: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        _, err_str = driver.cuGetErrorString(err)
        err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        raise RuntimeError(f"{msg}: {err_msg}")


def _nvrtc_check(err: nvrtc.nvrtcResult, msg: str) -> None:
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err_str = nvrtc.nvrtcGetErrorString(err)
        err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        raise RuntimeError(f"{msg}: {err_msg}")


def _compile_ptx(device: driver.CUdevice, arch_override: Optional[str] = None) -> bytes:
    """Compile CUDA C source to PTX.

    Use a virtual architecture target such as compute_90 for PTX generation.
    This keeps the PTX JIT-able by the CUDA driver for the current GPU.
    """
    err, prog = nvrtc.nvrtcCreateProgram(
        KERNEL_CODE.encode(),
        b"green_context_bench.cu",
        0,
        None,
        None,
    )
    _nvrtc_check(err, "nvrtcCreateProgram failed")

    try:
        err, major, minor = driver.cuDeviceComputeCapability(device)
        _cuda_check(err, "cuDeviceComputeCapability failed")

        arch = arch_override or f"compute_{major}{minor}"
        if arch.startswith("sm_"):
            # nvrtcGetPTX is most naturally paired with compute_XX.
            # Accept sm_XX from the user but convert it to compute_XX.
            arch = "compute_" + arch.removeprefix("sm_")

        options = [
            f"--gpu-architecture={arch}".encode(),
            b"--std=c++17",
            b"--use_fast_math",
        ]

        err, = nvrtc.nvrtcCompileProgram(prog, len(options), options)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(prog, log)
            _nvrtc_check(err, f"nvrtcCompileProgram failed:\n{bytes(log).decode(errors='replace').strip()}")

        err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        _nvrtc_check(err, "nvrtcGetPTXSize failed")

        ptx = bytearray(ptx_size)
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        _nvrtc_check(err, "nvrtcGetPTX failed")

        return bytes(ptx)
    finally:
        err, = nvrtc.nvrtcDestroyProgram(prog)
        _nvrtc_check(err, "nvrtcDestroyProgram failed")


def _load_kernel(ptx: bytes) -> tuple[driver.CUmodule, driver.CUfunction]:
    err, module = driver.cuModuleLoadData(ptx)
    _cuda_check(err, "cuModuleLoadData failed")

    try:
        err, func = driver.cuModuleGetFunction(module, b"scale")
        _cuda_check(err, "cuModuleGetFunction failed")
        return module, func
    except Exception:
        _cuda_check(driver.cuModuleUnload(module)[0], "cuModuleUnload cleanup failed")
        raise


def _make_kernel_params(dptr: driver.CUdeviceptr, elements: int, alpha: float) -> ctypes.Array:
    arg0 = ctypes.c_void_p(int(dptr))
    arg1 = ctypes.c_int(elements)
    arg2 = ctypes.c_float(alpha)

    return (ctypes.c_void_p * 3)(
        ctypes.addressof(arg0),
        ctypes.addressof(arg1),
        ctypes.addressof(arg2),
    )


def _launch_scale(
    func: driver.CUfunction,
    stream: driver.CUstream | int,
    dptr: driver.CUdeviceptr,
    elements: int,
    block_size: int,
    alpha: float,
) -> None:
    params = _make_kernel_params(dptr, elements, alpha)
    grid = (elements + block_size - 1) // block_size

    err, = driver.cuLaunchKernel(
        func,
        grid, 1, 1,
        block_size, 1, 1,
        0,
        stream,
        params,
        0,
    )
    _cuda_check(err, "cuLaunchKernel failed")


def _sync(stream: driver.CUstream | int) -> None:
    if int(stream) != 0:
        _cuda_check(driver.cuStreamSynchronize(stream)[0], "cuStreamSynchronize failed")
    else:
        _cuda_check(driver.cuCtxSynchronize()[0], "cuCtxSynchronize failed")


def _measure_with_events(
    func: driver.CUfunction,
    stream: driver.CUstream | int,
    dptr: driver.CUdeviceptr,
    elements: int,
    iterations: int,
    block_size: int,
    alpha: float,
) -> float:
    err, start = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)
    _cuda_check(err, "cuEventCreate(start) failed")
    err, stop = driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT)
    _cuda_check(err, "cuEventCreate(stop) failed")

    try:
        _cuda_check(driver.cuEventRecord(start, stream)[0], "cuEventRecord(start) failed")
        for _ in range(iterations):
            _launch_scale(func, stream, dptr, elements, block_size, alpha)
        _cuda_check(driver.cuEventRecord(stop, stream)[0], "cuEventRecord(stop) failed")
        _cuda_check(driver.cuEventSynchronize(stop)[0], "cuEventSynchronize(stop) failed")

        err, elapsed_ms = driver.cuEventElapsedTime(start, stop)
        _cuda_check(err, "cuEventElapsedTime failed")
        return float(elapsed_ms) / 1_000.0
    finally:
        _cuda_check(driver.cuEventDestroy(stop)[0], "cuEventDestroy(stop) failed")
        _cuda_check(driver.cuEventDestroy(start)[0], "cuEventDestroy(start) failed")


def _measure_with_wall_clock(
    func: driver.CUfunction,
    stream: driver.CUstream | int,
    dptr: driver.CUdeviceptr,
    elements: int,
    iterations: int,
    block_size: int,
    alpha: float,
) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        _launch_scale(func, stream, dptr, elements, block_size, alpha)
    _sync(stream)
    return time.perf_counter() - start


def _run_kernel_once(
    *,
    ctx: driver.CUcontext,
    stream: driver.CUstream | int,
    ptx: bytes,
    config: BenchConfig,
) -> tuple[float, float, Optional[bool]]:
    _cuda_check(driver.cuCtxSetCurrent(ctx)[0], "cuCtxSetCurrent failed")
    module, func = _load_kernel(ptx)

    dptr: Optional[driver.CUdeviceptr] = None
    try:
        host = np.random.rand(config.elements).astype(np.float32)

        err, dptr = driver.cuMemAlloc(host.nbytes)
        _cuda_check(err, "cuMemAlloc failed")

        err, = driver.cuMemcpyHtoD(dptr, host.ctypes.data, host.nbytes)
        _cuda_check(err, "cuMemcpyHtoD failed")

        for _ in range(config.warmup):
            _launch_scale(func, stream, dptr, config.elements, config.block_size, config.alpha)
        _sync(stream)

        if config.timing == "event":
            elapsed = _measure_with_events(
                func, stream, dptr, config.elements, config.iterations,
                config.block_size, config.alpha,
            )
        elif config.timing == "wall":
            elapsed = _measure_with_wall_clock(
                func, stream, dptr, config.elements, config.iterations,
                config.block_size, config.alpha,
            )
        else:
            raise ValueError(f"unknown timing mode: {config.timing}")

        out = np.empty_like(host)
        err, = driver.cuMemcpyDtoH(out.ctypes.data, dptr, host.nbytes)
        _cuda_check(err, "cuMemcpyDtoH failed")

        verified: Optional[bool] = None
        if config.verify:
            expected = host * np.float32(config.alpha ** (config.warmup + config.iterations))
            verified = bool(np.allclose(out, expected, rtol=2e-4, atol=2e-5))

        return elapsed, float(out[0]), verified
    finally:
        if dptr is not None:
            _cuda_check(driver.cuMemFree(dptr)[0], "cuMemFree failed")
        _cuda_check(driver.cuModuleUnload(module)[0], "cuModuleUnload failed")


def _run_benchmark(
    *,
    label: str,
    ctx: driver.CUcontext,
    stream: driver.CUstream | int,
    ptx: bytes,
    config: BenchConfig,
) -> BenchResult:
    elapsed_runs: list[float] = []
    sample = float("nan")
    verified_values: list[bool] = []

    for _ in range(config.repeats):
        elapsed, sample, verified = _run_kernel_once(
            ctx=ctx,
            stream=stream,
            ptx=ptx,
            config=config,
        )
        elapsed_runs.append(elapsed)
        if verified is not None:
            verified_values.append(verified)

    stdev = statistics.stdev(elapsed_runs) if len(elapsed_runs) > 1 else 0.0
    verified_result = all(verified_values) if verified_values else None

    return BenchResult(
        label=label,
        elapsed_s_mean=statistics.mean(elapsed_runs),
        elapsed_s_stdev=stdev,
        elapsed_s_runs=tuple(elapsed_runs),
        sample=sample,
        verified=verified_result,
    )


def _round_up_to_alignment(value: int, alignment: int) -> int:
    return int(math.ceil(value / alignment) * alignment)


def _create_green_context(
    device: driver.CUdevice,
    sm_fraction: float,
) -> GreenContextInfo:
    err, sm_res = driver.cuDeviceGetDevResource(
        device,
        driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
    )
    _cuda_check(err, "cuDeviceGetDevResource(SM) failed")

    total_sms = int(sm_res.sm.smCount)
    min_partition = int(sm_res.sm.minSmPartitionSize)
    alignment = int(sm_res.sm.smCoscheduledAlignment) or min_partition

    requested_sms = max(min_partition, int(math.ceil(total_sms * sm_fraction)))
    desired_sms = _round_up_to_alignment(requested_sms, alignment)
    desired_sms = min(desired_sms, total_sms)

    err, groups, _, _ = driver.cuDevSmResourceSplitByCount(1, sm_res, 0, desired_sms)
    _cuda_check(err, "cuDevSmResourceSplitByCount failed")

    err, desc = driver.cuDevResourceGenerateDesc(groups, len(groups))
    _cuda_check(err, "cuDevResourceGenerateDesc failed")

    flags = driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    err, green = driver.cuGreenCtxCreate(desc, device, flags)
    _cuda_check(err, "cuGreenCtxCreate failed")

    try:
        err, ctx = driver.cuCtxFromGreenCtx(green)
        _cuda_check(err, "cuCtxFromGreenCtx failed")

        err, stream = driver.cuGreenCtxStreamCreate(
            green,
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,
        )
        _cuda_check(err, "cuGreenCtxStreamCreate failed")

        return GreenContextInfo(
            green=green,
            ctx=ctx,
            stream=stream,
            requested_sms=requested_sms,
            allocated_sms=desired_sms,
            total_sms=total_sms,
            min_partition=min_partition,
            alignment=alignment,
        )
    except Exception:
        _cuda_check(driver.cuGreenCtxDestroy(green)[0], "cuGreenCtxDestroy cleanup failed")
        raise


def _destroy_green_context(info: GreenContextInfo) -> None:
    # A stream created from a green context is still a CUstream handle.
    # Destroy it explicitly before destroying the owning green context.
    if int(info.stream) != 0:
        _cuda_check(driver.cuStreamDestroy(info.stream)[0], "cuStreamDestroy(green stream) failed")
    _cuda_check(driver.cuGreenCtxDestroy(info.green)[0], "cuGreenCtxDestroy failed")


def _print_device_info(device: driver.CUdevice) -> None:
    err, name = driver.cuDeviceGetName(128, device)
    _cuda_check(err, "cuDeviceGetName failed")

    err, major, minor = driver.cuDeviceComputeCapability(device)
    _cuda_check(err, "cuDeviceComputeCapability failed")

    err, sm_count = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        device,
    )
    _cuda_check(err, "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed")

    decoded_name = name.decode(errors="replace").rstrip("\x00")
    print(f"Device: {decoded_name}")
    print(f"Compute capability: {major}.{minor}")
    print(f"Reported SM count: {sm_count}")


def _validate_config(args: argparse.Namespace) -> BenchConfig:
    if args.elements <= 0:
        raise ValueError("--elements must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.block_size <= 0 or args.block_size % 32 != 0:
        raise ValueError("--block-size must be a positive multiple of 32")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not (0.0 < args.sm_fraction <= 1.0):
        raise ValueError("--sm-fraction must be within (0, 1]")
    if args.alpha <= 0.0:
        raise ValueError("--alpha must be positive")

    return BenchConfig(
        device_index=args.device,
        elements=args.elements,
        iterations=args.iterations,
        warmup=args.warmup,
        block_size=args.block_size,
        alpha=args.alpha,
        sm_fraction=args.sm_fraction,
        repeats=args.repeats,
        timing=args.timing,
        verify=args.verify,
        arch=args.arch,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA Green Context SM-partition microbenchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--elements", type=int, default=1_048_576, help="Vector length.")
    parser.add_argument("--iterations", type=int, default=100, help="Timed kernel iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup kernel launches.")
    parser.add_argument("--block-size", type=int, default=256, help="CUDA threads per block; use multiples of 32.")
    parser.add_argument("--alpha", type=float, default=1.0001, help="Scale factor.")
    parser.add_argument("--sm-fraction", type=float, default=0.5, help="Fraction of SMs requested for the green context.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of benchmark repeats.")
    parser.add_argument(
        "--timing",
        choices=("event", "wall"),
        default="event",
        help="Use CUDA event time or CPU wall-clock time.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify output against expected result.")
    parser.add_argument(
        "--arch",
        default=None,
        help="Optional NVRTC virtual architecture override, e.g. compute_90. sm_90 is accepted and converted.",
    )
    return parser


def _format_result(result: BenchResult) -> str:
    runs = ", ".join(f"{v:.6f}" for v in result.elapsed_s_runs)
    verify_text = "n/a" if result.verified is None else str(result.verified)
    return (
        f"{result.label}:\n"
        f"  mean:      {result.elapsed_s_mean:.6f} s\n"
        f"  stdev:     {result.elapsed_s_stdev:.6f} s\n"
        f"  runs:      [{runs}]\n"
        f"  sample:    {result.sample:.6f}\n"
        f"  verified:  {verify_text}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _validate_config(args)

    err, = driver.cuInit(0)
    _cuda_check(err, "cuInit failed")

    err, dev = driver.cuDeviceGet(config.device_index)
    _cuda_check(err, "cuDeviceGet failed")

    _print_device_info(dev)

    print("\nCompiling CUDA kernel to PTX...")
    ptx = _compile_ptx(dev, config.arch)
    print(f"PTX size: {len(ptx):,} bytes")

    err, base_ctx = driver.cuCtxCreate(None, 0, dev)
    _cuda_check(err, "cuCtxCreate(default) failed")

    try:
        base = _run_benchmark(
            label="Default context",
            ctx=base_ctx,
            stream=0,
            ptx=ptx,
            config=config,
        )
    finally:
        _cuda_check(driver.cuCtxDestroy(base_ctx)[0], "cuCtxDestroy(default) failed")

    green_info: Optional[GreenContextInfo] = None
    try:
        green_info = _create_green_context(dev, config.sm_fraction)
        green = _run_benchmark(
            label="Green context",
            ctx=green_info.ctx,
            stream=green_info.stream,
            ptx=ptx,
            config=config,
        )
    finally:
        if green_info is not None:
            _destroy_green_context(green_info)

    slowdown = green.elapsed_s_mean / base.elapsed_s_mean if base.elapsed_s_mean > 0 else float("inf")
    throughput_base = (config.elements * config.iterations) / base.elapsed_s_mean
    throughput_green = (config.elements * config.iterations) / green.elapsed_s_mean

    print("\nCUDA Green Context Benchmark")
    print(f"Timing mode: {config.timing}")
    print(f"Elements: {config.elements:,}")
    print(f"Iterations per repeat: {config.iterations:,}")
    print(f"Warmup launches: {config.warmup:,}")
    print(f"Block size: {config.block_size}")
    print(f"Repeats: {config.repeats}")

    if green_info is not None:
        print("\nGreen Context SM Allocation")
        print(f"  requested fraction:     {config.sm_fraction:.4f}")
        print(f"  requested SMs:          {green_info.requested_sms}/{green_info.total_sms}")
        print(f"  allocated SMs:          {green_info.allocated_sms}/{green_info.total_sms}")
        print(f"  effective fraction:     {green_info.allocated_sms / green_info.total_sms:.4f}")
        print(f"  min partition size:     {green_info.min_partition}")
        print(f"  co-schedule alignment:  {green_info.alignment}")

    print("\nResults")
    print(_format_result(base))
    print(_format_result(green))

    print("\nDerived Metrics")
    print(f"  default throughput:     {throughput_base:,.0f} elements/s")
    print(f"  green throughput:       {throughput_green:,.0f} elements/s")
    print(f"  slowdown vs default:    {slowdown:.3f}x")

    if config.verify and not (base.verified and green.verified):
        print("\nVerification failed.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
