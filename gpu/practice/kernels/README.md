# Kernel Case Studies (practice)

References and backlog for hands-on GPU kernel case studies — the *do* half of
this track. Learn the patterns in [PMPP](../../courses/pmpp/README.md) and
[Heterogeneous Systems](../../courses/heterogeneous-systems/README.md), then
implement and profile them here with [Profiling](../profiling/README.md).

## Scope

Focus on kernels and runtime paths that explain real inference behavior:

- memory coalescing, shared-memory tiling, and occupancy
- reduction, scan, histogram, convolution, and sparse traversal patterns
- GEMM, GEMV, fused softmax, attention, and low-bit GEMM
- Triton, CUTLASS, CUDA C++, and production inference kernels
- Nsight Compute evidence: SM utilization, memory throughput, L2 hit rate, Tensor Core usage, warp stalls

Related background: [PMPP](../../courses/pmpp/README.md), [Profiling](../profiling/README.md)

## Resource Index

Resources here are grouped by media type.

### YouTube / Lecture Series

#### Selected Kernel Lectures from GPU MODE

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)
- [GPU MODE YouTube channel](https://www.youtube.com/@GPUMODE/videos)

GPU MODE itself is tracked in [GPU MODE](../../courses/gpu-mode/README.md). This section only pulls out the lectures that are directly useful for kernel case studies.

High-value kernel lectures:

| Topic | Why it matters |
|---|---|
| Lecture 12: Flash Attention | attention tiling, SRAM reuse, avoiding materialized attention matrices |
| Lecture 14: Practitioner's Guide to Triton | practical Triton kernel authoring |
| Lecture 15: CUTLASS | GEMM abstractions, layouts, tiling, Tensor Core usage |
| Lecture 18: Fused Kernels | why fusing removes memory traffic and launch overhead |
| Lecture 22: Hacker's Guide to Speculative Decoding in vLLM | decode scheduler and verification path intuition |
| Lecture 23: Tensor Cores | why shapes and precision decide kernel paths |
| Lecture 29: Triton Internals | how Triton lowers kernels and where performance comes from |
| Lecture 34: Low Bit Triton Kernels | quantized weights, packing, dequantization, low-bit matmul |
| Lecture 36: CUTLASS and FlashAttention 3 | modern attention kernels and CUTLASS/CuTe style |
| Lecture 37: Introduction to SASS and GPU Microarchitecture | what the compiler actually emits |
| Lecture 40: FlashInfer | inference-time attention and serving kernels |
| Lecture 57: CuTe | CUTLASS layout algebra and modern GEMM construction |

#### Parallel patterns

The classic parallel-pattern building blocks (reduction, scan, histogram,
convolution, sparse, graph) are covered as a course in
[Heterogeneous Systems (Onur Mutlu)](../../courses/heterogeneous-systems/README.md).
Each meeting there maps to a case study in the backlog below.

#### CUDA Crash Course

- [CUDA Crash Course](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU): useful for basic CUDA kernel patterns.

### Articles / Worklogs

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM): the best single worklog for learning matmul optimization by iteration.

### Websites / Docs

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/): start with vector add, fused softmax, and matmul.
- [NVIDIA CUTLASS documentation](https://github.com/NVIDIA/cutlass): start with the repository docs and examples.
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/): metric definitions and profiling workflow.

### Repositories

- [CoffeeBeforeArch CUDA examples](https://github.com/CoffeeBeforeArch/cuda_programming): small CUDA kernels for vector add, matmul, reduction, histogram, and convolution.
- [CUTLASS](https://github.com/NVIDIA/cutlass): NVIDIA's template library for high-performance GEMM and related kernels.
- [vLLM](https://github.com/vllm-project/vllm): production serving kernels and scheduler/runtime integration.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): attention kernels built around tiling and IO awareness.

### Books

- [PMPP](../../courses/pmpp/README.md): background text for basic kernel patterns.

### Podcasts / Ongoing Feeds

- No separate kernel podcast is tracked yet. GPU MODE works as the ongoing feed for modern ML kernel topics.

## Topic Sequence

1. **Vector add:** launch shape, indexing, coalescing, timing.
2. **Reduction:** synchronization, shared memory, warp-level operations.
3. **Tiled matmul:** shared-memory tiling, arithmetic intensity, occupancy.
4. **Fused softmax:** row-wise reductions, numerical stability, memory traffic.
5. **FlashAttention:** attention without materializing the full score matrix.
6. **CUTLASS GEMM:** Tensor Core path, layouts, tile hierarchy.
7. **Triton matmul:** high-level kernel authoring and generated code inspection.
8. **Low-bit GEMM:** packing, scales, fused dequantization, Marlin/AWQ-style behavior.
9. **PagedAttention / FlashInfer:** inference-time attention and KV-cache layout.
10. **Speculative decoding verification path:** scheduler behavior, batch shape, target verification cost.

## Case Study Backlog

| Case study | Core question |
|---|---|
| CUDA vector add | Are global loads/stores coalesced? |
| Reduction | How does synchronization shape throughput? |
| Tiled matmul | When does arithmetic intensity become high enough? |
| GEMV | Why is batch=1 decode memory-bound or launch-bound? |
| Fused softmax | Why does fusion reduce memory traffic? |
| FlashAttention | How does tiling avoid `O(n^2)` HBM writes? |
| CUTLASS GEMM | How do Tensor Core layouts and tile shapes work? |
| Triton matmul | How close can a Python-authored kernel get? |
| AWQ Marlin | Why does fused dequantization matter? |
| vLLM PagedAttention | How does KV layout affect serving? |
| Speculative verification | How does one target pass verify K tokens? |

## Profiling Checklist

For every kernel case study, record the same small set of facts:

| Item | Questions |
|---|---|
| Workload shape | What are M/N/K, batch, sequence length, dtype, and layout? |
| Launch shape | How many blocks, threads, warps, and waves? |
| Memory traffic | Which tensors are read/written? Are accesses coalesced? |
| Reuse | What stays in registers, shared memory, L1, or L2? |
| Compute path | CUDA cores, Tensor Cores, or special instructions? |
| Bottleneck | Launch overhead, memory bandwidth, latency, occupancy, or compute? |
| Evidence | Which Nsight Compute metrics support the conclusion? |
| Next experiment | What single change improves or falsifies the hypothesis? |
