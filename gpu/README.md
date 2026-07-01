# GPU Systems

A learning track for GPU architecture and CUDA kernel programming, organized by
**role** rather than by topic. The goal is to learn how GPUs actually run
kernels and why real inference workloads are fast or slow.

## How this track is organized

Each subfolder has one job. If you are unsure where something belongs, ask which
role it plays: are you *learning* it, *looking it up*, or *practicing* it?

| Tier | Role | Use it for |
|---|---|---|
| [`courses/`](courses/) | **Learn** | Structured lecture series, followed in order |
| [`reference/`](reference/) | **Look up** | Official docs and architecture facts, consulted on demand |
| [`practice/`](practice/) | **Do** | Hands-on kernel work and profiling |

This split also resolves the CUDA overlap: CUDA-the-course lives in
[`courses/pmpp`](courses/pmpp/README.md), CUDA-the-reference lives in
[`reference/cuda`](reference/cuda/README.md), and CUDA-the-practice lives in
[`practice/kernels`](practice/kernels/README.md).

## Suggested path

1. **Spine:** work through [PMPP](courses/pmpp/README.md) as the primary course.
2. **Complement:** use [Heterogeneous Systems (Onur Mutlu)](courses/heterogeneous-systems/README.md)
   in parallel for a second pass over the classic parallel patterns.
3. **Practice as you go:** for each concept, profile a real kernel in
   [`practice/`](practice/) and confirm the behavior with
   [Nsight Compute](practice/profiling/README.md).
4. **Reference on demand:** consult [`reference/`](reference/) for the exact
   NVIDIA architecture and CUDA API details.
5. **Later / advanced:** track [GPU MODE](courses/gpu-mode/README.md) as an
   ongoing feed for modern ML kernels.

## Index

### courses/ — learn
- [PMPP / Programming Massively Parallel Processors](courses/pmpp/README.md) — primary spine
- [Heterogeneous Systems (Onur Mutlu)](courses/heterogeneous-systems/README.md) — parallel-pattern complement
- [GPU MODE](courses/gpu-mode/README.md) — advanced ML-kernel feed

### reference/ — look up
- [GPU Architecture](reference/architecture/README.md) — execution model + NVIDIA architecture
- [CUDA](reference/cuda/README.md) — NVIDIA CUDA programming and best-practices docs

### practice/ — do
- [Profiling](practice/profiling/README.md) — Nsight Compute
- [Kernel Case Studies](practice/kernels/README.md) — hands-on kernel analysis
