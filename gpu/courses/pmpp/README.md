# PMPP / Programming Massively Parallel Processors

The **primary spine** of this GPU track: the CUDA and parallel-programming
foundation based on *Programming Massively Parallel Processors*. Everything else
supports this course — pair it with
[Heterogeneous Systems](../heterogeneous-systems/README.md) for a second pass,
look up details in [`reference/`](../../reference/), and practice each concept in
[`practice/`](../../practice/).

## Primary Resource

- [PMPP 2021 lecture playlist](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)

## Study Priority

Focus first on:

1. GPU programming model: thread, block, grid
2. CUDA memory hierarchy
3. Tiled matrix multiplication
4. Convolution and stencil memory patterns
5. Parallel reduction and scan
6. Occupancy, memory coalescing, and divergence
7. Arithmetic intensity and roofline thinking

## LLM Inference Lens

For this repository, read PMPP with these questions in mind:

- Why do GEMM and GEMV behave differently on a GPU?
- When is a kernel memory-bound instead of compute-bound?
- How does tiling reduce global-memory traffic?
- Why does batch size change Tensor Core utilization?
- Which Nsight Compute metrics explain the observed bottleneck?

## Notes

Add lecture notes here as the course progresses. Prefer short, measurement-oriented notes over long summaries.
