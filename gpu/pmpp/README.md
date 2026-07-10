# PMPP / Programming Massively Parallel Processors

The **primary spine** of this GPU track: the CUDA and parallel-programming
foundation based on *Programming Massively Parallel Processors*. Everything else
supports this course — pair it with
[Heterogeneous Systems](../heterogeneous-systems/README.md) for a second pass,
look up details in [`appendix/`](../appendix/), and practice each concept in
[`labs/`](../labs/).

## Primary Resource

- [PMPP 2021 lecture playlist](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)

## Lecture Notes

| Lecture | Topic | Notes |
| ------- | ----- | ----- |
| 2 | Data parallel programming and the CUDA programming model | [lec02](lec02/README.md) |
| 3 | Multidimensional grids and data | [lec03](lec03/README.md) |
| 4 | GPU architecture | [lec04](lec04/README.md) |

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

Add new lecture notes to the index above as the course progresses. Prefer
measurement-oriented notes that connect CUDA programming patterns to GPU
performance behavior.
