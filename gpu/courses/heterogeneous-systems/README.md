# Heterogeneous Systems (Onur Mutlu)

A complementary course to [PMPP](../pmpp/README.md), used in parallel for a
second pass over the classic parallel patterns behind GPU kernels. Where PMPP is
the spine, this course reinforces the same patterns from a computer-architecture
angle and adds irregular workloads (sparse, graph).

## Primary Resource

- [Hands-on Acceleration on Heterogeneous Computing Systems, Fall 2021](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi_OwkTgEyA6tk3UsoPBH737)

## Parallel patterns

Classical building blocks behind GPU kernels. Each pattern maps directly to a
case study in [`practice/kernels`](../../practice/kernels/README.md).

| Meeting | Topic | Kernel pattern |
|---|---|---|
| 2 | SIMD processors and GPU architecture | execution model |
| 4 | GPU memory hierarchy | memory traffic and locality |
| 5 | GPU performance considerations | bottleneck diagnosis |
| 6 | Reduction | tree reductions, synchronization |
| 7 | Histogram | atomics, contention, privatization |
| 8 | Convolution | stencil/convolution memory reuse |
| 9 | Prefix sum / scan | parallel dependencies |
| 10 | Sparse matrices | irregular memory access |
| 11 | Graph search | divergence and irregular workloads |

## How to use with PMPP

- Watch a PMPP lecture on a pattern, then the matching meeting here for a second
  explanation.
- After both, implement and profile the pattern in
  [`practice/kernels`](../../practice/kernels/README.md).

## Notes

Add lecture notes here as the course progresses. Prefer short,
measurement-oriented notes over long summaries.
