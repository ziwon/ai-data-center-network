# Profiling (practice)

Nsight Compute and related GPU performance-analysis references. This is the
hands-on tool you run *while* learning: for every concept in
[PMPP](../../courses/pmpp/README.md) and every kernel in
[`../kernels`](../kernels/README.md), confirm the behavior with real metrics
here.

## Primary Resource

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## Metric Questions

- Is the kernel launch-overhead-bound, memory-bound, or compute-bound?
- What are SM Busy, SM throughput, DRAM throughput, and L2 hit rate saying?
- Is memory access coalesced?
- Is occupancy limited by registers, shared memory, or block size?
- Does the workload hit Tensor Cores or CUDA cores?
