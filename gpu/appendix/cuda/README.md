# CUDA (reference)

Official NVIDIA CUDA programming documentation. This is a **lookup** resource
for API and language details — not a course. Learn CUDA through
[PMPP](../../pmpp/README.md); write and analyze kernels in
[`labs/kernels`](../../labs/kernels/README.md); check hardware behavior
in [`../architecture`](../architecture/README.md).

## Core references

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  — language, execution model, memory model, API.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  — performance guidelines (coalescing, occupancy, memory transfers).

## Lookup topics

- Kernel launch configuration and indexing
- Memory spaces: global, shared, constant, local, registers
- Synchronization and atomics
- Streams, events, and asynchronous copies
- Compilation, `nvcc`, and PTX/SASS basics
