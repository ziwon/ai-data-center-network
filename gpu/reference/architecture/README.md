# GPU Architecture (reference)

Look-up material for the GPU execution model and NVIDIA hardware architecture.
Use this to check *how the hardware behaves*; use [`../cuda`](../cuda/README.md)
for the CUDA programming API and [`../../courses/pmpp`](../../courses/pmpp/README.md)
to actually learn the model.

## Execution model

The concepts behind every CUDA kernel:

- Thread, block, grid
- Warp and SIMT execution
- SM, registers, shared memory, L1/L2, global memory (HBM)
- Memory coalescing
- Occupancy
- Warp divergence
- Basic scheduling intuition

Video primer: [Fundamentals of GPU Architecture](https://www.youtube.com/playlist?list=PLxNPSjHT5qvscDTMaIAY9boOOXAJAS7y4)
— a short refresher; the same ground is covered in depth by
[PMPP](../../courses/pmpp/README.md).

## NVIDIA architecture references

- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

## Lookup topics

- CUDA execution model mapped onto SM hardware
- Memory hierarchy and memory spaces
- Occupancy limiters (registers, shared memory, block size)
- Shared memory and bank conflicts
- Tensor Cores and precision formats
- Hopper and Blackwell architecture changes
