# GPU MODE

Advanced, ongoing feed for modern ML kernels (formerly CUDA MODE). Treat this as
a **later / advanced** course: come here after the [PMPP](../pmpp/README.md)
foundation is in place. Specific kernel lectures pulled out for hands-on work
live in [`labs/kernels`](../labs/kernels/README.md).

## Resource Index

### YouTube / Lecture Series

- [GPU MODE YouTube channel](https://www.youtube.com/@GPUMODE/videos)
- [GPU MODE lecture playlist](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)

### Websites / Repositories

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures): slides, notebooks, and lecture references.

### Articles / Notes

- Third-party lecture notes or summaries tied to specific GPU MODE talks.

### Podcasts / Ongoing Feeds

- GPU MODE YouTube channel as an ongoing technical feed.

## Must Watch First

- Compute and memory architecture lectures
- CUDA for Python programmers
- Profiling and performance-analysis lectures
- Triton introduction and optimization lectures

## Watch Later

- PyTorch optimizer internals
- Quantization kernel talks
- FlashAttention or attention-kernel talks
- Distributed and multi-GPU performance talks

## Routing

Where each lecture theme connects in this track:

| Lecture theme | Related area |
|---|---|
| CUDA programming basics | [PMPP](../pmpp/README.md) |
| GPU architecture concepts | [Architecture reference](../appendix/architecture/README.md) |
| CUDA API / best-practices details | [CUDA reference](../appendix/cuda/README.md) |
| Nsight Compute workflow or metrics | [Profiling](../labs/profiling/README.md) |
| Triton, CUTLASS, FlashAttention, low-bit kernels | [Kernel case studies](../labs/kernels/README.md) |
| Reduction, scan, histogram, convolution patterns | [Heterogeneous Systems](../heterogeneous-systems/README.md) |
| NCCL, collectives, multi-GPU behavior | Distributed GPU systems (not yet tracked) |
