# Lab: GPU Communication Reference

## Goal

Provide real GPU/NCCL entrypoints for the Chapter 4 communication labs.

The other labs in this directory are CPU-portable concept models. This lab is intentionally not portable: it should be run only on a host with CUDA, NCCL, PyTorch distributed support, and at least two visible NVIDIA GPUs.

This code adapts the chapter's important GPU-facing communication ideas:

- `ddp_no_overlap.py`
- `ddp_overlap.py`
- gradient bucket and NCCL all-reduce benchmark patterns

## Scripts

| Script | Purpose |
| --- | --- |
| `no_overlap.py` | Runs forward/backward, then manually all-reduces each parameter gradient after backward. |
| `ddp_overlap.py` | Uses PyTorch `DistributedDataParallel` so gradient buckets can be reduced during backward. |
| `allreduce_bucket_sweep.py` | Measures NCCL all-reduce latency and bandwidth across bucket sizes and dtypes. |
| `topology_sweep.py` | Measures peer-copy bandwidth for visible GPU pairs. |
| `dataparallel_vs_ddp.py` | Runs `DataParallel` in one process or DDP under `torchrun`. |
| `communicator_reuse.py` | Compares process-group recreation with process-group reuse. |
| `pipeline_1f1b.py` | CUDA stream sketch for fill-drain vs overlapped pipeline work. |
| `symmetric_memory_probe.py` | Checks whether the local PyTorch build exposes symmetric-memory support. |

## Run

```bash
torchrun --nproc_per_node=2 no_overlap.py
torchrun --nproc_per_node=2 ddp_overlap.py
torchrun --nproc_per_node=2 allreduce_bucket_sweep.py
python topology_sweep.py
python dataparallel_vs_ddp.py
torchrun --nproc_per_node=2 dataparallel_vs_ddp.py
torchrun --nproc_per_node=2 communicator_reuse.py
python pipeline_1f1b.py
python symmetric_memory_probe.py
```

Optional knobs:

```bash
torchrun --nproc_per_node=4 ddp_overlap.py --iterations 50 --warmup 10 --bucket-cap-mb 25
torchrun --nproc_per_node=4 allreduce_bucket_sweep.py --min-kb 16 --max-mb 512 --dtype fp16
```

## Expected Observation

- `no_overlap.py` exposes all gradient communication after backward.
- `ddp_overlap.py` should reduce step time when communication can overlap with backward compute.
- `allreduce_bucket_sweep.py` should show poor efficiency for very small buckets and better bandwidth as message size grows.

## Validation Notes

Use these with Chapter 4 operational checks:

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 ddp_overlap.py
nvidia-smi topo -m
```

For real profiling, capture an Nsight Systems trace and check whether NCCL kernels overlap with backward kernels. A successful run alone does not prove the fabric path is healthy.
