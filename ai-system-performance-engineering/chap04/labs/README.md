# Chapter 4 Labs

This directory contains executable labs that support Chapter 4 notes on distributed networking communication.

The labs adapt the chapter's distributed communication ideas into small, standalone examples:

| Lab | Claim |
| --- | --- |
| `communication-overlap/` | Exposed communication, not total communication, controls step time. |
| `gradient-bucketing/` | Many small gradient transfers waste latency; fused buffers and reduced precision lower communication cost. |
| `nixl-tier-handoff/` | Disaggregated inference should move selected KV blocks through packed point-to-point transfers, not CPU-staged block loops. |
| `topology-aware-bandwidth/` | Rank placement and link topology determine exposed communication cost. |
| `dataparallel-vs-ddp/` | DataParallel creates host orchestration and primary-device fan-in overhead. |
| `communicator-lifecycle/` | Reusing communicators avoids repeated setup on the training path. |
| `pipeline-tensor-parallel/` | 1F1B scheduling and communication overlap reduce pipeline bubbles. |
| `symmetric-memory-nvshmem/` | Persistent symmetric buffers avoid repeated registration and rendezvous overhead. |
| `gpu-communication-reference/` | Real CUDA/NCCL reference scripts for DDP overlap and all-reduce bucket sweeps. |

Run a lab from its directory:

```bash
python compare.py
```

These labs are intentionally CPU-portable. They model the scheduling and data-movement shape of NCCL/NIXL examples without requiring a multi-GPU `torchrun` environment.

`gpu-communication-reference/` is the exception. It is included so the same chapter has code that can be run unchanged on a real multi-GPU host.
