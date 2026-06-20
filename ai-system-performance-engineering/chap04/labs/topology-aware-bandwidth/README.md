# Lab: Topology-Aware Bandwidth

## Goal

Show why NCCL path quality depends on placement. A job can use GPUs successfully and still run over a slow PCIe or cross-socket path.

## Baseline

The baseline assigns ranks in a topology-agnostic order. Several collective edges cross slow links.

## Optimized

The optimized path groups ranks by fast local links first, then crosses the slower boundary with reduced traffic.

## Run

```bash
python compare.py
```

## Expected Observation

Both paths move the same logical gradient bytes. The topology-aware path should expose less communication time because more traffic stays on fast links.
