# Lab: NIXL-Style Tier Handoff

## Goal

Show why Chapter 4 separates training collectives from disaggregated inference data movement. NCCL is the right lens for all-reduce, but KV cache handoff is a point-to-point transfer problem.

This lab adapts the important idea behind baseline and optimized NIXL-style tier handoff examples.

## Baseline

The baseline copies selected KV blocks one at a time through a CPU staging buffer. Each block pays fixed scheduling overhead and creates fragmented movement.

## Optimized

The optimized path packs selected blocks into one contiguous transfer and unpacks at the target tier. This models a NIXL/UCX-style handoff where the movement layer sees a compact payload.

## Run

```bash
python compare.py
```

## Expected Observation

The optimized path should transfer the same selected KV bytes with fewer operations and lower elapsed time.

## What This Proves

Disaggregated serving performance depends on the shape of KV movement: block selection, packing, registration reuse, and prefill/decode placement matter as much as raw link bandwidth.
