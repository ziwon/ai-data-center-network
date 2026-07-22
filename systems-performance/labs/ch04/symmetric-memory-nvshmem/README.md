# Lab: Symmetric Memory and NVSHMEM Patterns

## Goal

Show why persistent symmetric buffers matter for GPU-driven communication patterns.

## Baseline

The baseline allocates and registers a transfer buffer for each operation and uses a CPU rendezvous for every exchange.

## Optimized

The optimized path allocates a symmetric pool once and reuses stable offsets for repeated exchanges.

## Run

```bash
python compare.py
```

## Expected Observation

The optimized path should reduce setup count and per-operation latency while preserving the same payload checksum.
