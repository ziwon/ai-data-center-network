# Lab: Communicator Lifecycle

## Goal

Show why communicators and registered buffers should be created once and reused.

## Baseline

The baseline recreates communication state for every step.

## Optimized

The optimized path pays setup once, then reuses the communicator across steps.

## Run

```bash
python compare.py
```

## Expected Observation

The optimized path should have the same output with much lower per-step overhead.
