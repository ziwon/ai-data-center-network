# Lab: DataParallel vs DDP

## Goal

Show why Chapter 4 treats PyTorch `DataParallel` as an anti-pattern for serious multi-GPU training.

## Baseline

The baseline models one Python process that scatters input, runs replicas, gathers output, and reduces gradients through a primary device.

## Optimized

The optimized path models one process per GPU with local forward/backward work and one collective gradient synchronization.

## Run

```bash
python compare.py
```

## Expected Observation

The DDP-shaped path should spend less time in host orchestration and primary-device fan-in.
