# Lab: Pipeline and Tensor Parallel Scheduling

## Goal

Show how pipeline bubbles and tensor-parallel synchronization affect distributed model execution.

## Baseline

The baseline models fill-drain pipeline execution with a synchronous tensor-parallel all-gather after every stage.

## Optimized

The optimized path models 1F1B pipeline scheduling and overlaps part of tensor communication with useful stage compute.

## Run

```bash
python compare.py
```

## Expected Observation

The optimized path should reduce bubble time and exposed tensor-parallel communication.
