# Lab: Communication and Computation Overlap

## Goal

Show why Chapter 4 focuses on **exposed communication time**. The total communication volume may be the same, but the optimized path starts each bucket transfer as soon as that gradient bucket is ready.

This lab is a portable analogue of no-overlap and overlap-enabled DDP training paths.

## Baseline

The baseline runs all backward-layer compute first, then synchronizes every gradient bucket. Communication is fully exposed at the end of the step.

## Optimized

The optimized path submits a bucket transfer immediately after each layer finishes. Later backward compute overlaps with earlier bucket communication, so only the final communication tail remains exposed.

## Run

```bash
python compare.py
```

## Expected Observation

Both paths produce the same checksum. The optimized path should report lower median step time and a higher overlap ratio.

## What This Proves

Overlap does not make communication disappear. It moves communication under useful compute so the training step pays only the part that cannot be hidden.
