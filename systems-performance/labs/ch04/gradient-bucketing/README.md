# Lab: Gradient Bucketing, Fusion, and Compression

## Goal

Show why Chapter 4 treats gradient synchronization as both a bandwidth problem and a launch-latency problem.

This lab adapts the important ideas behind gradient fusion and gradient compression benchmarks.

## Baseline

The baseline sends many small FP32 gradient buckets. This preserves precision, but every bucket pays a fixed communication launch cost.

## Optimized

The optimized path fuses buckets into one contiguous buffer and transfers it as FP16. This reduces both launch count and communication bytes. The reduced-precision checksum is compared with a tolerance instead of exact equality.

## Run

```bash
python compare.py
```

## Expected Observation

The optimized path should have fewer launches, fewer transferred bytes, lower modelled communication time, and a close output checksum.

## What This Proves

Gradient fusion and compression are useful when transfer latency or network bandwidth is exposed. They are not free: the precision error must be measured against the training tolerance.
