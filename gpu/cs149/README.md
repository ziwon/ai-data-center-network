# Stanford CS149 / Parallel Computing

Stanford CS149 is the systems foundation course for this GPU track. It connects
GPU programming with the broader parallel-computing model: multi-core CPUs,
SIMD, scheduling, locality, synchronization, memory models, distributed
data-parallel systems, DSLs, and hardware specialization.

Use this course to understand why GPU kernels, DNN execution, and accelerator
systems behave the way they do beyond CUDA syntax alone.

## Primary Resources

- [CS149 Fall 2023 course page](https://gfxcourses.stanford.edu/cs149/fall23)
- [CS149 Fall 2023 YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)

## Study Priority

Focus first on:

1. Parallelism, efficiency, and hardware utilization
2. Multi-core, SIMD, and multi-threading execution models
3. Work distribution, scheduling, locality, and contention
4. GPU architecture and CUDA programming abstractions
5. Data-parallel operations such as map, reduce, scan, and groupByKey
6. DNN execution on GPUs, including convolution, transformers, and fusion
7. Cache coherence, memory consistency, and synchronization
8. Domain-specific languages and hardware specialization

## GPU Systems Lens

For this repository, read CS149 with these questions in mind:

- How do CPU parallelism, SIMD, and GPU parallelism differ in their bottlenecks?
- When does scheduling overhead dominate useful work?
- How do locality and communication shape kernel and system performance?
- Why do reductions, scans, and data movement show up repeatedly in ML systems?
- Which memory model or synchronization assumption can make a parallel program incorrect?
- How do DNN execution patterns map onto general parallel-computing principles?

## Lecture Notes

| Lecture | Topic | Notes |
| ------- | ----- | ----- |
| 1 | Why Parallelism? Why Efficiency? | [lec01](lec01/README.md) |
| 2 | A Modern Multi-Core Processor | [lec02](lec02/README.md) |
| 3 | Multi-Core Architecture, Part II and ISPC | [lec03](lec03/README.md) |
| 4 | Parallel Programming Basics | [lec04](lec04/README.md) |
| 5 | Performance Optimization I: Work Distribution and Scheduling | [lec05](lec05/README.md) |
| 6 | Performance Optimization II: Locality, Communication, and Contention | [lec06](lec06/README.md) |
| 7 | GPU Architecture and CUDA Programming | [lec07](lec07/README.md) |
| 8 | Data-Parallel Thinking | [lec08](lec08/README.md) |
| 9 | Distributed Data-Parallel Computing Using Spark | [lec09](lec09/README.md) |
| 10 | Efficiently Evaluating DNNs on GPUs | [lec10](lec10/README.md) |
