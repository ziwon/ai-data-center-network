# Lecture 1: Why Parallelism? Why Efficiency?

Source: [Stanford CS149 2023 Lecture 1](https://www.youtube.com/watch?v=V1tINV2-9p4)

Course materials:

* [CS149 Fall 2023 Lecture 1 course page](https://gfxcourses.stanford.edu/cs149/fall23/lecture/whyparallelism/)
* [Lecture 1 slides PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/whyparallelism/01_whyparallelism_huXfOJ4.pdf)

## Table of Contents

* [Goal](#goal)
* [Lecture Overview](#lecture-overview)
* [Speedup](#speedup)
* [Efficiency](#efficiency)
* [The Three Course Themes](#the-three-course-themes)
* [Why Parallelism Became Necessary](#why-parallelism-became-necessary)
* [Program as Instructions](#program-as-instructions)
* [Instruction-Level Parallelism](#instruction-level-parallelism)
* [Memory and Locality](#memory-and-locality)
* [GPU Systems Lens](#gpu-systems-lens)
* [Practical Tips and Notes](#practical-tips-and-notes)
* [Lecture Summary](#lecture-summary)
* [Key Terms](#key-terms)
* [Questions](#questions)
* [Answers](#answers)

---

## Goal

이번 강의의 목표는 왜 현대 시스템에서 parallelism이 선택 사항이 아니라 기본 조건이 되었는지, 그리고 단순히 "병렬로 돌렸다"는 사실만으로는 충분하지 않은 이유를 이해하는 것이다.

핵심 메시지는 다음과 같다.

> Parallel computing의 목표는 processor를 많이 쓰는 것 자체가 아니라, 문제를 더 빠르고 더 효율적으로 푸는 것이다. Speedup은 중요하지만, communication, synchronization, work imbalance, memory movement 때문에 hardware를 비효율적으로 쓰면 parallelism은 쉽게 낭비된다.

이 강의는 다음을 다룬다.

* Parallel computer의 기본 정의
* Speedup의 의미와 한계
* Communication cost와 work imbalance가 병렬 성능을 제한하는 방식
* Parallel programming에서 work decomposition, assignment, synchronization이 중요한 이유
* 왜 single-thread CPU 성능 향상이 예전처럼 공짜로 오지 않는지
* Processor가 program을 instruction stream으로 실행한다는 관점
* Instruction-level parallelism, clock frequency, memory hierarchy, cache locality의 기초
* GPU와 accelerator를 공부할 때 efficiency를 먼저 생각해야 하는 이유

---

## Lecture Overview

강의는 "parallel computer는 여러 processing element가 협력해 문제를 빠르게 푸는 컴퓨터"라는 정의에서 시작한다. 여기서 중요한 단어는 두 개다. 여러 processor를 사용한다는 점보다, 문제를 빠르게 풀고 hardware를 효율적으로 사용해야 한다는 점이 더 중요하다.

초반 데모는 학생들을 processor로 보고 숫자 합을 계산하게 한다. 한 명이 모든 숫자를 더하면 시간이 오래 걸린다. 두 명, 네 명, 더 많은 사람을 투입하면 겉보기에는 더 빨라질 수 있지만, 곧 communication과 coordination 비용이 드러난다. 부분합을 서로 알려야 하고, 누군가는 일이 먼저 끝나 기다리며, 너무 많은 사람이 참여하면 계산보다 전달 비용이 커진다. 이 데모는 이후 GPU kernel에서도 그대로 반복되는 문제를 보여준다.

중반부는 course theme을 잡는다. 좋은 parallel program은 work를 안전하게 나누고, processor에 잘 배정하고, communication과 synchronization이 speedup을 잡아먹지 않게 만든다. 동시에 hardware를 알아야 한다. 같은 algorithm이라도 cache, memory bandwidth, latency, processor 구조를 모르면 실제 machine에서 왜 느린지 설명할 수 없다.

후반부는 역사적 배경이다. 과거에는 single-thread CPU 성능이 빠르게 개선되어, 코드를 병렬화하지 않아도 다음 세대 CPU에서 프로그램이 빨라지는 경우가 많았다. 하지만 clock frequency scaling과 instruction-level parallelism만으로 성능을 계속 올리기 어려워지면서, 성능 향상의 중심은 multi-core, SIMD, GPU, domain-specific accelerator로 옮겨갔다. 따라서 이제 programmer도 parallelism과 machine efficiency를 직접 이해해야 한다.

마지막으로 processor 관점에서 program은 instruction의 list라는 관점을 소개한다. Instruction은 arithmetic을 수행하고, register와 memory에서 값을 읽고 쓰며, branch로 control flow를 바꾼다. 여기서 성능은 단순히 연산 개수만이 아니라 memory access latency, cache hit/miss, locality에 의해 크게 달라진다.

---

## Speedup

Parallel processing의 가장 직접적인 목표는 speedup이다.

```text
speedup(P processors) = execution time using 1 processor
                      / execution time using P processors
```

예를 들어 한 processor에서 40초 걸리는 일을 네 processor에서 10초에 끝내면 speedup은 4다. 이상적으로는 processor 수를 `P`배 늘리면 실행 시간이 `1/P`로 줄어든다. 하지만 강의 데모가 보여주는 것처럼 실제 speedup은 보통 이보다 작다.

| Limiting factor | What happens | GPU systems lens |
| --------------- | ------------ | ---------------- |
| Communication | Partial results must be exchanged | Global memory traffic, inter-SM communication, distributed training all-reduce |
| Synchronization | Workers wait for others | Barriers, kernel boundaries, stream dependencies |
| Work imbalance | Some workers idle early | Irregular kernels, sparse workloads, dynamic batching |
| Overhead | Coordination consumes useful time | Kernel launch overhead, scheduling overhead, framework overhead |
| Locality | Data is far from the processor | Cache miss, HBM traffic, PCIe/NVLink movement |

Speedup은 "얼마나 빨라졌는가"를 말하지만, "machine을 얼마나 잘 썼는가"까지 말해주지는 않는다. 10개 processor로 2배 speedup을 얻었다면 프로그램은 빨라졌지만, available hardware 대부분은 idle이거나 overhead에 쓰였을 수 있다.

## Efficiency

강의에서 반복되는 중요한 구분은 fast와 efficient다.

```text
efficiency(P processors) = speedup(P) / P
```

이 식으로 보면 10개 processor에서 2배 speedup은 efficiency 20%다. 반대로 4개 processor에서 3.6배 speedup은 efficiency 90%다. 더 많은 processor를 쓰는 것이 항상 더 좋은 선택은 아니다.

Efficiency는 두 관점에서 중요하다.

| Perspective | Question |
| ----------- | -------- |
| Programmer | 주어진 machine capability를 얼마나 잘 활용하고 있는가? |
| Hardware designer | 성능, silicon area, power, cost 사이에서 어떤 capability를 넣어야 하는가? |

GPU에서는 이 구분이 특히 중요하다. Kernel이 CPU보다 빠르더라도 GPU 전체 관점에서는 memory bandwidth만 조금 쓰고 SM 대부분이 놀고 있을 수 있다. 반대로 latency가 긴 operation이 많아도 충분한 thread-level parallelism과 좋은 locality가 있으면 높은 throughput을 낼 수 있다.

## The Three Course Themes

Lecture 1은 CS149 전체를 세 가지 theme으로 정리한다.

| Theme | Meaning | Later CS149 topics |
| ----- | ------- | ------------------ |
| Writing scalable parallel programs | Work를 나누고 배정하며 communication/synchronization을 관리한다 | Data parallelism, scheduling, task graphs, CUDA |
| Understanding parallel hardware | Abstraction이 hardware에서 어떻게 구현되는지 이해한다 | Multi-core CPU, SIMD, GPU, cache coherence |
| Thinking about efficiency | Faster가 efficient와 같지 않음을 측정하고 판단한다 | Locality, bandwidth, memory models, DNN execution |

이 repository의 GPU track에서는 세 번째 theme이 가장 실무적이다. CUDA syntax를 아는 것만으로는 충분하지 않다. Kernel이 왜 memory-bound인지, 왜 occupancy가 낮은지, 왜 data movement가 전체 latency를 지배하는지 설명할 수 있어야 한다.

## Why Parallelism Became Necessary

강의는 "예전에는 왜 parallel processing을 피할 수 있었는가?"라는 질문을 던진다. 과거에는 single-thread CPU performance가 빠르게 증가했다. Software developer가 코드를 병렬화하지 않아도 다음 세대 CPU에서 프로그램이 자연스럽게 빨라지는 경우가 많았다.

그 성능 향상의 큰 축은 두 가지였다.

1. Instruction-level parallelism을 exploiting하는 superscalar execution
2. CPU clock frequency 증가

하지만 이 경로는 한계에 부딪혔다. Clock을 계속 높이면 power와 heat 문제가 커지고, 단일 instruction stream에서 자동으로 뽑아낼 수 있는 independent work도 제한적이다. 그래서 현대 성능 향상은 여러 core, SIMD lanes, hardware threads, GPU SMs, tensor cores, accelerator 같은 명시적 parallel resources를 활용하는 방향으로 이동했다.

실무적으로 이 말은 다음과 같다.

* Algorithm 안에 parallel work가 있는지 찾아야 한다.
* Parallel work가 있더라도 communication과 synchronization을 줄여야 한다.
* Computation보다 data movement가 비싼 경우가 많다는 점을 먼저 의심해야 한다.
* Hardware가 제공하는 parallelism의 형태에 맞게 program structure를 바꿔야 한다.

## Program as Instructions

Processor 관점에서 program은 instruction의 sequence다. C/C++ source code는 compiler를 거쳐 machine instruction으로 바뀐다. Processor는 instruction을 fetch하고, decode하고, execute하고, register나 memory state를 갱신한다.

Instruction은 대략 다음 일을 한다.

| Instruction kind | Example role |
| ---------------- | ------------ |
| Arithmetic | Add, multiply, compare |
| Memory access | Load from address, store to address |
| Control flow | Branch, jump, call, return |

이 관점이 중요한 이유는 parallelism의 단위가 source code line이 아니라 실제 work와 dependency라는 점 때문이다. Loop가 있어도 iteration 사이에 dependency가 없으면 병렬화할 수 있다. 반대로 source code가 간단해 보여도 memory load가 길게 stall되면 processor는 연산기를 제대로 쓰지 못한다.

## Instruction-Level Parallelism

Instruction-level parallelism, 또는 ILP는 한 instruction stream 안에서 independent instruction을 동시에 또는 겹쳐서 실행하는 방식이다. Superscalar CPU는 여러 execution unit을 두고, dependency가 없는 instruction을 한 cycle에 여러 개 issue하려고 한다.

예를 들어 다음 두 연산은 서로 독립이면 동시에 실행될 수 있다.

```c
a = b + c;
x = y * z;
```

하지만 다음은 dependency가 있다.

```c
a = b + c;
x = a * z;
```

두 번째 instruction은 첫 번째 결과 `a`가 필요하므로 먼저 실행할 수 없다. CPU는 이런 dependency 안에서 가능한 ILP를 찾지만, 단일 thread에서 자동으로 얻을 수 있는 병렬성은 제한적이다. 이것이 multi-core, SIMD, GPU 같은 더 명시적인 parallel execution model이 필요한 이유다.

## Memory and Locality

Lecture 1의 후반부는 memory hierarchy의 직관을 잡는다. Programmer는 memory를 하나의 linear address space처럼 보지만, 실제 machine은 register, L1/L2/L3 cache, DRAM 등 여러 계층으로 data를 이동시킨다.

강의 슬라이드의 예시는 cache locality를 두 가지로 나눈다.

| Locality | Meaning | Example |
| -------- | ------- | ------- |
| Spatial locality | 가까운 address를 곧 사용할 가능성 | Contiguous array scan |
| Temporal locality | 같은 address를 다시 사용할 가능성 | Reusing a loaded value |

Cache는 locality가 있을 때 memory access latency를 줄인다. 하지만 locality가 없거나 working set이 cache보다 크면 cache miss가 늘고, processor는 DRAM access를 기다리며 stall된다.

강의 슬라이드의 Kaby Lake 예시에서는 data 위치에 따라 latency가 크게 달라진다.

| Data location | Approximate latency in cycles |
| ------------- | ----------------------------- |
| L1 cache | 4 |
| L2 cache | 12 |
| L3 cache | 38 |
| DRAM, best case | ~248 |

숫자 자체보다 중요한 것은 비율이다. Arithmetic unit이 아무리 많아도 data가 제때 오지 않으면 실행은 멈춘다. GPU에서도 같은 문제가 다른 형태로 나타난다. HBM bandwidth가 높아도 uncoalesced access, 낮은 arithmetic intensity, 반복적인 host-device transfer가 있으면 실제 throughput은 크게 낮아진다.

## GPU Systems Lens

Lecture 1은 아직 CUDA syntax를 다루지 않지만, GPU 성능을 이해하는 기준을 이미 제공한다.

| CS149 Lecture 1 concept | GPU/CUDA interpretation |
| ----------------------- | ----------------------- |
| Work decomposition | Thread, block, grid로 work를 나누는 방식 |
| Work assignment | Blocks to SMs, warps to schedulers |
| Communication cost | Global memory, shared memory, atomics, collectives |
| Synchronization | `__syncthreads()`, kernel launch boundaries, stream ordering |
| Work imbalance | Divergent branches, irregular loops, variable sequence lengths |
| Locality | Coalescing, shared-memory tiling, cache reuse |
| Efficiency | Occupancy, achieved bandwidth, SM utilization, Tensor Core utilization |

LLM inference와 training에서도 같은 질문을 던질 수 있다.

* GEMM은 충분한 arithmetic intensity를 갖는가?
* Attention은 memory movement와 synchronization 중 무엇이 병목인가?
* Batch size와 sequence length가 GPU utilization을 어떻게 바꾸는가?
* Kernel fusion은 communication과 memory traffic을 얼마나 줄이는가?
* 분산 training에서 all-reduce cost가 scaling efficiency를 제한하는가?

## Practical Tips and Notes

### Speedup만 보지 말고 사용한 자원을 같이 보라

성능 실험에서는 wall-clock time만 기록하지 말고 사용한 hardware scale도 같이 적어야 한다. 예를 들어 "2배 빨라졌다"는 말은 CPU core 2개를 썼는지, GPU 8개를 썼는지에 따라 의미가 완전히 다르다.

| Record together | Why it matters |
| --------------- | -------------- |
| Runtime | User-visible performance |
| Processor/GPU count | Resource cost |
| Utilization | Whether hardware was actually busy |
| Memory bandwidth | Whether data movement is the bottleneck |
| Synchronization time | Whether waiting dominates useful work |

### Communication은 계산보다 늦게 최적화할 문제가 아니다

강의 데모에서 communication은 처음부터 speedup을 제한한다. GPU에서도 마찬가지다. Kernel을 작성할 때 "각 thread가 무엇을 계산하는가"와 함께 "각 thread가 어떤 data를 어디서 읽고 어디에 쓰는가"를 동시에 봐야 한다.

### Locality는 algorithm property다

Cache hit은 hardware가 자동으로 해주는 최적화처럼 보이지만, locality는 program이 만들어낸다. Contiguous access, data reuse, tiling, fusion은 모두 locality를 높이기 위한 program-level decision이다.

### Fast path와 efficient path는 다를 수 있다

작은 input에서는 CPU가 GPU보다 빠를 수 있다. GPU가 느려서가 아니라 launch overhead, transfer overhead, 낮은 occupancy가 useful work보다 클 수 있기 때문이다. 반대로 큰 input에서는 같은 algorithm이라도 GPU가 압도적으로 빠를 수 있다. 항상 problem size와 overhead를 같이 보아야 한다.

## Lecture Summary

Lecture 1의 결론은 다음과 같다.

* Single-thread performance는 예전처럼 빠르게 좋아지지 않는다.
* 큰 성능 향상을 얻으려면 multiple processing elements 또는 specialized hardware를 사용해야 한다.
* Parallel program은 work partitioning, communication, synchronization 때문에 어렵다.
* Hardware characteristics를 모르면 병목을 잘못 짚기 쉽다.
* 특히 data movement와 locality가 modern parallel computing의 핵심이다.
* GPU programming은 CUDA API 암기가 아니라 efficiency reasoning에서 시작해야 한다.

## Key Terms

| Term | Meaning |
| ---- | ------- |
| Parallel computer | 여러 processing element가 협력해 문제를 빠르게 푸는 computer |
| Speedup | 1 processor 실행 시간 대비 P processor 실행 시간의 비율 |
| Efficiency | Speedup을 사용한 processor 수로 나눈 값 |
| Communication | Processor나 worker 사이에 data/result를 전달하는 비용 |
| Synchronization | 여러 worker의 진행 순서를 맞추기 위해 기다리는 비용 |
| Work imbalance | 일부 worker는 idle이고 일부 worker만 계속 일하는 상태 |
| Instruction-level parallelism | 한 instruction stream 안의 independent instruction을 겹쳐 실행하는 병렬성 |
| Superscalar execution | 한 cycle에 여러 instruction을 issue하려는 CPU 실행 방식 |
| Memory hierarchy | Register, cache, DRAM 등 여러 latency/capacity 계층 |
| Spatial locality | 가까운 memory address를 연속적으로 접근하는 경향 |
| Temporal locality | 같은 data를 반복적으로 접근하는 경향 |

## Questions

1. Speedup이 2배라면 항상 좋은 parallelization이라고 말할 수 있는가?
2. 10개 processor로 2배 speedup을 얻은 경우 efficiency는 얼마인가?
3. 강의 데모에서 processor 수를 늘려도 speedup이 제한된 이유는 무엇인가?
4. 왜 과거에는 많은 software developer가 parallel programming을 미룰 수 있었는가?
5. Single-thread performance scaling이 둔화되면 programmer에게 어떤 책임이 생기는가?
6. Processor 관점에서 program을 instruction list로 보는 이유는 무엇인가?
7. ILP와 multi-core parallelism은 어떤 점에서 다른가?
8. Spatial locality와 temporal locality의 차이는 무엇인가?
9. Cache가 memory latency를 줄이는 조건은 무엇인가?
10. GPU kernel이 CPU보다 빠르더라도 inefficient할 수 있는 이유는 무엇인가?

## Answers

1. 아니다. 사용한 processor 수와 efficiency를 같이 봐야 한다.
2. `2 / 10 = 0.2`, 즉 20%다.
3. 부분 결과 전달, synchronization, work imbalance, coordination overhead 때문이다.
4. CPU clock frequency와 single-thread performance가 빠르게 개선되어 코드를 바꾸지 않아도 프로그램이 빨라지는 경우가 많았기 때문이다.
5. Work를 병렬화하고, data movement를 줄이며, hardware 특성에 맞게 프로그램을 구성해야 한다.
6. 실제 dependency, memory access, branch, arithmetic operation이 성능과 병렬화 가능성을 결정하기 때문이다.
7. ILP는 한 thread의 instruction stream 내부에서 hardware가 찾는 병렬성이고, multi-core parallelism은 여러 processing element에 명시적으로 work를 나누는 병렬성이다.
8. Spatial locality는 가까운 address를 함께 쓰는 성질이고, temporal locality는 같은 data를 다시 쓰는 성질이다.
9. Access pattern에 locality가 있고 working set이 cache hierarchy에서 효과적으로 재사용될 때다.
10. Kernel launch, data transfer, memory bandwidth, 낮은 occupancy, synchronization overhead 때문에 hardware 대부분을 쓰지 못할 수 있기 때문이다.
