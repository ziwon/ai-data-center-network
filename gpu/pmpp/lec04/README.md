# Lecture 4: GPU Architecture

Source: [PMPP 2021 Lecture 4](https://www.youtube.com/watch?v=pBQJAwogMoE&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=4)

## Table of Contents

* [Goal](#goal)
* [Lecture Overview](#lecture-overview)
* [GPU as a Collection of SMs](#gpu-as-a-collection-of-sms)
* [Thread Blocks Are Assigned to SMs](#thread-blocks-are-assigned-to-sms)
* [Why a Block Must Fit on One SM](#why-a-block-must-fit-on-one-sm)
* [Block-Level Collaboration](#block-level-collaboration)
* [Transparent Scalability](#transparent-scalability)
* [Warps as the Scheduling Unit](#warps-as-the-scheduling-unit)
* [SIMD Execution](#simd-execution)
* [Control Divergence](#control-divergence)
* [Latency Hiding](#latency-hiding)
* [Occupancy](#occupancy)
* [Choosing a Block Size](#choosing-a-block-size)
* [Querying Device Properties](#querying-device-properties)
* [Practical Tips and Notes](#practical-tips-and-notes)
* [Lecture Summary](#lecture-summary)
* [Key Terms](#key-terms)
* [Questions](#questions)
* [Answers](#answers)

---

## Goal

이번 강의의 목표는 지금까지 작성한 CUDA kernel이 실제 GPU hardware 위에서 어떻게 실행되는지 이해하는 것이다. 2강과 3강에서는 thread, block, grid를 프로그래밍 모델로 사용했다. 4강은 이 추상화가 GPU의 streaming multiprocessor, warp scheduler, SIMD execution, register file, shared memory, global memory 같은 hardware 구조에 어떻게 mapping되는지 설명한다.

핵심 메시지는 다음과 같다.

> CUDA grid는 block 단위로 SM에 배치되고, block 안의 thread는 warp 단위로 scheduling된다. GPU는 개별 operation의 latency를 낮추는 processor가 아니라, 많은 warp를 resident 상태로 두고 long-latency operation 동안 다른 warp를 실행해 throughput을 높이는 processor다.

이 강의는 다음을 다룬다.

* GPU가 여러 streaming multiprocessor로 구성되는 방식
* V100 예시: 80 SMs, SM당 64 FP32 cores
* Thread block이 SM에 배치되는 규칙
* Block 내부 thread가 같은 SM에 있어야 하는 이유
* `__syncthreads()`와 shared memory가 block-local collaboration인 이유
* Block 간 synchronization을 피해야 하는 이유와 deadlock 위험
* Transparent scalability가 가능한 CUDA block 독립성
* Warp가 SM scheduling의 기본 단위인 이유
* SIMD model의 장점과 control divergence의 비용
* GPU가 long latency를 warp-level multithreading으로 숨기는 방식
* Occupancy의 의미와 block size 선택의 hardware 제약
* `cudaGetDeviceProperties`로 device limit을 확인하는 방법

---

## Lecture Overview

강의는 3강 복습으로 시작한다. 3강에서는 2D grid와 2D block을 사용해 image와 matrix를 처리했다. RGB to grayscale은 output pixel 하나를 thread 하나가 맡는 단순한 2D data parallelism이었다. Image blur는 output pixel 하나가 여러 input pixel을 읽기 때문에 output boundary와 input boundary를 분리해야 했다. Matrix multiplication은 output matrix의 element 하나를 thread 하나가 맡고, dot product loop는 thread 내부에서 sequential하게 수행했다.

4강의 본론은 "이 thread들이 GPU에서 실제로 어디에 올라가는가?"라는 질문이다. CUDA code에서는 grid, block, thread라는 논리적 hierarchy를 보지만, hardware에서는 SM, core, register, warp scheduler, instruction dispatch unit, memory hierarchy가 이 hierarchy를 실행한다. GPU 전체에는 여러 SM이 있고, 각 SM 안에는 여러 execution core와 control logic, register file, shared memory가 있다. 모든 SM은 global memory에 접근할 수 있다.

강의는 먼저 block scheduling을 설명한다. Thread block은 쪼개져 여러 SM에 흩어지지 않는다. Block 안의 모든 thread는 같은 SM에 배치되고, 하나의 SM은 여러 block을 동시에 resident 상태로 둘 수 있다. Grid에 block이 너무 많으면 일부 block은 아직 실행되지 않은 채 기다린다. 이 규칙 때문에 block 안의 thread는 synchronization과 shared memory를 통해 협력할 수 있지만, block 간 synchronization은 일반 CUDA kernel 안에서 안전하지 않다.

그 다음에는 SM 내부로 zoom-in한다. SM에 올라온 thread는 warp라는 scheduling unit으로 묶인다. Warp size는 device-specific이지만 현재까지 CUDA GPU에서는 보통 32 threads다. 같은 warp의 thread는 SIMD 방식으로 같은 instruction을 실행한다. 이 방식은 instruction fetch/decode/control logic의 비용을 여러 core에 나눠 쓰게 해 GPU가 많은 arithmetic unit을 넣을 수 있게 한다. 대신 warp 안의 thread가 서로 다른 branch를 타면 control divergence가 생기고, 일부 lane은 inactive 상태로 낭비된다.

마지막으로 GPU가 latency를 다루는 방식을 설명한다. CPU는 out-of-order execution, 큰 cache, wide issue 등으로 single-thread latency를 줄이는 방향에 가깝다. GPU는 반대로 많은 thread와 warp를 resident 상태로 두고, 한 warp가 memory access나 multi-cycle arithmetic 때문에 stall되면 다른 ready warp를 실행한다. 이 능력을 높이는 중요한 지표가 occupancy다.

---

## GPU as a Collection of SMs

GPU는 여러 streaming multiprocessor로 구성된다. 보통 줄여서 SM이라고 부른다. 각 SM은 여러 core, control logic, register file, shared memory, cache 같은 hardware resource를 갖고 있다. 모든 SM은 GPU global memory에 접근할 수 있다.

| Level | Role |
| ----- | ---- |
| GPU | 여러 SM과 global memory를 포함하는 device |
| SM | Thread block을 resident 상태로 두고 warp를 scheduling하는 execution unit |
| Core | Arithmetic operation을 실제로 수행하는 execution unit |
| Global memory | 모든 SM이 접근할 수 있는 device memory |
| Shared memory | 같은 thread block의 thread들이 협력할 때 쓰는 빠른 on-chip memory |

강의의 기준 hardware는 Volta V100이다. V100은 강의 실습 서버에서 사용하는 GPU로 소개된다.

| V100 resource | Value in lecture |
| ------------- | ---------------- |
| SM count | 80 |
| FP32 cores per SM | 64 |
| Total FP32 cores | `80 * 64 = 5120` |
| Max threads per SM | 2048 |
| Max blocks per SM | 32 |
| Max threads per block | 1024 |

이 표에서 중요한 점은 core 수와 thread 수가 같지 않다는 것이다. V100의 SM 하나에는 64 FP32 cores가 있지만, 동시에 resident 상태로 둘 수 있는 thread는 2048개다. 즉, GPU는 "한 core에 thread 하나"로 단순히 생각하면 안 된다. SM은 실제로 한 순간에 실행하는 thread보다 훨씬 많은 thread context를 보유하고, warp scheduler가 그중 ready warp를 골라 실행한다.

---

## Thread Blocks Are Assigned to SMs

CUDA grid는 block의 집합이고, block은 thread의 집합이다. Hardware에서 scheduling될 때 기본 규칙은 다음과 같다.

```text
one thread block -> one SM
all threads in the same block -> the same SM
one SM -> multiple resident thread blocks possible
```

Block 내부 thread 일부를 한 SM에 놓고 나머지를 다른 SM에 놓는 방식은 허용되지 않는다. 반대로 하나의 SM이 여러 block을 동시에 resident 상태로 둘 수는 있다. 예를 들어 어떤 SM에 세 개의 thread block이 올라가 있을 수 있다.

Grid가 GPU 전체에서 동시에 처리할 수 있는 block 수보다 크면 어떻게 되는가? 나머지 block은 논리적으로 대기한다. 먼저 resident 상태로 올라간 block들이 끝나면, 그 자리를 waiting block이 차지한다.

```text
grid
  block 0  -> SM 0
  block 1  -> SM 1
  block 2  -> SM 0
  ...
  block k  -> waiting until resources are available
```

여기서 "assigned to an SM"은 반드시 지금 core에서 instruction을 실행 중이라는 뜻이 아니다. 더 정확히는 해당 block의 thread들이 SM의 execution resource를 예약했다는 뜻이다. 각 thread는 register, thread slot, control metadata 같은 resource를 필요로 한다.

---

## Why a Block Must Fit on One SM

Block은 한 번에 통째로 SM에 배치되어야 한다. 예를 들어 block 하나가 1024 threads이고 어떤 SM에 512 thread slots만 비어 있다면, CUDA runtime이나 hardware scheduler가 block 절반만 먼저 올리고 나머지 절반을 나중에 올리는 식으로 처리하지 않는다.

이 규칙은 synchronization 때문에 필요하다. Block 안의 thread는 `__syncthreads()` 같은 barrier synchronization을 사용할 수 있다. 만약 1024-thread block 중 512 threads만 먼저 실행되고 나머지 512 threads는 아직 scheduling되지 않았다면 다음 deadlock이 가능하다.

1. 먼저 올라간 512 threads가 barrier에 도착한다.
2. 이 512 threads는 나머지 512 threads가 barrier에 도착하기를 기다린다.
3. 아직 올라가지 못한 512 threads는 먼저 올라간 512 threads가 끝나 resource를 반납하기를 기다린다.
4. 양쪽이 서로를 기다리므로 진행이 멈춘다.

따라서 block을 SM에 배치하려면 그 block의 모든 thread가 필요한 resource를 확보할 수 있어야 한다.

> [!WARNING]
> Thread block은 scheduling과 resource allocation의 atomic한 단위로 생각해야 한다. Block 일부만 먼저 실행하고 나머지를 나중에 실행하는 모델을 가정하면 barrier semantics를 잘못 이해하게 된다.

---

## Block-Level Collaboration

CUDA가 grid를 block으로 나누는 중요한 이유는 block 내부 thread가 서로 협력할 수 있기 때문이다. 같은 block의 thread는 같은 SM에 배치되므로, hardware는 이들 사이의 collaboration을 비교적 효율적으로 지원할 수 있다.

대표적인 collaboration mechanism은 두 가지다.

| Mechanism | Scope | Purpose |
| --------- | ----- | ------- |
| `__syncthreads()` | Threads in the same block | Block 안의 모든 thread가 특정 지점에 도착할 때까지 기다림 |
| Shared memory | Threads in the same block | Block-local data exchange와 reuse |

`__syncthreads()`는 barrier다. Block 안의 모든 thread가 barrier에 도착해야 그 이후 instruction으로 진행할 수 있다. 이 기능은 이후 shared memory tiling, reduction, stencil computation 같은 pattern에서 핵심이 된다.

Shared memory도 block-local이다. 같은 block에 속한 thread는 shared memory를 통해 intermediate data를 교환할 수 있지만, 다른 block의 thread는 그 shared memory에 접근할 수 없다. 이 제약은 단점처럼 보일 수 있지만, CUDA의 scalability를 가능하게 하는 핵심 설계다.

---

## Transparent Scalability

CUDA programming model에서 block 간 thread는 일반적으로 서로 synchronize하지 않는다. 이 독립성 덕분에 block은 어떤 순서로 실행되어도 된다.

```text
Allowed block execution orders:
  block 0, block 1, block 2, ...
  block 127, block 126, ...
  some blocks in parallel, then remaining blocks later
```

이 특성을 transparent scalability라고 부른다. 같은 CUDA kernel이 작은 GPU에서도, 큰 GPU에서도 동작할 수 있다. 작은 GPU는 같은 grid의 block을 더 많이 sequential하게 실행하고, 큰 GPU는 더 많은 block을 동시에 실행한다. Programmer가 code를 바꾸지 않아도 hardware parallelism이 늘어나면 더 많은 block이 병렬로 처리될 수 있다.

Block 간 synchronization을 kernel 내부에서 억지로 구현하려고 하면 이 모델이 깨진다. 예를 들어 global memory에 flag를 두고 spin loop로 다른 block을 기다리는 코드를 작성하면, 기다리는 대상 block이 아직 실행조차 되지 않았을 수 있다. 그러면 실행 중인 block은 기다리고, waiting block은 SM resource가 비기를 기다리므로 deadlock이 발생할 수 있다.

| Synchronization target | CUDA kernel 안에서의 의미 |
| ---------------------- | ------------------------- |
| Threads in same block | `__syncthreads()`로 가능 |
| Threads in different blocks | 일반 kernel 내부에서 가정하면 위험 |
| All blocks in a grid | Kernel completion을 synchronization point로 사용 |

실무적으로는 block 사이에 global synchronization이 필요하면 kernel을 분리한다. 같은 stream에서 kernel을 순차 launch하거나 host/device synchronization으로 이전 kernel의 completion을 보장하면, 그 completion point가 grid-wide synchronization 역할을 한다.

---

## Warps as the Scheduling Unit

SM 내부에서 thread는 개별 thread 단위로 scheduling되지 않는다. Thread block은 warp라는 작은 묶음으로 나뉘고, warp가 SM scheduling의 기본 단위가 된다.

```text
thread block
  warp 0: threads 0..31
  warp 1: threads 32..63
  warp 2: threads 64..95
  ...
```

Warp size는 device-specific property지만, 강의 시점까지 CUDA GPU에서는 일반적으로 32 threads다. 따라서 1024-thread block은 32 warps로 나뉘고, 64-thread block은 2 warps로 나뉜다.

Warp는 "같이 움직이는 thread 묶음"이다. 같은 warp의 thread들은 같은 instruction stream을 공유한다. SM의 warp scheduler는 resident warp들 중 ready 상태인 warp를 골라 instruction을 issue한다. 어떤 warp가 memory access나 long-latency operation 때문에 기다려야 하면 scheduler는 다른 ready warp를 고른다.

Block과 warp의 차이를 구분하는 것이 중요하다.

| Unit | Meaning |
| ---- | ------- |
| Block | Programmer가 지정하는 collaboration scope |
| Warp | Hardware scheduling과 SIMD execution의 기본 단위 |
| Thread | Logical work item, register context를 갖는 execution context |

---

## SIMD Execution

같은 warp의 thread는 성능 모델상 SIMD 또는 SIMT model로 이해할 수 있다. SIMD는 single instruction, multiple data의 약자다. 하나의 instruction을 fetch/decode/dispatch하고, warp 안의 여러 thread가 각자의 data에 대해 같은 operation을 수행한다.

```text
one instruction
  -> thread 0 uses data[0]
  -> thread 1 uses data[1]
  -> thread 2 uses data[2]
  ...
  -> thread 31 uses data[31]
```

SIMD의 장점은 control overhead를 amortize하는 것이다. CPU처럼 thread마다 독립적인 instruction fetch/decode/control logic을 많이 두면 hardware budget이 control logic에 많이 쓰인다. GPU는 같은 instruction을 여러 lane에 dispatch함으로써 instruction fetch와 decode unit의 비용을 여러 execution unit에 나눠 쓴다. 그 결과 더 많은 silicon area를 arithmetic unit에 투자할 수 있다.

강의에서는 V100 SM 구조를 예로 든다. V100의 SM 하나에는 64 FP32 cores가 있고, 이것이 4개의 processing block으로 나뉜다. 각 processing block은 16 FP32 cores와 instruction dispatch unit, register file 등을 갖는다. Warp size는 32이므로, 하나의 warp가 16-core processing block에서 여러 cycle에 걸쳐 실행될 수 있다.

| V100 SM component | Lecture framing |
| ----------------- | --------------- |
| 64 FP32 cores | Floating-point operation을 수행 |
| 64 INT32 cores | Integer operation을 수행 |
| 4 processing blocks | SM 내부 execution cluster |
| 16 FP32 cores per processing block | 하나의 dispatch/control unit이 여러 core를 제어 |
| Warp scheduler | Ready warp를 선택 |
| L0 instruction cache | Processing block에 가까운 instruction cache |
| L1 instruction cache/data cache/shared memory | SM 내부에서 공유되는 on-chip resource |

이 구조는 GPU architecture마다 바뀐다. 강의는 Volta V100을 기준으로 설명하지만, Pascal이나 이후 architecture는 processing block 구성과 core 배치가 다를 수 있다. Volta 이후에는 independent thread scheduling 같은 세부 변화도 있지만, 성능을 해석할 때 변하지 않는 핵심은 "warp 단위 issue/scheduling"과 "control logic을 여러 lane에 나눠 쓰는 SIMD-style execution"이다.

---

## Control Divergence

SIMD execution의 대표적인 비용은 control divergence다. 같은 warp의 thread는 같은 instruction을 실행해야 하는데, branch 때문에 일부 thread는 `then` path를, 다른 thread는 `else` path를 원할 수 있다.

예를 들어 다음 code를 보자.

```c
if (threadIdx.x < 24) {
    /* code A */
} else {
    /* code B */
}

/* code C */
```

첫 번째 warp가 `threadIdx.x = 0..31`을 포함한다고 하면, threads 0..23은 `code A`를 실행하고 threads 24..31은 `code B`를 실행해야 한다. 하지만 warp는 같은 instruction을 함께 실행해야 한다. 따라서 hardware는 대략 다음처럼 처리한다.

1. Warp 전체가 branch condition을 평가한다.
2. `code A` path를 실행한다. 이때 threads 0..23만 active이고, threads 24..31은 inactive다.
3. `code B` path를 실행한다. 이때 threads 24..31만 active이고, threads 0..23은 inactive다.
4. Branch 이후 reconverge해서 `code C`를 함께 실행한다.

Inactive thread가 있는 동안 해당 lane은 다른 유용한 일을 하지 못한다. 이것이 SIMD efficiency를 낮춘다.

```text
SIMD efficiency = active lanes / total lanes
```

위 예제에서 `code A`를 실행할 때는 24/32 lanes만 active이고, `code B`를 실행할 때는 8/32 lanes만 active다. Branch path가 많아질수록, 또는 각 path의 instruction 수가 길어질수록 낭비가 커진다.

Loop도 divergence를 만든다. 각 thread의 loop bound가 data-dependent하면 같은 warp 안에서도 thread마다 iteration 수가 다를 수 있다.

```c
int n = A[threadIdx.x];

for (int i = 0; i < n; ++i) {
    /* code A */
}
```

어떤 thread는 4 iterations만 수행하고, 다른 thread는 1000 iterations를 수행한다면, 짧게 끝난 thread들은 warp의 나머지 thread가 loop를 끝낼 때까지 inactive 상태로 묶인다. 강의에서는 이런 상황을 "한 thread가 다른 thread들을 hostage로 잡는" 형태로 설명한다.

Control divergence를 줄이는 한 가지 실무 전략은 같은 warp에 비슷한 control path를 타는 data를 모으는 것이다. 예를 들어 loop bound가 data value에 의해 결정된다면, 가능할 때 data를 sort해서 비슷한 `n` 값을 갖는 thread들이 같은 warp에 배치되도록 만들 수 있다. 이것은 compiler가 자동으로 해결하기 어렵고, 대개 programmer나 algorithm design이 책임져야 하는 문제다.

---

## Latency Hiding

GPU는 long-latency operation을 없애기보다 숨긴다. 한 warp가 memory load miss, DRAM access, multi-cycle arithmetic, dependency chain 같은 이유로 다음 instruction을 실행할 수 없으면, scheduler는 그 warp를 core에서 잠시 빼고 다른 ready warp를 실행한다.

```text
warp 0 executes
  -> long-latency memory access
  -> warp 0 waits
warp scheduler selects warp 1
warp 1 executes
  -> long-latency operation
  -> warp 1 waits
warp scheduler selects warp 2
...
warp 0 becomes ready again
  -> warp 0 resumes
```

이것은 CPU의 context switch와 다르다. GPU에서 resident warp의 registers와 program counter는 여전히 SM에 남아 있다. Warp를 바꾼다고 해서 register context를 memory에 저장하고 다른 thread의 context를 memory에서 복원하는 heavy context switch를 수행하는 것이 아니다. 이미 resident 상태인 여러 warp 중 ready warp를 선택하는 hardware multithreading에 가깝다.

CPU와 GPU의 latency 전략은 다르다.

| Processor style | Main goal | Typical mechanisms |
| --------------- | --------- | ------------------ |
| CPU | Low latency for individual thread | Out-of-order execution, large cache, branch prediction, wide issue |
| GPU | High throughput for many data-parallel operations | Many cores, many resident warps, warp scheduling, latency hiding |

따라서 "GPU가 CPU보다 빠르다"는 문장은 정확하지 않다. 단일 element를 더하는 latency만 보면 CPU가 더 빠를 수 있다. GPU가 강한 영역은 millions of elements처럼 massive data parallel workload를 높은 throughput으로 처리하는 경우다.

---

## Occupancy

Occupancy는 SM에 active 또는 resident 상태로 올라간 warp/thread 수가 hardware가 허용하는 최대치에 비해 얼마나 되는지를 나타내는 비율이다. 강의에서는 active를 "실제로 이 순간 core에서 실행 중"이 아니라 "SM에 assigned/resident"라는 의미로 사용한다.

```text
occupancy = active resident warps or threads / maximum resident warps or threads
```

V100 예시에서 SM은 최대 2048 threads를 resident 상태로 둘 수 있다. 어떤 kernel configuration이나 resource 사용량 때문에 SM에 1024 threads만 resident로 올라갈 수 있다면 occupancy는 50%다.

높은 occupancy가 보통 유리한 이유는 latency hiding 때문이다. Resident warp가 많을수록 한 warp가 stall될 때 scheduler가 선택할 ready warp를 찾을 확률이 높아진다. 하지만 강의에서는 항상 occupancy가 높을수록 성능이 좋은 것은 아니라고도 말한다. 일부 kernel에서는 lower occupancy가 더 나을 수 있다. 그래도 일반적인 출발점은 occupancy를 충분히 확보하는 것이다.

Occupancy를 제한하는 요인은 여러 가지다.

| Constraint | How it limits occupancy |
| ---------- | ----------------------- |
| Max threads per SM | SM에 resident로 둘 수 있는 thread 수의 상한 |
| Max blocks per SM | 작은 block을 너무 많이 필요로 할 때 block count가 먼저 limit이 됨 |
| Max threads per block | block 하나가 가질 수 있는 thread 수의 상한 |
| Registers per SM | Thread당 register 사용량이 많으면 resident thread 수가 줄어듦 |
| Shared memory per SM | Block당 shared memory 사용량이 많으면 resident block 수가 줄어듦 |

---

## Choosing a Block Size

Block size는 occupancy에 직접 영향을 준다. 강의의 V100 예제를 그대로 정리하면 다음과 같다.

Assume:

```text
max threads per SM = 2048
max blocks per SM  = 32
max threads per block = 1024
```

| Threads per block | Blocks needed for 2048 threads | Hardware result | Occupancy implication |
| ----------------- | ------------------------------ | --------------- | --------------------- |
| 256 | 8 blocks | 8 <= 32, fits cleanly | Full occupancy possible |
| 32 | 64 blocks | 64 > 32, block limit reached | Only `32 * 32 = 1024` threads, 50% occupancy |
| 768 | 2 full blocks + 512 leftover threads | Cannot schedule a partial 768-thread block | 1536 resident threads, leftover capacity wasted |

이 예제는 block size를 작게 잡는 것이 항상 좋은 선택이 아님을 보여준다. 32 threads per block은 warp 하나라서 단순해 보이지만, V100에서 full 2048 resident threads를 채우려면 64 blocks가 필요하다. 그런데 SM당 block 수 상한이 32이므로 1024 threads에서 막힌다.

반대로 768 threads per block은 block 수 상한에는 걸리지 않지만, 2048이 768로 나누어떨어지지 않는다. 두 block을 올리면 1536 threads가 resident가 되고, 남은 512 thread slots는 768-thread block 하나를 더 올리기에는 부족하다. Block은 partial하게 올라갈 수 없기 때문이다.

실무적으로 block size를 고를 때는 다음 조건을 함께 본다.

* Warp size의 배수인가?
* Target GPU의 max threads per SM을 잘 나눌 수 있는가?
* Max blocks per SM에 너무 빨리 걸리지 않는가?
* Kernel의 register 사용량이 resident thread 수를 줄이지 않는가?
* Block당 shared memory 사용량이 resident block 수를 줄이지 않는가?
* Memory access pattern과 coalescing에 불리하지 않은가?

> [!TIP]
> 초반 CUDA code에서는 128, 256, 512 threads per block을 먼저 실험하는 경우가 많다. 하지만 최종 선택은 target GPU의 occupancy, register pressure, shared memory usage, memory access pattern을 함께 보고 benchmark로 확인해야 한다.

---

## Querying Device Properties

CUDA는 device property를 조회하는 API를 제공한다. 강의에서는 `cudaGetDeviceProperties`를 언급한다. 이 API를 사용하면 현재 GPU의 max threads per block, max threads per multiprocessor, SM 수 같은 정보를 확인할 수 있다.

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("name: %s\n", prop.name);
printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
printf("maxThreadsPerMultiProcessor: %d\n",
       prop.maxThreadsPerMultiProcessor);
```

이 값들은 architecture마다 다르다. 따라서 특정 강의 slide의 V100 수치를 절대 규칙으로 외우기보다, target GPU에서 직접 property를 확인하는 습관이 필요하다.

---

## Practical Tips and Notes

### Treat Blocks as Independent Work Units

CUDA kernel을 설계할 때 block은 독립적으로 실행 가능한 work unit이어야 한다. Block 간에 실행 순서, 동시 실행 여부, 같은 SM 배치 여부를 가정하면 transparent scalability가 깨지고 deadlock 위험이 생긴다.

> [!WARNING]
> Kernel 내부에서 global memory flag와 spin loop로 block 간 barrier를 직접 만들지 마라. 일부 block이 아직 scheduling되지 않은 상태라면 실행 중인 block과 waiting block이 서로를 기다릴 수 있다.

### Separate Block Synchronization from Warp Reconvergence

`__syncthreads()`는 block-level barrier다. Control divergence 뒤의 reconvergence는 warp 내부 execution control이다. 둘은 서로 다른 개념이다.

| Concept | Scope | Trigger |
| ------- | ----- | ------- |
| Warp reconvergence | Threads in same warp | Divergent branch/loop 이후 |
| `__syncthreads()` | Threads in same block | Programmer가 명시적으로 호출 |
| Kernel boundary | All blocks in grid | Kernel launch completion |

Warp 하나가 loop divergence 때문에 늦어지는 것과, block 안의 모든 warp가 `__syncthreads()`에서 기다리는 것은 다른 문제다. 성능 분석에서 이 둘을 구분해야 한다.

### Watch Divergence at Warp Granularity

Branch가 있다는 사실 자체보다 같은 warp 안에서 branch direction이 갈리는지가 중요하다. 모든 thread가 같은 branch를 타면 divergence 비용은 작다. Warp 안에서 half는 `then`, half는 `else`로 갈리면 두 path를 모두 실행하면서 inactive lane이 생긴다.

### Use Occupancy as a Diagnostic, Not a Goal by Itself

Occupancy는 latency hiding 능력을 설명하는 중요한 지표지만 최종 성능 지표는 아니다. Occupancy를 높이려고 register를 지나치게 줄이면 spilling이 생길 수 있고, shared memory tiling을 줄이면 memory traffic이 늘 수 있다. 좋은 kernel은 occupancy, memory bandwidth, arithmetic intensity, instruction mix를 함께 본다.

### Block Size Selection Checklist

| Check | Why it matters |
| ----- | -------------- |
| Multiple of 32 | Warp를 partial하게 낭비하지 않기 위한 기본 조건 |
| 128/256/512 baseline | 여러 GPU에서 실험하기 쉬운 일반적인 시작점 |
| Max blocks per SM | 너무 작은 block은 block count limit에 걸릴 수 있음 |
| Register usage | Thread당 register가 많으면 occupancy가 제한됨 |
| Shared memory per block | Block당 shared memory가 크면 resident block 수가 줄어듦 |
| Benchmark | Occupancy만으로는 실제 runtime을 예측할 수 없음 |

### Quick Reference

| Symptom | First check |
| ------- | ----------- |
| Kernel hangs | Cross-block synchronization or spin waiting |
| Good occupancy but poor speed | Memory coalescing, divergence, arithmetic intensity |
| Low occupancy | Block size, register count, shared memory per block |
| Branch-heavy kernel is slow | Warp-level control divergence |
| Small block size underperforms | Max blocks per SM limiting resident threads |
| High register kernel underperforms | Register pressure or spilling |

---

## Lecture Summary

이번 강의는 CUDA programming model과 GPU architecture를 연결했다. CUDA grid는 block으로 구성되고, block은 SM에 배치된다. 같은 block의 thread는 같은 SM에 있어야 하므로 `__syncthreads()`와 shared memory 같은 block-local collaboration이 가능하다. 반대로 block 간에는 synchronization을 가정하지 않아야 한다. 이 독립성이 있어야 같은 kernel이 작은 GPU와 큰 GPU에서 모두 실행되는 transparent scalability가 가능하다.

SM 내부에서는 thread가 warp 단위로 scheduling된다. Warp는 보통 32 threads이고, 같은 warp의 thread는 SIMD 방식으로 같은 instruction을 실행한다. SIMD는 instruction fetch/decode/control logic 비용을 여러 execution lane에 나눠 쓰게 해 throughput-oriented GPU architecture를 가능하게 한다. 하지만 branch나 data-dependent loop가 warp 내부에서 갈라지면 control divergence가 발생하고, inactive lane 때문에 SIMD efficiency가 낮아진다.

GPU는 CPU처럼 single-thread latency를 낮추는 데 집중하지 않는다. 대신 많은 warp를 SM에 resident 상태로 두고, 한 warp가 long-latency operation을 기다릴 때 다른 ready warp를 실행한다. 이 latency hiding이 GPU throughput의 핵심이다. Occupancy는 SM에 resident한 warp/thread 수가 최대 가능치에 비해 얼마나 되는지를 나타내며, block size, register usage, shared memory usage, hardware limit에 의해 결정된다.

---

## Key Terms

| Term | Meaning |
| ---- | ------- |
| Streaming multiprocessor | CUDA thread block을 실행하는 GPU의 주요 execution unit |
| SM | Streaming multiprocessor의 약어 |
| Core | Arithmetic instruction을 수행하는 execution unit |
| Global memory | 모든 SM이 접근할 수 있는 GPU device memory |
| Shared memory | 같은 block의 thread들이 공유하는 SM-local memory |
| Thread block | 같은 SM에 배치되고 협력할 수 있는 CUDA thread group |
| Resident thread | SM resource를 확보해 active 상태로 올라가 있는 thread |
| Barrier synchronization | 여러 thread가 특정 지점에 모두 도착할 때까지 기다리는 synchronization |
| `__syncthreads()` | CUDA block-level barrier primitive |
| Transparent scalability | 같은 code가 hardware parallelism 크기에 따라 block을 더 병렬 또는 순차적으로 실행할 수 있는 특성 |
| Warp | SM scheduling의 기본 단위, 보통 32 threads |
| SIMD | Single instruction, multiple data execution model |
| Control divergence | 같은 warp의 thread들이 서로 다른 control path를 택하는 상황 |
| SIMD efficiency | SIMD execution 중 active lane 비율 |
| Reconvergence | Divergent path 이후 warp thread들이 다시 같은 instruction stream으로 합류하는 것 |
| Latency hiding | 한 warp가 stall될 때 다른 ready warp를 실행해 pipeline idle을 줄이는 기법 |
| Occupancy | SM에 resident한 warp/thread 수와 hardware maximum의 비율 |
| Register pressure | Thread당 register 사용량이 많아 occupancy나 spilling에 영향을 주는 상태 |
| `cudaGetDeviceProperties` | CUDA device capability와 limit을 조회하는 API |

---

## Questions

1. GPU에서 SM은 어떤 역할을 하는가?
2. V100 예시에서 SM 수와 SM당 FP32 core 수는 각각 얼마인가?
3. CUDA thread block은 hardware에서 어떤 단위로 SM에 배치되는가?
4. Block 안의 thread 일부만 먼저 SM에 올릴 수 없는 이유는 무엇인가?
5. `__syncthreads()`는 어떤 scope의 synchronization인가?
6. Shared memory는 어떤 thread들이 공유할 수 있는가?
7. Block 간 synchronization을 kernel 내부에서 직접 구현하면 왜 deadlock 위험이 있는가?
8. Transparent scalability는 무엇을 의미하는가?
9. Warp는 무엇이며 일반적인 warp size는 얼마인가?
10. SIMD execution의 hardware상 장점은 무엇인가?
11. Control divergence는 언제 발생하는가?
12. Divergent `if-else`에서 inactive thread는 왜 다른 일을 할 수 없는가?
13. Data-dependent loop bound가 warp divergence를 만들 수 있는 이유는 무엇인가?
14. GPU의 latency hiding은 어떤 방식으로 동작하는가?
15. GPU warp switching과 CPU context switching의 차이는 무엇인가?
16. Occupancy는 어떻게 정의되는가?
17. V100에서 32 threads per block이 occupancy를 제한할 수 있는 이유는 무엇인가?
18. 768 threads per block 예제에서 512 thread slots가 남아도 더 배치하지 못하는 이유는 무엇인가?
19. Register usage가 occupancy를 낮출 수 있는 이유는 무엇인가?
20. `cudaGetDeviceProperties`로 확인할 수 있는 정보에는 무엇이 있는가?

---

## Answers

1. SM은 thread block을 resident 상태로 두고, block 내부 thread를 warp 단위로 scheduling해 실행하는 GPU의 주요 execution unit이다.
2. 강의의 V100 예시에서는 80 SMs, SM당 64 FP32 cores다.
3. Thread block 전체가 하나의 SM에 배치된다.
4. 일부 thread만 실행되면 `__syncthreads()` 같은 barrier에서 먼저 실행된 thread가 아직 scheduling되지 않은 thread를 기다리고, scheduling되지 않은 thread는 resource가 비기를 기다리는 deadlock이 가능하기 때문이다.
5. 같은 block 안의 thread들을 대상으로 하는 block-level synchronization이다.
6. 같은 block에 속하고 같은 SM에 배치된 thread들이 shared memory를 공유할 수 있다.
7. 기다리는 대상 block이 아직 SM에 scheduling되지 않았을 수 있고, 실행 중인 block이 끝나지 않으면 그 block은 시작할 수 없어 서로 기다리는 deadlock이 생길 수 있다.
8. Block이 서로 독립적이어서 같은 kernel이 작은 GPU에서는 더 순차적으로, 큰 GPU에서는 더 병렬로 실행될 수 있는 특성이다.
9. Warp는 SM scheduling과 SIMD execution의 기본 thread 묶음이며, 일반적으로 32 threads다.
10. 하나의 instruction fetch/decode/dispatch logic을 여러 execution lane이 공유해 control overhead를 줄이고 arithmetic unit을 더 많이 둘 수 있다.
11. 같은 warp의 thread들이 branch나 loop에서 서로 다른 execution path 또는 다른 iteration 수를 요구할 때 발생한다.
12. 같은 warp의 thread는 같은 instruction을 실행해야 하므로, 현재 path에 속하지 않은 thread는 다른 instruction을 실행하지 못하고 inactive 상태가 된다.
13. Thread마다 loop bound가 다르면 일부 thread는 먼저 loop를 끝내지만, 같은 warp의 나머지 thread가 loop를 계속 실행하는 동안 같이 묶여 있어야 하기 때문이다.
14. 한 warp가 long-latency operation으로 stall되면 scheduler가 다른 ready warp를 선택해 pipeline이 idle 상태로 남지 않게 한다.
15. GPU에서는 resident warp의 register와 program counter가 SM에 남아 있고 scheduler가 ready warp를 고른다. CPU context switching처럼 register context를 memory에 저장하고 복원하는 heavy operation이 아니다.
16. SM에 resident한 active warp 또는 thread 수를 hardware가 허용하는 최대 resident warp/thread 수로 나눈 비율이다.
17. V100에서 max threads per SM은 2048이고 max blocks per SM은 32다. 32-thread block으로 2048 threads를 채우려면 64 blocks가 필요하지만 SM은 32 blocks까지만 지원하므로 1024 threads에서 제한된다.
18. Block은 partial하게 배치될 수 없다. 768-thread block을 하나 더 올리려면 768 slots가 필요하므로 512 slots만으로는 부족하다.
19. SM의 register file은 유한하다. Thread당 register 수가 많으면 더 적은 thread만 resident 상태로 둘 수 있어 occupancy가 낮아진다.
20. GPU 이름, SM 수, max threads per block, max threads per multiprocessor, block/grid dimension limit 등 device capability와 resource limit을 확인할 수 있다.
