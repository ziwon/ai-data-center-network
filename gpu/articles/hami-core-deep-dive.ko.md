# HAMi-core Deep Dive: `libvgpu.so`는 GPU를 어떻게 나누는가

HAMi의 Kubernetes 쪽 구성은 Pod를 배치하고, GPU memory/core quota를 정하고,
컨테이너에 `libvgpu.so`를 주입한다. 하지만 GPU 분할의 가장 흥미로운 부분은 그
다음이다. 컨테이너 안에서 `libvgpu.so`가 CUDA와 NVML 호출 경로에 끼어들어
application이 보는 GPU의 크기와 사용할 수 있는 자원을 바꾼다.

HAMi-core는 이 역할을 맡는 in-container GPU resource controller다. 공식 README도
HAMi-core를 CUDA runtime과 CUDA driver 사이의 API call을 hijacking하는 library로
설명한다. 이 글은 HAMi-core가 만드는 “가상 GPU”가 정확히 어디에서 생기는지, 그리고
그 한계가 어디에 있는지 코드 경로 중심으로 정리한다.

소스 기준은 2026년 7월 7일에 확인한 `Project-HAMi/HAMi` `2487a24`와
`Project-HAMi/HAMi-core` `8f3a89c`다. 구현은 계속 바뀔 수 있으므로, 세부 함수명보다
control flow와 enforcement 위치를 보는 편이 좋다.

## 한 줄 요약

HAMi-core는 GPU를 하드웨어적으로 자르지 않는다. 대신 `libvgpu.so`를 컨테이너에
preload하고, CUDA driver API와 NVML API를 hook해서 다음 세 가지를 수행한다.

| 기능 | 구현 위치 | 성격 |
| --- | --- | --- |
| memory virtualization | `cuDeviceTotalMem_v2`, NVML memory query hook | application이 보는 GPU memory 크기를 quota처럼 보이게 함 |
| memory enforcement | `cuMemAlloc_v2`, `cuMemAllocManaged`, allocator/accounting | quota 초과 allocation을 OOM으로 막음 |
| compute throttling | `cuLaunchKernel`, `cuLaunchKernelEx`, `rate_limiter` | kernel launch 경로에서 software rate limiting 적용 |

가장 중요한 구분은 memory와 compute의 제한 강도가 다르다는 점이다. Memory allocation은
명시적인 API event라서 quota check와 accounting을 붙이기 쉽다. 반면 compute는 SM을
물리적으로 분할하는 것이 아니라 kernel launch 시점의 software throttling에 가깝다.

![HAMi-core hook path](assets/hami-core-hook-path.svg)

## Kubernetes에서 컨테이너 안으로 넘어오는 값

HAMi-core는 혼자 동작하지 않는다. Kubernetes 쪽 HAMi device plugin이 컨테이너 시작
전에 필요한 값을 넣어 준다. NVIDIA device plugin의 `Allocate` 경로는 non-MIG 모드에서
대략 다음 항목을 container response에 넣는다.

| 주입 항목 | 의미 |
| --- | --- |
| `CUDA_DEVICE_MEMORY_LIMIT_<index>` | logical GPU별 memory quota. 예: `CUDA_DEVICE_MEMORY_LIMIT_0=12000m` |
| `CUDA_DEVICE_SM_LIMIT` | compute 사용률 상한. 예: `40` |
| `CUDA_DEVICE_MEMORY_SHARED_CACHE` | process 간 memory accounting에 쓰는 cache 경로 |
| `CUDA_OVERSUBSCRIBE` | memory scaling/oversubscription 설정 전달 |
| `LIBCUDA_LOG_LEVEL` | HAMi-core log level |
| `/usr/local/vgpu/libvgpu.so` 또는 hook path의 `libvgpu.so` | 실제 hook library |
| `/etc/ld.so.preload` | container process가 `libvgpu.so`를 먼저 로드하게 만드는 장치 |
| `/tmp/vgpulock` | 여러 process가 공유하는 lock directory |

즉 HAMi scheduler가 “이 Pod는 GPU memory 12GB, core 40%”라고 결정하면, device plugin은
그 결정을 환경변수와 mount로 바꿔 컨테이너에 전달한다. HAMi-core는 이 값을 읽어
application process 안에서 enforcement를 시작한다.

이 경계가 HAMi의 핵심이다.

```text
Kubernetes scheduler decision
  -> Pod annotation
  -> Device Plugin Allocate
  -> env + mount injection
  -> libvgpu.so runtime hook
```

## Hook은 어디에서 걸리는가

HAMi-core의 진입점은 `src/libvgpu.c`다. 여기서 가장 눈에 띄는 구현은 `dlsym` 자체를
override하는 부분이다. 많은 CUDA framework는 CUDA driver symbol을 직접 link하거나
`dlsym`으로 찾는다. HAMi-core는 `dlsym(handle, symbol)` 호출을 가로챈 뒤, symbol 이름이
`cu...`이면 CUDA hook table을, `nvml...`이면 NVML hook table을 먼저 확인한다.

단순화하면 다음과 같다.

```text
application calls dlsym("cuMemAlloc_v2")
  -> libvgpu.so dlsym override
  -> symbol starts with "cu"
  -> return HAMi-core cuMemAlloc_v2 wrapper
  -> wrapper does quota/accounting
  -> call the real cuMemAlloc_v2 when allowed
```

`libvgpu.c`에는 `DLSYM_HOOK_FUNC(cuMemAlloc_v2)`,
`DLSYM_HOOK_FUNC(cuDeviceTotalMem_v2)`, `DLSYM_HOOK_FUNC(cuLaunchKernel)` 같은 entry가
길게 나열되어 있다. 별도 `src/cuda/hook.c`의 `cuda_library_entry[]`는 hook 대상 CUDA
symbol 목록을 관리한다. 이 목록에는 memory allocation, context, stream, kernel launch,
CUDA graph, virtual memory API까지 포함된다.

NVML도 같은 방식으로 hook된다. `__dlsym_hook_section_nvml()`는
`nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`,
`nvmlDeviceGetComputeRunningProcesses` 같은 관측 API를 가로챈다. 그래서 `nvidia-smi`나
framework의 NVML 기반 모니터링도 HAMi-core가 만든 memory view를 보게 된다.

## 초기화는 두 단계다

HAMi-core의 초기화는 `preInit()`과 `postInit()`로 나뉜다.

| 단계 | 호출 계기 | 주요 작업 |
| --- | --- | --- |
| `preInit()` | CUDA symbol lookup 또는 `cuInit` 전후 | logging 초기화, real `dlsym` 확보, real CUDA library loading, hook table 초기화 |
| `postInit()` | `cuInit` 성공 후 또는 kernel launch 전 `ensure_post_init()` | allocator 초기화, visible device mapping, host PID 탐지, utilization watcher 초기화 |

이 분리가 필요한 이유는 CUDA process의 실제 상태가 `cuInit` 전후로 다르기 때문이다.
CUDA library symbol을 hook하는 작업은 일찍 해야 하지만, process가 어떤 GPU context를
만들었는지, NVML에서 어떤 host PID로 보이는지는 CUDA 초기화 이후에야 안정적으로 알 수
있다.

`postInit()`에서 특히 중요한 단계는 host PID 탐지다. 컨테이너 내부 PID와 host에서
NVML이 보는 PID가 다를 수 있기 때문에, HAMi-core는 NVML의 running process 목록을
비교하고 primary context를 retain한 뒤 새로 나타난 PID를 찾아 accounting에 연결한다.
이 과정이 성공하면 `pidfound=1`이 되고, kernel launch rate limiter도 활성화될 수 있다.
실패하면 container PID 기반 accounting fallback으로 내려간다.

최근 코드에서는 `postInit()`의 host PID detection을 shared memory semaphore로
직렬화한다. 별도로 `utils.c`에는 `/tmp/vgpulock/lock`에 `flock()`을 거는 unified lock
구현도 남아 있다. 이 두 종류의 동기화는 모두 HAMi-core가 process-local library이면서도
node/process 간 공통 accounting을 해야 한다는 구조적 사실을 보여 준다.

## Memory virtualization: GPU가 작아 보이게 만들기

GPU memory 분할에서 첫 번째 illusion은 “이 GPU의 총 memory가 quota만큼만 있는 것처럼
보이게 하는 것”이다. 예를 들어 물리 GPU가 80GB여도 Pod가 `gpumem: 12000`을 받았다면,
application은 12GB GPU를 받은 것처럼 동작해야 한다.

CUDA driver API 쪽에서는 `cuDeviceTotalMem_v2` hook이 이 역할을 한다. 원래 함수는
물리 GPU의 total memory를 돌려주지만, HAMi-core wrapper는
`get_current_device_memory_limit(dev)` 값을 `bytes`에 넣고 `CUDA_SUCCESS`를 반환한다.

이것은 application compatibility에 중요하다. 많은 framework는 model load 전에
`total_memory`를 보고 batch size, workspace, cache 정책을 정한다. 여기서 물리 GPU
전체 memory가 보이면, framework가 quota보다 큰 plan을 세운 뒤 나중에 OOM을 맞을 수
있다.

NVML hook도 비슷한 이유로 필요하다. `nvidia-smi`와 monitoring agent가 보는 memory
total/used/free 값이 quota와 맞아야 사용자 경험과 과금 지표가 일관된다.

하지만 이 단계는 어디까지나 관측값의 virtualization이다. 실제 물리 GPU memory가
잘려서 별도 address space가 생긴 것은 아니다. 진짜 enforcement는 allocation path에서
일어난다.

## Memory enforcement: allocation 앞에서 OOM을 만든다

Memory quota enforcement의 중심은 `src/cuda/memory.c`와 `src/allocator/allocator.c`다.

대표 경로는 다음과 같다.

```text
cuMemAlloc_v2(dptr, bytesize)
  -> ENSURE_RUNNING()
  -> allocate_raw()
  -> add_chunk()
  -> oom_check(dev, bytesize)
  -> real cuMemAlloc_v2 or cuMemoryAllocate
  -> add_gpu_device_memory_usage()
```

`oom_check(dev, addon)`는 현재 device memory usage와 quota를 비교한다. 새 allocation을
더했을 때 limit을 넘으면 `CUDA_ERROR_OUT_OF_MEMORY` 계열 실패를 반환한다. 이 실패는
application 입장에서는 일반 CUDA OOM처럼 보인다.

최근 allocator 구현에서 눈에 띄는 점은 비싼 GPU allocation을 lock 밖에서 수행하려는
구조다. `add_chunk()`는 먼저 OOM pre-check를 하고, 실제 `cuMemAlloc_v2` 또는
`cuMemoryAllocate`를 호출한 뒤, tracking list와 shared usage update는 mutex 안에서
처리한다. 중간에 다른 process가 memory를 소비했을 수 있으므로 tracking 직전 OOM check를
한 번 더 수행하고, 초과하면 방금 받은 allocation을 free한다.

이 설계는 다음 trade-off를 가진다.

| 선택 | 장점 | 남는 위험 |
| --- | --- | --- |
| allocation 전 OOM pre-check | quota 초과를 빠르게 차단 | check와 실제 allocation 사이 race 가능 |
| GPU allocation을 lock 밖에서 수행 | global lock hold time 감소 | tracking 직전 재검사가 필요 |
| shared usage accounting | 여러 process를 하나의 quota 아래 묶을 수 있음 | stale process slot cleanup과 lock contention이 필요 |

`cuMemAllocManaged`, `cuMemAllocPitch_v2`, host allocation 계열도 별도 wrapper를 갖는다.
모든 CUDA memory API가 완전히 같은 정확도로 accounting되는 것은 아니므로, 새 CUDA API가
추가될 때 hook table과 allocator 경로가 따라가야 한다. HAMi-core에 CUDA hook
consistency를 검사하는 script가 있는 이유도 여기에 있다.

## Multi-process accounting: 왜 shared region이 필요한가

하나의 Pod 안에서도 GPU process는 여러 개일 수 있다. Python dataloader, model worker,
vLLM engine process, framework child process가 모두 GPU memory를 잡을 수 있다. quota가
컨테이너 단위라면 process별 사용량을 합쳐야 한다.

HAMi-core는 이를 위해 shared memory region과 semaphore를 사용한다. `multiprocess`
디렉터리에는 memory limit, utilization watcher, shared region tool이 분리되어 있다.
핵심은 process-local allocator list만으로는 충분하지 않다는 점이다.

```text
process A allocates 4GB
process B allocates 5GB
process C tries to allocate 6GB

quota = 12GB
current usage = 9GB
C must fail even if C itself has allocated 0GB so far
```

그래서 allocation path는 per-process local list와 shared usage accounting을 함께
갱신한다. Process가 종료될 때는 exit handler가 slot을 정리해야 하고, 비정상 종료가
있으면 stale slot cleanup이 필요하다. `oom_check()`가 limit 초과 시 `clear_proc_slot`
계열 정리를 시도하는 이유도 이 때문이다.

운영 관점에서는 이 구조가 두 가지를 의미한다.

첫째, HAMi-core는 GPU memory quota를 process-local limit으로만 구현하지 않는다. 같은
quota를 공유하는 여러 process를 묶어 보기 때문에 inference worker pool 같은 workload에
적합하다.

둘째, shared accounting은 lock과 cleanup 문제를 피할 수 없다. 고밀도 workload에서 많은
process가 동시에 `cuInit`이나 allocation을 수행하면, startup latency나 allocation
latency가 튈 수 있다.

## Compute limit: SM partition이 아니라 rate limiter다

`CUDA_DEVICE_SM_LIMIT` 또는 HAMi의 `gpucores`는 이름 때문에 하드웨어 SM partition처럼
오해되기 쉽다. HAMi-core의 non-MIG 경로에서는 그렇지 않다.

Kernel launch wrapper를 보면 `cuLaunchKernel`과 `cuLaunchKernelEx`에서
`ensure_post_init()`, `pre_launch_kernel()`, `rate_limiter(...)`를 거친 뒤 real CUDA
launch 함수를 호출한다. `rate_limiter`는 현재 device, grid count, block count, cached
SM limit, utilization policy를 보고 software token을 조정한다.

중요한 점은 이 제한이 “SM 40%를 이 container에 독점 배정”한다는 뜻이 아니라는 것이다.
더 정확한 표현은 다음과 같다.

```text
kernel launch stream을 관찰하고 지연시켜
장기 평균 compute usage가 설정한 limit 근처에 머물도록 유도한다.
```

따라서 memory quota와 compute quota의 실패 모드가 다르다.

| 항목 | memory quota | compute quota |
| --- | --- | --- |
| 개입 지점 | allocation API | kernel launch API |
| 실패 형태 | `CUDA_ERROR_OUT_OF_MEMORY` | launch 지연, throughput 감소 |
| 강제력 | 비교적 명확함 | workload shape에 민감한 soft limit |
| 하드웨어 격리 | 없음 | 없음 |
| 주요 관찰 지표 | peak memory, OOM, model load 성공률 | p95/p99 latency, throughput, neighbor impact |

짧고 큰 kernel을 드문드문 날리는 workload와, 작은 kernel을 매우 자주 날리는 workload는
같은 `gpucores` 값에서도 체감 성능이 다를 수 있다. Rate limiter가 kernel launch 경로에
있기 때문이다.

## NVML hook: `nvidia-smi`가 보는 세계

HAMi-core가 CUDA만 hook했다면 application의 memory allocation은 제한할 수 있어도,
사용자와 운영자가 보는 GPU 상태는 물리 GPU 그대로였을 것이다. 그래서 NVML hook이
중요하다.

NVML은 `nvidia-smi`, DCGM 계열 exporter, framework monitoring 코드가 GPU 상태를 읽는
주요 경로다. HAMi-core는 `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`,
`nvmlDeviceGetComputeRunningProcesses` 등 다수의 NVML symbol을 hook한다.

이 hook은 두 가지 목적을 갖는다.

| 목적 | 설명 |
| --- | --- |
| 사용자 경험 | 컨테이너 안의 `nvidia-smi`가 quota 기준 memory를 보여 주게 함 |
| accounting | host PID, process utilization, memory usage를 HAMi-core shared state와 연결 |

이 때문에 HAMi-core는 “제한기”이면서 동시에 “관측값 변환기”다. GPU를 실제로 나누는
것만큼, 사용자가 나뉜 GPU를 받은 것처럼 보이게 만드는 것도 중요한 기능이다.

## 어디까지 믿을 수 있는 격리인가

HAMi-core의 구조를 보면 격리 수준을 과대평가하면 안 된다는 결론이 자연스럽게 나온다.

| 경계 | HAMi-core non-MIG에서의 의미 |
| --- | --- |
| memory capacity | CUDA allocation path에서 quota를 적용 |
| SM/compute | kernel launch path에서 soft throttling |
| L2 cache, memory bandwidth | 강한 분리 없음 |
| PCIe/NVLink bandwidth | 강한 분리 없음 |
| fault isolation | MIG/vGPU 같은 hardware/virtualization boundary보다 약함 |
| 보안 경계 | 신뢰하지 않는 tenant 간 hard boundary로 보기 어려움 |

즉 HAMi-core는 utilization 개선과 운영상 quota에는 매우 유용하지만, hardware
partition은 아니다. 외부 고객이 섞이는 cloud tenancy, 강한 fault isolation, noisy
neighbor 차단이 핵심이면 MIG, vGPU, SR-IOV 같은 backend를 별도로 검토해야 한다.

반대로 같은 조직 내부의 inference service, notebook, batch inference, 작은 model serving
같은 환경에서는 이 trade-off가 꽤 매력적이다. Full GPU를 할당하면 낭비되는 VRAM을
작은 quota로 나누고, application을 거의 수정하지 않고도 기존 CUDA framework를 실행할 수
있기 때문이다.

## 실험으로 확인할 것

HAMi-core를 평가할 때는 문서의 기능 목록보다 failure mode를 직접 보는 편이 낫다.

| 실험 | 확인할 것 |
| --- | --- |
| `cuDeviceTotalMem` 확인 | quota를 바꾸면 framework와 `nvidia-smi`가 보는 total memory가 바뀌는가 |
| allocation OOM | quota보다 큰 `cuMemAlloc`이 물리 여유와 무관하게 실패하는가 |
| multi-process quota | process A/B/C의 합산 memory가 quota를 넘을 때 C가 실패하는가 |
| abnormal exit cleanup | GPU process kill 후 stale accounting이 정리되는가 |
| kernel throttling | `gpucores` 값을 바꾸면 throughput과 p99 latency가 어떻게 바뀌는가 |
| startup storm | 수십-수백 process 동시 `cuInit`에서 latency가 튀는가 |
| NVML consistency | container 내부 `nvidia-smi`, DCGM exporter, framework metric이 서로 맞는가 |
| bypass surface | `/etc/ld.so.preload`, static linking, privileged container 같은 우회 경로가 통제되는가 |

특히 inference platform에서는 평균 throughput보다 tail latency를 봐야 한다. HAMi-core의
compute limit은 kernel launch 경로의 software control이므로, neighbor workload가 있는
상태에서 p95/p99가 얼마나 흔들리는지가 실제 사용자 경험에 더 가깝다.

## 운영 팁

첫째, HAMi-core quota는 SKU보다 작게 잡지 말고 workload peak memory를 기준으로 잡아야
한다. LLM serving에서는 weight, KV cache, CUDA graph capture, workspace, allocator
fragmentation이 모두 peak에 영향을 준다.

둘째, `CUDA_DEVICE_SM_LIMIT`를 SLA로 표현하지 않는 편이 좋다. 사용자에게 “40% SM
독점”이라고 말하면 오해가 생긴다. “compute usage를 soft limit한다”는 운영 문구가 더
정확하다.

셋째, node pool을 나누는 것이 낫다. HAMi-core soft slicing node와 MIG node, full GPU
node를 섞으면 scheduler policy와 사용자 기대가 복잡해진다. Workload class별로 node
pool을 분리하면 failure mode도 단순해진다.

넷째, observability를 HAMi 기준으로 맞춰야 한다. 물리 GPU 전체 utilization만 보면
플랫폼 효율은 보이지만 tenant별 quota 준수 여부는 보이지 않는다. Pod별 requested quota,
actual memory usage, OOM, startup latency, p99 latency를 같이 봐야 한다.

다섯째, CUDA version upgrade 때 hook coverage를 확인해야 한다. CUDA driver API가 늘거나
framework가 새로운 allocation path를 쓰면, hook table과 accounting 경로가 따라가지 못할
수 있다.

## 결론

HAMi-core의 본질은 “CUDA/NVML을 속이는 library”가 아니라, Kubernetes가 정한 GPU quota를
CUDA process 내부에서 실행 가능한 정책으로 바꾸는 runtime enforcement layer다. Memory는
관측값 virtualization과 allocation-time OOM으로 비교적 직접적으로 제한한다. Compute는
kernel launch 경로에서 rate limiting하는 soft control이다.

그래서 HAMi-core를 평가할 때는 “GPU를 쪼갠다”는 말보다 다음 질문이 더 정확하다.

```text
이 workload의 memory allocation path와 kernel launch pattern은
libvgpu.so가 안정적으로 관찰하고 제한할 수 있는가?
```

이 질문에 대한 답이 “그렇다”라면 HAMi-core는 GPU utilization을 크게 올릴 수 있다. 답이
“아니다”라면 MIG, vGPU, full GPU allocation, 또는 workload-level batching/placement를
다시 검토해야 한다.

## References

- [Project-HAMi/HAMi-core](https://github.com/Project-HAMi/HAMi-core)
- [HAMi-core README](https://github.com/Project-HAMi/HAMi-core/blob/master/README.md)
- [HAMi-core `src/libvgpu.c`](https://github.com/Project-HAMi/HAMi-core/blob/master/src/libvgpu.c)
- [HAMi-core `src/cuda/memory.c`](https://github.com/Project-HAMi/HAMi-core/blob/master/src/cuda/memory.c)
- [HAMi-core `src/allocator/allocator.c`](https://github.com/Project-HAMi/HAMi-core/blob/master/src/allocator/allocator.c)
- [HAMi-core multiprocess memory limit](https://github.com/Project-HAMi/HAMi-core/blob/master/src/multiprocess/multiprocess_memory_limit.c)
- [HAMi NVIDIA device plugin allocation path](https://github.com/Project-HAMi/HAMi/blob/master/pkg/device-plugin/nvidiadevice/nvinternal/plugin/server.go)
