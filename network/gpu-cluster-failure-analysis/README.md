# GPU 클러스터 장애 분석: ECC, Xid, RDMA, NCCL Hang은 어떻게 연결되는가

## 목차

- [요약](#요약)
- [1. ECC 오류는 모두 같은 장애가 아니다](#1-ecc-오류는-모두-같은-장애가-아니다)
- [2. Xid는 GPU 장애의 가장 중요한 단서다](#2-xid는-gpu-장애의-가장-중요한-단서다)
- [3. nvidia-smi ECC counter를 볼 때 주의할 점](#3-nvidia-smi-ecc-counter를-볼-때-주의할-점)
- [4. Row Remapping과 RAS Repair](#4-row-remapping과-ras-repair)
- [5. NCCL Hang이 항상 네트워크 문제는 아니다](#5-nccl-hang이-항상-네트워크-문제는-아니다)
- [6. NVLink와 PCIe도 같이 봐야 한다](#6-nvlink와-pcie도-같이-봐야-한다)
- [7. DCGM은 운영 진단의 기준점으로 삼는 것이 좋다](#7-dcgm은-운영-진단의-기준점으로-삼는-것이-좋다)
- [8. 운영 판단 기준: 언제 drain하고 언제 RMA를 검토할까?](#8-운영-판단-기준-언제-drain하고-언제-rma를-검토할까)
- [9. 장애 대응 플레이북](#9-장애-대응-플레이북)
- [10. 결론](#10-결론)
- [References](#references)

## 요약

대규모 GPU 클러스터를 운영하다 보면 겉으로는 네트워크 장애처럼 보이지만 실제 원인은 GPU RAS, ECC, PCIe, NVLink 쪽에서 시작되는 경우가 있다.

대표적인 증상은 다음과 같다.

```bash
NCCL watchdog timeout
CUDA error: uncorrectable ECC error encountered
NVRM: Xid
mlx5 CQE error
GPU has fallen off the bus
NVLink error counter 증가
```

이런 로그를 보면 처음에는 NCCL, RDMA, InfiniBand, RoCE, MTU, PFC, ECN 같은 네트워크 문제를 의심하기 쉽다. 하지만 GPU 클러스터에서는 장애가 한 계층에만 머물지 않는다.

GPU 메모리 오류가 발생하면 CUDA context가 깨질 수 있고, 해당 프로세스가 참여하던 NCCL communicator가 실패하며, collective operation 전체가 멈춘 것처럼 보일 수 있다. 그 결과 운영자는 "네트워크가 멈췄다"라고 느끼지만, 실제 시작점은 GPU ECC 또는 Xid 이벤트일 수 있다.

이 글에서는 GPU 장애를 다음 흐름으로 정리한다.

```text
ECC / RAS Event
  ↓
Xid 발생
  ↓
CUDA context reset 또는 application abort
  ↓
NCCL communicator failure
  ↓
RDMA / NVLink / PCIe 계층에서 timeout 또는 error 증가
  ↓
분산 학습 job hang 또는 node drain 필요
```

핵심 원칙은 단순하다.

```text
NCCL 로그만 보지 않는다.
RDMA counter만 보지 않는다.
nvidia-smi 결과만 보지 않는다.
Xid, ECC, row remap, NVLink, PCIe, DCGM, scheduler event를 하나의 타임라인으로 묶어 본다.
```

## 1. ECC 오류는 모두 같은 장애가 아니다

ECC는 Error Correcting Code의 약자다. GPU 메모리에서 bit flip이나 memory cell 오류가 발생했을 때 이를 감지하거나 일부 수정하기 위한 기능이다.

운영 관점에서 ECC 오류는 크게 세 가지로 나눠서 봐야 한다.

| 구분 | 의미 | 운영 판단 |
| --- | --- | --- |
| Correctable ECC | 오류가 감지되었고 수정됨 | 단발성인지, 증가 추세인지 확인 |
| Uncorrectable Contained ECC | 수정은 불가능하지만 영향 범위가 격리됨 | 해당 application 재시작, GPU reset 검토 |
| Uncorrectable Uncontained ECC | 오류 영향이 격리되지 않음 | 즉시 node drain, GPU reset 또는 RMA 검토 |

중요한 점은 "ECC error가 있다"는 사실만으로 GPU 교체를 단정하면 안 된다는 것이다. 반대로 "correctable이니까 괜찮다"라고 무시해서도 안 된다.

GPU 운영에서는 절대값보다 다음 항목이 더 중요하다.

```text
- 같은 GPU에서 반복되는가?
- volatile counter가 빠르게 증가하는가?
- uncorrectable error가 발생했는가?
- row remap pending / failure 상태인가?
- DCGM diagnostic에서 hardware failure가 재현되는가?
- 동일 job, 동일 rank, 동일 GPU index에서 반복되는가?
```

## 2. Xid는 GPU 장애의 가장 중요한 단서다

NVIDIA GPU 장애 분석에서 Xid는 반드시 봐야 한다. Xid는 NVIDIA driver가 기록하는 GPU fault code이며, `dmesg`나 system log에서 확인할 수 있다.

```bash
dmesg -T | egrep -i 'xid|nvrm|ecc|pcie|aer'
journalctl -k | egrep -i 'xid|nvrm|ecc|pcie|aer'
```

운영에서 자주 보는 대표 Xid 예시는 다음과 같다. 정확한 의미와 조치는 driver, GPU 세대, MIG 사용 여부, Fabric Manager 구성에 따라 달라질 수 있으므로 공식 Xid catalog와 GPU debug guideline을 함께 확인해야 한다.

| Xid | 의미 | 운영 해석 |
| --- | --- | --- |
| Xid 31 | GPU memory page fault | kernel, driver, application memory access 문제 가능 |
| Xid 43 | GPU stopped processing | application crash, driver fault, context failure 가능 |
| Xid 48 | Uncorrectable ECC error | 심각한 ECC 이벤트로 보고 격리 필요 |
| Xid 63 | Row remap entry recorded | row remap 기록 성공, reset/repair 상태 확인 |
| Xid 64 | Row remap entry recording failed | row remap 기록 실패, RMA 후보 검토 |
| Xid 79 | GPU has fallen off the bus | PCIe, 전원, hardware, link 문제 가능 |
| Xid 94 | Contained memory error | application 단위 격리 가능, 해당 application 재시작 |
| Xid 95 | Uncontained memory error | GPU reset 필요, node drain 후 진단 |
| Xid 140 | ECC unrecovered error | 최신 driver에서 별도 보고될 수 있는 unrecovered ECC 계열 이벤트 |

특히 Xid 94와 Xid 95는 구분해야 한다.

Xid 94는 contained error다. 즉 오류가 발생했지만 GPU driver가 영향 범위를 특정 application으로 격리한 상태다. 일반적으로 해당 application 또는 job은 실패할 수 있지만, 같은 GPU에서 실행 중인 다른 application까지 반드시 영향을 받는 것은 아니다. 운영상으로는 실패한 job을 재시작하고, 편한 시점에 GPU reset을 검토한다.

반면 Xid 95는 uncontained error다. 오류 영향이 안정적으로 격리되지 않았다는 의미이므로 훨씬 심각하게 봐야 한다. 이 경우에는 node drain, GPU reset, DCGM 진단, 반복 시 RMA 검토가 필요하다.

따라서 운영 문서에는 다음처럼 쓰는 것이 안전하다.

```text
Xid 94: application-level containment이 가능한 memory error. 해당 job 재시작 및 GPU 상태 확인.
Xid 95: uncontained memory error. node drain 후 GPU reset / DCGM 진단 / RMA 검토.
```

## 3. nvidia-smi ECC counter를 볼 때 주의할 점

GPU ECC 상태는 `nvidia-smi -q -d ECC`로 확인할 수 있다.

```bash
nvidia-smi -q -d ECC
```

추가로 row remapper와 repair 상태도 같이 봐야 한다.

```bash
nvidia-smi -q | egrep -A40 'ECC Errors|Row Remapper|Remapped Rows|Repair|Pending'
```

여기서 중요한 개념이 volatile counter와 aggregate counter다.

| Counter | 의미 |
| --- | --- |
| Volatile ECC counter | driver load 이후 발생한 ECC error 수 |
| Aggregate ECC counter | GPU lifetime 동안 누적된 ECC error 수 |

운영자가 주의해야 할 점은 aggregate counter를 단순히 "현재 장애 상태"로 해석하면 안 된다는 것이다. Aggregate counter는 장기간 누적값이며, GPU 세대와 driver 정책에 따라 clearing 가능 여부가 제한될 수 있다.

따라서 aggregate counter가 0이 아니라고 해서 곧바로 장애 GPU라고 판단하면 안 된다. 대신 다음을 함께 봐야 한다.

```text
- volatile counter가 증가 중인가?
- uncorrectable error인가?
- row remap이 발생했는가?
- pending repair 상태인가?
- remapping failure가 있는가?
- 같은 GPU에서 반복되는가?
- DCGM diagnostic에서 재현되는가?
```

정리하면, ECC counter는 "교체 여부를 결정하는 단독 지표"가 아니라 "추가 진단을 시작하게 만드는 신호"로 보는 것이 맞다.

## 4. Row Remapping과 RAS Repair

A100 이후 세대부터 NVIDIA 데이터센터 GPU는 memory error recovery 기능이 강화되었다. 대표적인 기능이 dynamic page offlining, row remapping, RAS repair다.

Row remapping은 문제가 있는 HBM row를 spare row로 대체하는 기능이다. 운영 관점에서는 다음 상태를 봐야 한다.

```bash
nvidia-smi -q | egrep -A20 'Row Remapper|Remapped Rows|Pending|Failure'
```

확인해야 할 항목은 다음과 같다.

```text
- Remapped Rows
- Pending Row Remapping
- Row Remapping Failure
- Uncorrectable ECC
- Repair Pending
```

단발성 correctable ECC보다 더 주의해야 하는 것은 다음 케이스다.

```text
- uncorrectable ECC 발생
- row remap pending 상태 지속
- remapping failure 발생
- 동일 GPU에서 Xid 48, 63, 64, 94, 95 반복
- DCGM diagnostic failure
```

이 경우에는 GPU를 계속 사용하지 말고 scheduler에서 drain한 뒤 진단해야 한다.

Kubernetes 환경이라면 다음처럼 node를 cordon/drain하는 식의 운영 절차가 필요하다.

```bash
kubectl cordon <gpu-node>
kubectl drain <gpu-node> --ignore-daemonsets --delete-emptydir-data
```

Slurm 환경이라면 해당 node를 drain 상태로 전환한다.

```bash
scontrol update NodeName=<gpu-node> State=DRAIN Reason="GPU ECC/Xid investigation"
```

## 5. NCCL Hang이 항상 네트워크 문제는 아니다

분산 학습에서 NCCL hang이 발생하면 보통 네트워크를 먼저 본다.

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```

물론 RDMA, InfiniBand, RoCE 설정 문제는 매우 흔하다.

확인할 항목은 다음과 같다.

```text
- MTU mismatch
- PFC 설정 오류
- ECN 설정 오류
- RoCE GID index 불일치
- mlx5 port error 증가
- switch buffer drop
- PCIe AER error
- NIC firmware / driver mismatch
```

하지만 GPU ECC 또는 Xid 이벤트가 발생하면 NCCL 입장에서는 특정 rank가 갑자기 사라진 것처럼 보일 수 있다. 이 경우 다른 rank들은 collective operation에서 기다리다가 timeout이 발생한다.

즉 다음과 같은 패턴이 가능하다.

```text
GPU ECC error
  ↓
CUDA context failure
  ↓
해당 rank abort
  ↓
NCCL communicator failure
  ↓
다른 rank에서 watchdog timeout
  ↓
운영자는 NCCL/RDMA 장애로 인식
```

그래서 NCCL hang을 분석할 때는 네트워크 로그만 보면 안 된다.

최소한 다음 네 가지를 함께 봐야 한다.

```bash
# GPU driver / Xid
dmesg -T | egrep -i 'xid|nvrm|ecc'

# GPU 상태
nvidia-smi -q -d ECC
nvidia-smi -q | egrep -A30 'Row Remapper|Repair|NVLink'

# NIC / RDMA 상태
dmesg -T | egrep -i 'mlx5|rdma|infiniband|roce|cqe|aer'

# NCCL 로그
grep -iE 'nccl|watchdog|timeout|cuda error|unhandled' <job-log>
```

## 6. NVLink와 PCIe도 같이 봐야 한다

GPU 간 통신은 NVLink, PCIe, NIC, switch fabric이 함께 얽혀 있다. 따라서 GPU 장애를 볼 때 NVLink counter와 PCIe error도 함께 확인해야 한다.

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi nvlink --errorcounters
```

PCIe 쪽은 kernel log에서 AER error를 확인한다.

```bash
dmesg -T | egrep -i 'pcie|aer|fallen off|xid 79'
```

Xid 79가 보이면 GPU가 PCIe bus에서 사라졌다는 의미이므로, 단순 application 문제가 아니라 hardware, riser, cable, power, PCIe link, motherboard, thermal 이슈까지 봐야 한다.

## 7. DCGM은 운영 진단의 기준점으로 삼는 것이 좋다

NVIDIA DCGM은 데이터센터 GPU 운영에서 사실상 표준에 가까운 진단 도구다. 장애가 의심될 때는 `dcgmi diag`를 사용한다.

```bash
# 기본 진단
dcgmi diag -r 1

# 좀 더 긴 진단
dcgmi diag -r 2

# 장애 발생 후 post-mortem 진단
dcgmi diag -r 3

# 관리자 주도 정밀 진단
dcgmi diag -r 4
```

운영 정책은 다음처럼 가져가는 것이 좋다.

```text
Level 1: 배포/기본 상태 확인
Level 2: job 실패 후 빠른 hardware sanity check
Level 3: 장애 발생 후 post-mortem 진단
Level 4: drain 후 정밀 진단
```

Kubernetes에서는 DCGM Exporter를 통해 GPU telemetry를 Prometheus로 수집하고, 특정 조건에서 alert를 발생시키는 구조가 좋다.

예를 들어 다음 지표를 모니터링한다.

```text
- ECC correctable / uncorrectable error
- Xid event
- retired pages
- row remap 상태
- NVLink error counter
- PCIe replay / AER error
- GPU temperature
- power violation
- throttling reason
```

## 8. 운영 판단 기준: 언제 drain하고 언제 RMA를 검토할까?

운영 기준은 너무 공격적이어도 문제고, 너무 느슨해도 문제다.

아래는 보수적인 운영 기준 예시다.

| 상황 | 조치 |
| --- | --- |
| correctable ECC 단발성 | 관찰, counter 증가 여부 확인 |
| correctable ECC 반복 증가 | node 관찰 강화, DCGM 진단 |
| Xid 94 | 해당 job 실패 처리, GPU reset 또는 node drain 검토 |
| Xid 95 | 즉시 node drain, GPU reset, DCGM 진단 |
| Xid 48 | 심각한 ECC 이벤트로 보고 drain 권장 |
| row remap pending | GPU reset 또는 maintenance window 확보 |
| row remapping failure | RMA 후보 |
| Xid 79 | PCIe/hardware 문제로 보고 즉시 격리 |
| DCGM diag fail | hardware issue로 보고 교체/RMA 검토 |
| 동일 GPU 반복 장애 | scheduler에서 제외 후 vendor case 오픈 |

중요한 원칙은 다음이다.

```text
한 번의 counter보다 반복성이 중요하다.
Correctable보다 uncorrectable이 중요하다.
Aggregate보다 volatile 증가 추세가 중요하다.
Xid 하나보다 Xid + ECC + DCGM + job failure의 조합이 중요하다.
```

## 9. 장애 대응 플레이북

실제 운영에서는 다음 순서로 보면 된다.

### Step 1. Job 실패 시점 확인

```bash
kubectl logs <pod> -n <namespace>
# 또는
sacct / scontrol / slurm job log 확인
```

확인할 키워드:

```text
NCCL timeout
CUDA error
uncorrectable ECC
illegal memory access
watchdog
rank failed
```

### Step 2. GPU Xid 확인

```bash
dmesg -T | egrep -i 'xid|nvrm'
```

### Step 3. ECC / row remap 확인

```bash
nvidia-smi -q -d ECC
nvidia-smi -q | egrep -A40 'Row Remapper|Remapped Rows|Repair|Pending'
```

### Step 4. NVLink / PCIe 확인

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi nvlink --errorcounters
dmesg -T | egrep -i 'pcie|aer|fallen off'
```

### Step 5. RDMA / NIC 확인

```bash
dmesg -T | egrep -i 'mlx5|rdma|infiniband|roce|cqe'
```

가능하다면 NIC counter도 함께 본다.

```bash
ethtool -S <interface>
```

InfiniBand 환경이라면 다음도 확인한다.

```bash
ibstat
ibv_devinfo
perfquery
```

### Step 6. DCGM 진단

```bash
dcgmi diag -r 3
```

장애가 반복되면 node를 drain하고 더 긴 진단을 수행한다.

```bash
dcgmi diag -r 4
```

### Step 7. Scheduler에서 격리

Kubernetes:

```bash
kubectl cordon <gpu-node>
kubectl drain <gpu-node> --ignore-daemonsets --delete-emptydir-data
```

Slurm:

```bash
scontrol update NodeName=<gpu-node> State=DRAIN Reason="GPU Xid/ECC investigation"
```

## 10. 결론

GPU 클러스터 장애는 단순히 "GPU 문제" 또는 "네트워크 문제"로 나누기 어렵다.

NCCL hang은 RDMA 문제일 수도 있지만, GPU ECC나 Xid 이벤트로 인해 특정 rank가 죽으면서 발생할 수도 있다. RDMA error처럼 보이는 증상도 PCIe, GPU memory, NVLink, CUDA context failure와 연결되어 있을 수 있다.

따라서 GPU 인프라 운영자는 다음 관점으로 장애를 봐야 한다.

```text
NCCL 로그만 보지 않는다.
RDMA counter만 보지 않는다.
nvidia-smi 결과만 보지 않는다.
Xid, ECC, row remap, NVLink, PCIe, DCGM, scheduler event를 함께 본다.
```

최종적으로 운영 판단은 다음 기준으로 정리할 수 있다.

```text
Xid 94는 contained error로 보고 application/job 영향 범위를 확인한다.
Xid 95는 uncontained error로 보고 즉시 node drain을 검토한다.
Correctable ECC는 증가 추세와 반복성을 본다.
Uncorrectable ECC는 즉시 격리 대상으로 본다.
Aggregate ECC counter만으로 RMA를 단정하지 않는다.
Row remapping failure, DCGM failure, 반복 Xid는 RMA 후보로 본다.
```

GPU 클러스터 운영에서 중요한 것은 장애를 빠르게 "한 원인"으로 단정하는 것이 아니다. 중요한 것은 GPU, PCIe, NVLink, RDMA, NCCL, scheduler 이벤트를 하나의 타임라인으로 묶어 보는 것이다.

그 타임라인을 만들 수 있어야 진짜 원인에 가까워진다.

## References

- [NVIDIA Xid Errors](https://docs.nvidia.com/deploy/xid-errors/)
- [NVIDIA Xid Catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
- [NVIDIA GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)
- [NVIDIA GPU Memory Error Management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/latest/index.html)
- [NVIDIA nvidia-smi Documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
