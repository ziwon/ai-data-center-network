# InfiniBand 패킷 형식 참조 문서

[English](packet-format-reference.md) | **한국어**

이 문서는 InfiniBand 패킷 헤더를 비트 단위로 정리한 참조 문서이며, [메인 분석 보고서](README_KO.md)를 보완하도록 작성되었습니다. 보고서가 `ib-packets` 데이터셋에서 무엇이 관측되었는지에 초점을 둔다면, 이 문서는 IB 패킷 헤더가 실제 전송 형식에서 어떻게 배치되는지에 초점을 둡니다. 구체적인 예시가 있는 곳에는 보고서의 프레임 수준 증거로 이어지는 앵커 링크를 제공합니다.

이 문서를 사용하는 경우:

- 헥스 덤프를 읽고 필드 경계를 식별할 때
- 주어진 BTH opcode 다음에 어떤 확장 헤더가 와야 하는지 검증할 때
- AETH syndrome 값을 디코드할 때
- 모든 전송 서비스에 걸쳐 BTH opcode를 조회할 때

원천 자료: IBA Architecture Specification Volume 1 (Release 1.5), Wireshark InfiniBand dissector 필드 목록, [Tencent Cloud 전송 계층 글](https://cloud.tencent.com/developer/article/2513460).

## 목차

- [헤더 시퀀스 개요](#헤더-시퀀스-개요)
- [Local Route Header (LRH) — 8 bytes](#local-route-header-lrh--8-bytes)
- [Global Route Header (GRH) — 40 bytes](#global-route-header-grh--40-bytes)
- [Base Transport Header (BTH) — 12 bytes](#base-transport-header-bth--12-bytes)
- [Extended Transport Headers](#extended-transport-headers)
  - [DETH — 8 bytes](#deth--8-bytes-datagram-eth)
  - [RETH — 16 bytes](#reth--16-bytes-rdma-eth)
  - [AETH — 4 bytes](#aeth--4-bytes-ack-eth)
  - [AtomicETH — 28 bytes](#atomiceth--28-bytes)
  - [AtomicAckETH — 8 bytes](#atomicacketh--8-bytes)
  - [ImmDt — 4 bytes](#immdt--4-bytes-immediate-data)
  - [IETH — 4 bytes](#ieth--4-bytes-invalidate-eth)
  - [RDETH — 4 bytes](#rdeth--4-bytes-reliable-datagram-eth)
  - [XRCETH — 4 bytes](#xrceth--4-bytes-extended-reliable-connection-eth)
- [MAD Common Header — 24 bytes](#mad-common-header--24-bytes)
- [SMP Directed Route Extension](#smp-directed-route-extension)
- [IPoIB Encapsulation — 4 bytes (RFC 4391)](#ipoib-encapsulation--4-bytes-rfc-4391)
- [CRC 커버리지](#crc-커버리지)
- [BTH Opcode 마스터 표](#bth-opcode-마스터-표)
- [Operation → Extended Header 매핑](#operation--extended-header-매핑)
- [참고](#참고)

## 헤더 시퀀스 개요

IB 패킷의 실제 전송 형식은 헤더, 페이로드, CRC가 엄격한 순서로 이어진 구조입니다. 어떤 확장 헤더가 어떤 순서로 나타나는지는 `LRH.LNH`(GRH 존재 여부)와 BTH opcode(어떤 확장 헤더가 따르는지)로 완전히 결정됩니다.

```
+-----+-------+-----+-----------------------+----------+------+------+
| LRH |  GRH? | BTH | Extended Header(s)    | Payload  | ICRC | VCRC |
+-----+-------+-----+-----------------------+----------+------+------+
   8     40    12      0..28+ bytes            variable    4      2
```

- `GRH`는 `LRH.LNH = 0x3`일 때만 존재합니다.
- 확장 헤더의 종류와 순서는 [Operation → Extended Header 매핑](#operation--extended-header-매핑)을 따릅니다.
- `ICRC`와 `VCRC`는 실제 전송 형식에서 항상 패킷의 끝에 위치합니다.

## Local Route Header (LRH) — 8 bytes

LRH는 모든 패킷의 첫 IB 헤더이며 fabric 로컬 라우팅에 사용됩니다.

비트 레이아웃 (big-endian):

| Byte | 비트 패턴 | 필드 | 폭 |
| ---: | --- | --- | ---: |
| 0 | `VVVV LLLL` | VL[3:0] / LVer[3:0] | 4 + 4 |
| 1 | `SSSS RR NN` | SL[3:0] / Reserved / LNH[1:0] | 4 + 2 + 2 |
| 2..3 | `DDDDDDDD DDDDDDDD` | DLID | 16 |
| 4 | `RRRRR PPP` | Reserved / PktLen[10:8] | 5 + 3 |
| 5 | `PPPPPPPP` | PktLen[7:0] | 8 |
| 6..7 | `SSSSSSSS SSSSSSSS` | SLID | 16 |

필드 의미:

| 필드 | 설명 |
| --- | --- |
| VL | Virtual Lane (0–15). VL15는 관리 트래픽용으로 예약 |
| LVer | Link version, 현재는 항상 0 |
| SL | Service Level (0–15), QoS 클래스에 매핑 |
| LNH | Link Next Header — LRH 다음에 오는 것을 선택 |
| DLID / SLID | Destination / Source Local ID (SM이 할당) |
| PktLen | 4바이트 워드 단위의 패킷 길이, LRH와 VCRC 제외 |

LNH 인코딩:

| 값 | 의미 | LRH 다음에 오는 것 |
| ---: | --- | --- |
| `0x0` | Raw IPv6 (legacy) | IPv6 헤더 직접 |
| `0x1` | Raw IPv4 (legacy) | IPv4 헤더 직접 |
| `0x2` | IBA Local | BTH (GRH 없음) |
| `0x3` | IBA Global | GRH + BTH |

구체적 예시: 이 데이터셋의 모든 패킷은 `LNH = 0x2`를 가지므로 GRH가 디코드되지 않습니다. `infiniband.pcap` frame 10의 단계별 예제는 메인 보고서의 [ERF 캡처 해부](README_KO.md#erf-캡처-해부) 섹션을 참고하세요.

## Global Route Header (GRH) — 40 bytes

GRH는 `LRH.LNH = 0x3`일 때만 나타나며 IB 서브넷 사이의 라우팅을 신호합니다. 포맷은 IPv6를 본떴습니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0 (high 4) | IPVer | 4 |
| 0 (low 4) + 1 (high 4) | TClass | 8 |
| 1 (low 4) + 2..3 | FlowLabel | 20 |
| 4..5 | PayLen | 16 |
| 6 | NxtHdr | 8 |
| 7 | HopLmt | 8 |
| 8..23 | SGID | 128 |
| 24..39 | DGID | 128 |

비고:

- `IPVer`는 항상 6입니다.
- `NxtHdr = 0x1B`(27 십진수)는 IBA next header(BTH)를 신호합니다.
- `PayLen`은 GRH 다음부터 ICRC 시작 지점까지의 바이트를 셉니다.
- SGID/DGID는 SM이 할당한 128비트 GID입니다.

이 데이터셋은 GRH를 가진 패킷을 포함하지 않으므로, 이 섹션은 향후 서브넷 간 캡처를 해석하기 위한 순수 참조 자료입니다.

## Base Transport Header (BTH) — 12 bytes

BTH는 전송 동작, 목적지 QP, packet sequence number를 선택합니다. 모든 IBA 패킷에 나타납니다 (즉 `LNH ∈ {0x2, 0x3}`).

비트 레이아웃:

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0 | OpCode | 8 |
| 1 (bit 7) | SE (Solicited Event) | 1 |
| 1 (bit 6) | M (Migration request) | 1 |
| 1 (bits 5..4) | PadCnt | 2 |
| 1 (bits 3..0) | TVer | 4 |
| 2..3 | P_Key | 16 |
| 4 (bit 7) | F (FECN) | 1 |
| 4 (bit 6) | B (BECN) | 1 |
| 4 (bits 5..0) | Reserved | 6 |
| 5..7 | DestQP | 24 |
| 8 (bit 7) | A (AckReq) | 1 |
| 8 (bits 6..0) | Reserved | 7 |
| 9..11 | PSN | 24 |

필드 의미:

| 필드 | 비고 |
| --- | --- |
| OpCode | 상위 3비트 = 전송 서비스, 하위 5비트 = 동작. [BTH Opcode 마스터 표](#bth-opcode-마스터-표) 참고 |
| SE | Solicited Event — 응답자의 CQ 이벤트를 트리거해야 하는 SEND 또는 RDMA WRITE 메시지의 마지막 패킷에 설정 |
| M | 자동 path migration 중 request / accept 신호로 사용 |
| PadCnt | 페이로드 끝에 추가되는 0–3 바이트로 4바이트 경계에 정렬 |
| TVer | Transport header version, 현재는 항상 0 |
| P_Key | 파티션 키; 상위 비트 = full vs limited 멤버십, 하위 15비트 = 파티션 ID |
| FECN / BECN | Forward / Backward Explicit Congestion Notification |
| DestQP | 24비트 destination Queue Pair 번호 |
| AckReq | RC 트래픽에서 설정 시 응답자가 ACK를 생성해야 함 |
| PSN | 24비트 Packet Sequence Number; 2²⁴로 wrap |

알아둘 가치가 있는 PSN 동작:

- expected PSN은 QP마다 추적됩니다. PSN이 expected 값과 같은 패킷은 윈도우를 전진시킵니다.
- *duplicate range* 안의 PSN(expected보다 오래되었지만 2²³ 이내)은 재전송으로 다뤄지며 페이로드를 다시 전달하지 않고 ACK됩니다.
- duplicate range를 벗어나면서 expected보다 이른 PSN은 sequence error이며 코드 0의 NAK를 트리거합니다.

구체적 예시: `infiniband.pcap` frame 10 BTH:

```
Opcode  = 4   (RC SEND Only)
SE = 0, M = 1, PadCnt = 0, TVer = 0
P_Key   = 0xffff   (full membership, 기본 파티션)
FECN = 0, BECN = 0
DestQP  = <masked>
AckReq  = 1
PSN     = 13896277
```

## Extended Transport Headers

BTH 다음에 어떤 확장 헤더가 따르는지는 opcode가 완전히 결정합니다. IBA 사양은 이를 opcode별 표로 인코딩합니다. 동작별 요약은 [Operation → Extended Header 매핑](#operation--extended-header-매핑)에 있습니다.

### DETH — 8 bytes (Datagram ETH)

UD와 RD 동작에 필요합니다. QP0/QP1을 통한 모든 MAD 트래픽도 사용합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..3 | Q_Key | 32 |
| 4 | Reserved | 8 |
| 5..7 | SrcQP | 24 |

Q_Key 관례:

- QP0 (SMP): `Q_Key = 0`
- QP1 (GMP): `Q_Key = 0x80010000`
- 그 외 UD QP: 애플리케이션 정의; 상위 비트 설정 = privileged

구체적 예시: `infiniband.pcap` frame 1 SMP 트래픽은 `DestQP = 0x000000`, `SrcQP = 0x00000000`, `Q_Key = 0x00000000`을 사용합니다.

### RETH — 16 bytes (RDMA ETH)

`RDMA READ Request`, `RDMA WRITE First`, `RDMA WRITE Only`, with-Immediate 변종에 존재합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..7 | VA (Virtual Address) | 64 |
| 8..11 | R_Key | 32 |
| 12..15 | DMALen | 32 |

응답자는 `R_Key`에 등록된 MR과 비교해 요청을 검증해야 합니다. VA는 MR 주소 범위 안에 있어야 하고, `[VA, VA + DMALen)`은 그 경계를 벗어나면 안 되며, MR의 접근 권한에는 필요한 READ 또는 WRITE 권한이 포함되어야 합니다.

이 데이터셋에는 RETH를 가진 패킷이 없습니다.

### AETH — 4 bytes (ACK ETH)

RC와 RD ACK 패킷, 그리고 RDMA READ의 first/last/only 응답 패킷에 존재합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0 | Syndrome | 8 |
| 1..3 | MSN (Message Sequence Number) | 24 |

Syndrome 인코딩 (8비트):

| 비트 | 필드 | 폭 |
| ---: | --- | ---: |
| 7 | Reserved | 1 |
| 6..5 | OpCode | 2 |
| 4..0 | Value | 5 |

OpCode 값:

| OpCode | 의미 | Value 필드 해석 |
| --- | --- | --- |
| `00` | ACK | Credit Count (0–30); `31` = credit 정보 미제공 |
| `01` | RNR NAK | RNR Timer (고정 표에서 retry delay 선택; IBA §9.7.5.2.8 참고) |
| `10` | Reserved | — |
| `11` | NAK | NAK code: `0`=PSN seq error, `1`=invalid request, `2`=remote access error, `3`=remote operation error, `4`=invalid RD request |

구체적 예시: `infiniband.pcap` frame 11은 `Syndrome = 31` 십진수 = `0x1F` = `0 00 11111`입니다. 이는 `OpCode = 00 (ACK), Value = 11111 (credit 정보 없음)`으로 디코드됩니다. 즉 흐름 제어 힌트가 없는 정상 acknowledgement입니다. 문맥은 [메인 보고서의 frame-11 매핑](README_KO.md#infinibandpcap)을 참고하세요.

### AtomicETH — 28 bytes

RC `CmpSwap`과 `FetchAdd` 요청 패킷에 존재합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..7 | VA | 64 |
| 8..11 | R_Key | 32 |
| 12..19 | Swap Data (CmpSwap) / Add Data (FetchAdd) | 64 |
| 20..27 | Compare Data (CmpSwap) / Reserved (FetchAdd) | 64 |

Atomic 동작은 at-most-once 보장입니다 — 재시도된 요청은 QP별 outstanding-atomic 큐와 매칭되어 read-modify-write를 다시 실행하지 않고 재전송됩니다.

이 데이터셋에는 atomic 동작이 없습니다.

### AtomicAckETH — 8 bytes

원래 값, 즉 atomic 적용 전의 값을 요청자에게 다시 운반합니다. `ATOMIC Acknowledge` 패킷에서 AETH 다음에 위치합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..7 | Original Remote Data | 64 |

### ImmDt — 4 bytes (Immediate Data)

수신자의 CQE에 전달되는 32비트 불투명 데이터를 운반합니다. 이름이 "with Immediate"로 끝나는 opcode에 존재합니다. 항상 extended header 중 마지막에 위치합니다 (`RDMA WRITE Only/Last with Immediate`인 경우 RETH 다음).

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..3 | Immediate Data | 32 |

### IETH — 4 bytes (Invalidate ETH)

응답자에서 무효화할 R_Key를 운반합니다. `SEND Last with Invalidate`와 `SEND Only with Invalidate`에 존재합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..3 | R_Key | 32 |

### RDETH — 4 bytes (Reliable Datagram ETH)

Reliable Datagram (RD) 전송에서 BTH와 DETH/RETH/etc 사이에 사용됩니다. EE (End-to-End) 컨텍스트 번호를 운반합니다. 실제로는 드뭅니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0 | Reserved | 8 |
| 1..3 | EE Context | 24 |

### XRCETH — 4 bytes (Extended Reliable Connection ETH)

XRC 전송에서 수신자 측 SRQ를 식별하는 데 사용됩니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..3 | XRC SRQ | 32 |

## MAD Common Header — 24 bytes

모든 MAD 메시지는 관리 클래스에 관계없이 이 24바이트 공통 헤더로 시작합니다. MAD 페이로드가 그 다음에 옵니다 — SMP의 경우 전체 MAD 길이는 256바이트로 고정됩니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0 | BaseVersion | 8 |
| 1 | MgmtClass | 8 |
| 2 | ClassVersion | 8 |
| 3 | Method | 8 |
| 4..5 | Status | 16 |
| 6..7 | ClassSpecific | 16 |
| 8..15 | TID (Transaction ID) | 64 |
| 16..17 | AttributeID | 16 |
| 18..19 | Reserved | 16 |
| 20..23 | AttributeModifier | 32 |
| 24..255 | MAD data payload | 232 bytes (SMP) |

이 데이터셋에서 보이는 흔한 MgmtClass 값:

| 값 | 클래스 | 사용처 |
| ---: | --- | --- |
| `0x01` | SMP (LID-routed) | LID 기반 Subnet Management |
| `0x03` | SubnAdm (SA) | Path 레코드, MC 멤버 레코드 |
| `0x04` | Performance Management | `PortCounters`, `PortCountersExtended`, `ClassPortInfo` |
| `0x32` | Vendor-specific OUI | `ibping` |
| `0x81` | SMP (Directed Route) | 초기 fabric 디스커버리 (`ib_initial_sniffer.pcap`) |

흔한 Method 값:

| 값 | Method | 비고 |
| ---: | --- | --- |
| `0x01` | Get | 속성 읽기 |
| `0x02` | Set | 속성 쓰기 |
| `0x03` | Send | Unsolicited |
| `0x05` | Trap | 비동기 통지 |
| `0x06` | Report | SA report |
| `0x07` | TrapRepress | 반복 trap 억제 |
| `0x12` | GetTable | SA 테이블 질의 |
| `0x13` | GetTraceTable | SA trace |
| `0x15` | GetMulti | SA multipart |
| `0x81` | GetResp | Get에 대한 응답 |
| `0x86` | ReportResp | Report에 대한 응답 |

구체적 예시: `infiniband.pcap` frame 1 = `MgmtClass=0x81 (Directed-route SMP), Method=0x01 (Get), AttributeID=0x0020 (SMInfo)`. 이는 `SubnGet(SMInfo)` 패킷입니다.

## SMP Directed Route Extension

`MgmtClass = 0x81`일 때, MAD 공통 헤더 다음에 directed-route 경로를 운반하는 추가 필드가 따릅니다. Wireshark dissector가 노출하고 이 데이터셋에서 보이는 필드들:

| 필드 | 폭 | 의미 |
| --- | ---: | --- |
| D (Direction Bit) | 1 (SMP의 status word 최상위 비트) | 0 = outbound, 1 = inbound |
| Hop Pointer | 8 | 경로의 현재 위치 |
| Hop Count | 8 | 경로의 총 hop 수 |
| M_Key | 64 | 관리 보호 키 |
| DrSLID | 16 | Directed-route source LID; `0xffff` = "use path" |
| DrDLID | 16 | Directed-route destination LID; `0xffff` = "use path" |

이 외에도 SMP MAD body는 포트 번호의 `InitialPath[64]`와 `ReturnPath[64]` 바이트 배열을 운반하지만, 이는 공통 SMP-DR 헤더 필드라기보다는 페이로드 필드입니다.

구체적 예시: `infiniband.pcap` frame 1은 `D=0, Hop Pointer=1, Hop Count=2, M_Key=0, DrSLID=0xffff, DrDLID=0xffff`를 가집니다 — 전형적인 second-hop 디스커버리 프로브.

## IPoIB Encapsulation — 4 bytes (RFC 4391)

IPoIB가 UD QP 위에서 IP 패킷을 운반할 때, BTH/DETH와 IP 계층 사이에 작은 헤더가 위치합니다.

| Byte(s) | 필드 | 폭 |
| ---: | --- | ---: |
| 0..1 | EtherType | 16 |
| 2..3 | Reserved (반드시 0) | 16 |

보이는 EtherType 값:

| 값 | 의미 |
| --- | --- |
| `0x0800` | IPv4 |
| `0x0806` | InfiniBand ARP (RFC 4391) |
| `0x86DD` | IPv6 |

IPoIB ARP는 EtherType에도 불구하고 Ethernet ARP와 같지 않습니다. RFC 4391은 20바이트 하드웨어 주소를 정의합니다: `QPN (24 bits) + Reserved (8 bits) + GID (128 bits)`. 그래서 `ib_ipping_sniffer.pcap`은 익숙해 보이지만 안에 IB 특유의 주소를 운반하는 ARP 레코드를 보여줍니다.

구체적 예시: `infiniband.pcap` frame 10은 `EtherType=0x0800, Reserved=0x0000`을 가지며 그 다음에 IPv4 ICMP Echo request가 옵니다. 메인 보고서의 [단계별 예제](README_KO.md#erf-캡처-해부)를 참고하세요.

## CRC 커버리지

InfiniBand는 패킷 수준에서 두 CRC를 정의합니다:

| CRC | 폭 | 계산 대상 | 목적 |
| --- | ---: | --- | --- |
| ICRC (Invariant CRC) | 32 | 가변 필드(variant header 비트)를 제외한 모든 부분 | end-to-end 무결성, 스위치를 거쳐도 불변 |
| VCRC (Variant CRC) | 16 | 링크상 전체 패킷 | per-link 무결성, 스위치가 재계산 |

ICRC에서 제외되는 가변 필드:

- `LRH.VL` — 스위치가 virtual lane을 remap할 수 있음
- `LRH.SL`/reserved 비트 — 스위치가 reserved 필드를 리셋할 수 있음
- `GRH.HopLmt` — 라우터가 감소시킴
- `GRH.TClass` — remarked 가능
- `GRH.FlowLabel` — remarked 가능
- `BTH.FECN` / `BECN` — 혼잡 통보 지점에서 설정
- `BTH` reserved variant 비트

ERF 캡처에서는 두 CRC 모두 필터 가능한 필드(`infiniband.invariant.crc`와 `infiniband.variant.crc`)로 노출됩니다. Wireshark dissector가 직접 검증하지는 않으므로, 검증 상태를 볼 때는 `erf.flags.rxe`를 참고하세요. 메인 보고서의 [보존 매트릭스](README_KO.md#erf가-보존하는-것-vs-숨기는-것)를 참고하세요.

## BTH Opcode 마스터 표

8비트 OpCode는 분할됩니다: 상위 3비트가 전송 서비스, 하위 5비트가 동작을 식별합니다.

전송 서비스 prefix:

| 비트 7..5 | 서비스 | 범위 |
| ---: | --- | --- |
| `000` | RC (Reliable Connection) | `0x00–0x1F` |
| `001` | UC (Unreliable Connection) | `0x20–0x3F` |
| `010` | RD (Reliable Datagram) | `0x40–0x5F` |
| `011` | UD (Unreliable Datagram) | `0x60–0x7F` |
| `100` | CNP (Congestion Notification, RoCEv2 전용) | `0x80–0x9F` |
| `101` | XRC (Extended Reliable Connection) | `0xA0–0xBF` |

동작 suffix (5비트, 각 전송의 범위 내에서 적용; 모든 suffix가 모든 서비스에 유효한 것은 아님):

| Suffix | 동작 |
| ---: | --- |
| `0x00` | SEND First |
| `0x01` | SEND Middle |
| `0x02` | SEND Last |
| `0x03` | SEND Last with Immediate |
| `0x04` | SEND Only |
| `0x05` | SEND Only with Immediate |
| `0x06` | RDMA WRITE First |
| `0x07` | RDMA WRITE Middle |
| `0x08` | RDMA WRITE Last |
| `0x09` | RDMA WRITE Last with Immediate |
| `0x0A` | RDMA WRITE Only |
| `0x0B` | RDMA WRITE Only with Immediate |
| `0x0C` | RDMA READ Request |
| `0x0D` | RDMA READ Response First |
| `0x0E` | RDMA READ Response Middle |
| `0x0F` | RDMA READ Response Last |
| `0x10` | RDMA READ Response Only |
| `0x11` | Acknowledge |
| `0x12` | ATOMIC Acknowledge |
| `0x13` | Compare & Swap |
| `0x14` | Fetch & Add |
| `0x16` | SEND Last with Invalidate |
| `0x17` | SEND Only with Invalidate |

전송 서비스별 실전 동작 지원:

| 서비스 | 지원 동작 |
| --- | --- |
| RC | 위의 모든 동작 |
| UC | SEND, RDMA WRITE만 (READ 없음, Atomic 없음, ACK 없음) |
| RD | XRC 전용을 제외한 모든 동작 |
| UD | SEND Only, SEND Only with Immediate (RDMA 없음, Atomic 없음, ACK 없음) |
| XRC | RC 동작에 BTH와 동작의 일반 extended header 사이에 XRCETH 추가 |

이 데이터셋에서 보이는 구체적 예시:

| 십진수 | 16진수 | 의미 | 위치 |
| ---: | --- | --- | --- |
| `4` | `0x04` | RC SEND Only | `infiniband.pcap` frame 10 |
| `17` | `0x11` | RC Acknowledge | `infiniband.pcap` frame 11 |
| `100` | `0x64` | UD SEND Only | 이 데이터셋의 모든 MAD 운반 패킷 |

## Operation → Extended Header 매핑

BTH 다음에 어떤 확장 헤더가 나타나는지는 opcode가 결정합니다. 헥스 덤프를 읽다가 "다음에 무엇이 와야 하지?"를 확인할 때 이 표를 사용하세요.

| 동작 | Extended header (BTH 다음 순서대로) |
| --- | --- |
| RC/UC SEND First/Middle | (없음) |
| RC/UC SEND Last/Only | (없음) |
| RC/UC SEND Last/Only with Immediate | ImmDt |
| RC SEND Last/Only with Invalidate | IETH |
| RC/UC RDMA WRITE First | RETH |
| RC/UC RDMA WRITE Middle/Last | (없음) |
| RC/UC RDMA WRITE Last with Immediate | ImmDt |
| RC/UC RDMA WRITE Only | RETH |
| RC/UC RDMA WRITE Only with Immediate | RETH + ImmDt |
| RC RDMA READ Request | RETH |
| RC RDMA READ Response First/Last/Only | AETH |
| RC RDMA READ Response Middle | (없음) |
| RC ACK / NAK | AETH |
| RC CmpSwap / FetchAdd | AtomicETH |
| RC ATOMIC Acknowledge | AETH + AtomicAckETH |
| UD SEND Only | DETH |
| UD SEND Only with Immediate | DETH + ImmDt |
| RD 모든 동작 | RDETH + DETH + (op-specific) |
| XRC 모든 동작 | XRCETH + (RC 동등 extended header) |

이 데이터셋의 관리 트래픽(`MgmtClass` 태그된 MAD를 가진 UD SEND Only)의 레이아웃:

```
LRH → BTH → DETH → MAD common header → MAD payload → ICRC → VCRC
```

`infiniband.pcap` frame 10의 IPoIB ICMP를 운반하는 RC SEND Only:

```
LRH → BTH → IPoIB encap (4B) → IPv4 → ICMP → ICRC → VCRC
```

가상의 RC RDMA READ Request → multi-packet response:

```
Request: LRH → BTH(RDMA READ Request) → RETH → ICRC → VCRC
First:   LRH → BTH(RDMA READ Response First)   → AETH → payload → ICRC → VCRC
Middle:  LRH → BTH(RDMA READ Response Middle)  → payload         → ICRC → VCRC
Last:    LRH → BTH(RDMA READ Response Last)    → AETH → payload → ICRC → VCRC
```

## 참고

- [메인 분석 보고서](README_KO.md) — `ib-packets` 데이터셋에서 관측된 증거
- [ERF 캡처 해부](README_KO.md#erf-캡처-해부) — ERF 외부 레코드가 무엇을 노출하고 무엇을 숨기는지
- [RDMA Read/Write 패킷 분석 모델](README_KO.md#rdma-readwrite-패킷-분석-모델) — 동작 흐름과 검증 규칙
- IBA Architecture Specification Volume 1, Release 1.5
- Wireshark IB dissector 필드 목록: `tshark -G fields | grep -i infiniband`
- [Tencent Cloud: RDMA - IB Specification Volume 1 Transport Layer](https://cloud.tencent.com/developer/article/2513460)
