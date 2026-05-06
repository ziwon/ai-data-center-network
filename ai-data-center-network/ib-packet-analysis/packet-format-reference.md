# InfiniBand Packet Format Reference

**English** | [한국어](packet-format-reference_KO.md)

This document is a bit-level reference for InfiniBand packet headers, intended as a companion to the [main analysis report](README.md). The report focuses on what was observed in the `ib-packets` dataset; this reference focuses on what every IB packet header looks like on the wire, with anchor links back to the report's frame-level evidence wherever a concrete example exists.

Use this reference when:

- Reading a hex dump and identifying field boundaries
- Validating which extended header should appear after a given BTH opcode
- Decoding an AETH syndrome value
- Looking up a BTH opcode across all transport services

Source material: IBA Architecture Specification Volume 1 (Release 1.5), Wireshark InfiniBand dissector field list, and the [Tencent Cloud transport-layer article](https://cloud.tencent.com/developer/article/2513460).

## Table of Contents

- [Header Sequence Overview](#header-sequence-overview)
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
- [CRC Coverage](#crc-coverage)
- [BTH Opcode Master Table](#bth-opcode-master-table)
- [Operation → Extended Header Mapping](#operation--extended-header-mapping)
- [See Also](#see-also)

## Header Sequence Overview

An IB packet on the wire is a strict concatenation of headers, payload, and CRCs. Which extended header appears, and in what order, is fully determined by `LRH.LNH` (presence of GRH) and the BTH opcode (which extended headers follow).

```
+-----+-------+-----+-----------------------+----------+------+------+
| LRH |  GRH? | BTH | Extended Header(s)    | Payload  | ICRC | VCRC |
+-----+-------+-----+-----------------------+----------+------+------+
   8     40    12      0..28+ bytes            variable    4      2
```

- `GRH` is present only when `LRH.LNH = 0x3`.
- The set and order of extended headers follow the [Operation → Extended Header Mapping](#operation--extended-header-mapping).
- `ICRC` and `VCRC` always close the packet on the wire.

## Local Route Header (LRH) — 8 bytes

The LRH is the first IB header on every packet, used for fabric-local routing.

Bit layout (big-endian):

| Byte | Bit pattern | Field | Width |
| ---: | --- | --- | ---: |
| 0 | `VVVV LLLL` | VL[3:0] / LVer[3:0] | 4 + 4 |
| 1 | `SSSS RR NN` | SL[3:0] / Reserved / LNH[1:0] | 4 + 2 + 2 |
| 2..3 | `DDDDDDDD DDDDDDDD` | DLID | 16 |
| 4 | `RRRRR PPP` | Reserved / PktLen[10:8] | 5 + 3 |
| 5 | `PPPPPPPP` | PktLen[7:0] | 8 |
| 6..7 | `SSSSSSSS SSSSSSSS` | SLID | 16 |

Field meanings:

| Field | Description |
| --- | --- |
| VL | Virtual Lane (0–15). VL15 is reserved for management traffic |
| LVer | Link version, currently always 0 |
| SL | Service Level (0–15), maps to QoS class |
| LNH | Link Next Header — selects what follows the LRH |
| DLID / SLID | Destination / Source Local IDs (assigned by the SM) |
| PktLen | Packet length in 4-byte words, excluding the LRH and VCRC |

LNH encoding:

| Value | Meaning | What follows the LRH |
| ---: | --- | --- |
| `0x0` | Raw IPv6 (legacy) | IPv6 header directly |
| `0x1` | Raw IPv4 (legacy) | IPv4 header directly |
| `0x2` | IBA Local | BTH (no GRH) |
| `0x3` | IBA Global | GRH + BTH |

Concrete example: every packet in this dataset carries `LNH = 0x2`, which is why no GRH is decoded. See the worked example for `infiniband.pcap` frame 10 in the main report's [ERF Capture Anatomy](README.md#erf-capture-anatomy) section.

## Global Route Header (GRH) — 40 bytes

The GRH appears only when `LRH.LNH = 0x3`, signaling routing across IB subnets. The format mirrors IPv6.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0 (high 4) | IPVer | 4 |
| 0 (low 4) + 1 (high 4) | TClass | 8 |
| 1 (low 4) + 2..3 | FlowLabel | 20 |
| 4..5 | PayLen | 16 |
| 6 | NxtHdr | 8 |
| 7 | HopLmt | 8 |
| 8..23 | SGID | 128 |
| 24..39 | DGID | 128 |

Notes:

- `IPVer` is always 6.
- `NxtHdr = 0x1B` (27 decimal) signals an IBA next header (BTH).
- `PayLen` counts bytes after the GRH up to the start of ICRC.
- SGID/DGID are 128-bit GIDs assigned by the SM.

This dataset does not contain any GRH-bearing packets, so this section is purely a reference for future cross-subnet captures.

## Base Transport Header (BTH) — 12 bytes

BTH selects the transport operation, the destination QP, and the packet sequence number. It appears on every IBA packet (i.e., when `LNH ∈ {0x2, 0x3}`).

Bit layout:

| Byte(s) | Field | Width |
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

Field meanings:

| Field | Notes |
| --- | --- |
| OpCode | High 3 bits = transport service, low 5 bits = operation. See the [BTH Opcode Master Table](#bth-opcode-master-table) |
| SE | Solicited Event — set on the last packet of a SEND or RDMA WRITE message that should trigger a CQ event on the responder |
| M | Used during automatic path migration to signal request / accept |
| PadCnt | 0–3 bytes added at the end of the payload to align to 4-byte boundaries |
| TVer | Transport header version, currently always 0 |
| P_Key | Partition key; high bit = full vs limited membership, low 15 bits = partition ID |
| FECN / BECN | Forward / Backward Explicit Congestion Notification |
| DestQP | 24-bit destination Queue Pair number |
| AckReq | When set on RC traffic, the responder must generate an ACK |
| PSN | 24-bit Packet Sequence Number; wraps modulo 2²⁴ |

PSN behaviors worth knowing:

- The expected PSN is tracked per QP. A packet whose PSN equals the expected value advances the window.
- A PSN within the *duplicate range* (older than expected, but within 2²³) is treated as a retransmission and acknowledged without delivering payload again.
- A PSN beyond the duplicate range but earlier than expected is a sequence error and triggers a NAK with code 0.

Concrete example: `infiniband.pcap` frame 10 BTH:

```
Opcode  = 4   (RC SEND Only)
SE = 0, M = 1, PadCnt = 0, TVer = 0
P_Key   = 0xffff   (full membership, default partition)
FECN = 0, BECN = 0
DestQP  = <masked>
AckReq  = 1
PSN     = 13896277
```

## Extended Transport Headers

Which extended header(s) follow the BTH is fully determined by the BTH opcode. The IBA spec encodes this as a per-opcode table; the per-operation summary is in [Operation → Extended Header Mapping](#operation--extended-header-mapping).

### DETH — 8 bytes (Datagram ETH)

Required for UD and RD operations. Also used by all MAD traffic over QP0/QP1.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..3 | Q_Key | 32 |
| 4 | Reserved | 8 |
| 5..7 | SrcQP | 24 |

Q_Key conventions:

- QP0 (SMP): `Q_Key = 0`
- QP1 (GMP): `Q_Key = 0x80010000`
- Other UD QPs: application-defined; high bit set = privileged

Concrete example: `infiniband.pcap` frame 1 SMP traffic uses `DestQP = 0x000000`, `SrcQP = 0x00000000`, `Q_Key = 0x00000000`.

### RETH — 16 bytes (RDMA ETH)

Present in `RDMA READ Request`, `RDMA WRITE First`, `RDMA WRITE Only`, and the with-Immediate variants.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..7 | VA (Virtual Address) | 64 |
| 8..11 | R_Key | 32 |
| 12..15 | DMALen | 32 |

The responder must validate the request against the registered MR for `R_Key`: VA must lie within the MR's address range, `[VA, VA + DMALen)` must be within bounds, and the access permissions of the MR must include READ or WRITE as needed.

This dataset contains no RETH-bearing packets.

### AETH — 4 bytes (ACK ETH)

Present in RC and RD ACK packets and in the first/last/only response packets of an RDMA READ.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0 | Syndrome | 8 |
| 1..3 | MSN (Message Sequence Number) | 24 |

Syndrome encoding (8 bits):

| Bit | Field | Width |
| ---: | --- | ---: |
| 7 | Reserved | 1 |
| 6..5 | OpCode | 2 |
| 4..0 | Value | 5 |

OpCode values:

| OpCode | Meaning | Value field interpretation |
| --- | --- | --- |
| `00` | ACK | Credit Count (0–30); `31` = no credit information supplied |
| `01` | RNR NAK | RNR Timer (selects retry delay from a fixed table; see IBA §9.7.5.2.8) |
| `10` | Reserved | — |
| `11` | NAK | NAK code: `0`=PSN seq error, `1`=invalid request, `2`=remote access error, `3`=remote operation error, `4`=invalid RD request |

Concrete example: `infiniband.pcap` frame 11 has `Syndrome = 31` decimal = `0x1F` = `0 00 11111`. This decodes to `OpCode = 00 (ACK), Value = 11111 (no credit info)` — a normal acknowledgment with no flow-control hint. See the [main report's frame-11 mapping](README.md#infinibandpcap) for context.

### AtomicETH — 28 bytes

Present in RC `CmpSwap` and `FetchAdd` request packets.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..7 | VA | 64 |
| 8..11 | R_Key | 32 |
| 12..19 | Swap Data (CmpSwap) / Add Data (FetchAdd) | 64 |
| 20..27 | Compare Data (CmpSwap) / Reserved (FetchAdd) | 64 |

Atomic operations are guaranteed at-most-once; retried requests are matched against a per-QP outstanding-atomic queue and replayed without re-executing the read-modify-write.

This dataset contains no atomic operations.

### AtomicAckETH — 8 bytes

Carries the original (pre-atomic) value back to the requester. Sits after AETH on `ATOMIC Acknowledge` packets.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..7 | Original Remote Data | 64 |

### ImmDt — 4 bytes (Immediate Data)

Carries 32 bits of opaque data delivered to the receiver's CQE. Present on opcodes whose name ends in "with Immediate". Always sits last among extended headers (after RETH if a `RDMA WRITE Only/Last with Immediate`).

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..3 | Immediate Data | 32 |

### IETH — 4 bytes (Invalidate ETH)

Carries an R_Key to be invalidated on the responder. Present on `SEND Last with Invalidate` and `SEND Only with Invalidate`.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..3 | R_Key | 32 |

### RDETH — 4 bytes (Reliable Datagram ETH)

Used by Reliable Datagram (RD) transport between BTH and DETH/RETH/etc. Carries an EE (End-to-End) context number. Rare in practice.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0 | Reserved | 8 |
| 1..3 | EE Context | 24 |

### XRCETH — 4 bytes (Extended Reliable Connection ETH)

Used by XRC transport to identify the SRQ on the receiver.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..3 | XRC SRQ | 32 |

## MAD Common Header — 24 bytes

Every MAD message begins with this 24-byte common header, regardless of management class. The MAD payload follows; for SMPs the total MAD length is fixed at 256 bytes.

| Byte(s) | Field | Width |
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
| 24..255 | MAD data payload | 232 bytes (SMPs) |

Common MgmtClass values seen in this dataset:

| Value | Class | Used by |
| ---: | --- | --- |
| `0x01` | SMP (LID-routed) | LID-routed Subnet Management |
| `0x03` | SubnAdm (SA) | Path records, MC member records |
| `0x04` | Performance Management | `PortCounters`, `PortCountersExtended`, `ClassPortInfo` |
| `0x32` | Vendor-specific OUI | `ibping` |
| `0x81` | SMP (Directed Route) | Initial fabric discovery (`ib_initial_sniffer.pcap`) |

Common Method values:

| Value | Method | Notes |
| ---: | --- | --- |
| `0x01` | Get | Read attribute |
| `0x02` | Set | Write attribute |
| `0x03` | Send | Unsolicited |
| `0x05` | Trap | Asynchronous notification |
| `0x06` | Report | SA report |
| `0x07` | TrapRepress | Suppress repeating traps |
| `0x12` | GetTable | SA table query |
| `0x13` | GetTraceTable | SA trace |
| `0x15` | GetMulti | SA multipart |
| `0x81` | GetResp | Response to Get |
| `0x86` | ReportResp | Response to Report |

Concrete example: `infiniband.pcap` frame 1 = `MgmtClass=0x81 (Directed-route SMP), Method=0x01 (Get), AttributeID=0x0020 (SMInfo)`. This is a `SubnGet(SMInfo)` packet.

## SMP Directed Route Extension

When `MgmtClass = 0x81`, additional fields follow the MAD common header to carry the directed-route path. The fields exposed by the Wireshark dissector and visible in this dataset:

| Field | Width | Meaning |
| --- | ---: | --- |
| D (Direction Bit) | 1 (top bit of the SMP's status word) | 0 = outbound, 1 = inbound |
| Hop Pointer | 8 | Current position in the path |
| Hop Count | 8 | Total hops in the path |
| M_Key | 64 | Management protection key |
| DrSLID | 16 | Directed-route source LID; `0xffff` = "use path" |
| DrDLID | 16 | Directed-route destination LID; `0xffff` = "use path" |

Beyond these, the SMP MAD body also carries `InitialPath[64]` and `ReturnPath[64]` byte arrays of port numbers, but those are payload fields rather than common SMP-DR header fields.

Concrete example: `infiniband.pcap` frame 1 has `D=0, Hop Pointer=1, Hop Count=2, M_Key=0, DrSLID=0xffff, DrDLID=0xffff` — a typical second-hop discovery probe.

## IPoIB Encapsulation — 4 bytes (RFC 4391)

When IPoIB carries an IP packet over a UD QP, a small header sits between the BTH/DETH and the IP layer.

| Byte(s) | Field | Width |
| ---: | --- | ---: |
| 0..1 | EtherType | 16 |
| 2..3 | Reserved (must be 0) | 16 |

EtherType values seen:

| Value | Meaning |
| --- | --- |
| `0x0800` | IPv4 |
| `0x0806` | InfiniBand ARP (RFC 4391) |
| `0x86DD` | IPv6 |

IPoIB ARP, despite the EtherType, is not the same as Ethernet ARP. RFC 4391 defines a 20-byte hardware address: `QPN (24 bits) + Reserved (8 bits) + GID (128 bits)`. This is why `ib_ipping_sniffer.pcap` shows ARP records that look familiar but carry IB-specific addressing inside.

Concrete example: `infiniband.pcap` frame 10 has `EtherType=0x0800, Reserved=0x0000`, followed directly by an IPv4 ICMP Echo request. See the main report's [worked example](README.md#erf-capture-anatomy).

## CRC Coverage

InfiniBand defines two CRCs at packet level:

| CRC | Width | Computed over | Purpose |
| --- | ---: | --- | --- |
| ICRC (Invariant CRC) | 32 | Everything except mutable fields (variant header bits) | End-to-end integrity, immutable across switches |
| VCRC (Variant CRC) | 16 | Entire packet on the link | Per-link integrity, recomputed by switches |

Mutable fields excluded from ICRC include:

- `LRH.VL` — switches may remap virtual lanes
- `LRH.SL`/reserved bits — switches may reset reserved fields
- `GRH.HopLmt` — decremented by routers
- `GRH.TClass` — may be remarked
- `GRH.FlowLabel` — may be remarked
- `BTH.FECN` / `BECN` — set by congestion-notification points
- `BTH` reserved variant bits

In the ERF captures both CRCs are exposed as filterable fields (`infiniband.invariant.crc` and `infiniband.variant.crc`). The Wireshark dissector does not validate them; trust `erf.flags.rxe` instead. See the main report's [preservation matrix](README.md#what-erf-preserves-vs-hides) for details.

## BTH Opcode Master Table

The 8-bit OpCode is partitioned: top 3 bits identify the transport service, bottom 5 bits the operation.

Transport-service prefix:

| Bits 7..5 | Service | Range |
| ---: | --- | --- |
| `000` | RC (Reliable Connection) | `0x00–0x1F` |
| `001` | UC (Unreliable Connection) | `0x20–0x3F` |
| `010` | RD (Reliable Datagram) | `0x40–0x5F` |
| `011` | UD (Unreliable Datagram) | `0x60–0x7F` |
| `100` | CNP (Congestion Notification, RoCEv2 only) | `0x80–0x9F` |
| `101` | XRC (Extended Reliable Connection) | `0xA0–0xBF` |

Operation suffix (5-bit, applies within each transport's range; not all suffixes valid for every service):

| Suffix | Operation |
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

Practical operation support per transport service:

| Service | Supported operations |
| --- | --- |
| RC | All of the above |
| UC | SEND, RDMA WRITE only (no READ, no Atomic, no ACK) |
| RD | All except XRC-specific |
| UD | SEND Only, SEND Only with Immediate (no RDMA, no Atomic, no ACK) |
| XRC | RC operations with the addition of an XRCETH between BTH and the operation's normal extended headers |

Concrete examples seen in this dataset:

| Decimal | Hex | Meaning | Where |
| ---: | --- | --- | --- |
| `4` | `0x04` | RC SEND Only | `infiniband.pcap` frame 10 |
| `17` | `0x11` | RC Acknowledge | `infiniband.pcap` frame 11 |
| `100` | `0x64` | UD SEND Only | All MAD-bearing packets across this dataset |

## Operation → Extended Header Mapping

Which extended header(s) appear after the BTH is determined by the opcode. Use this table when reading a hex dump and asking "what comes next?"

| Operation | Extended headers (in order, after BTH) |
| --- | --- |
| RC/UC SEND First/Middle | (none) |
| RC/UC SEND Last/Only | (none) |
| RC/UC SEND Last/Only with Immediate | ImmDt |
| RC SEND Last/Only with Invalidate | IETH |
| RC/UC RDMA WRITE First | RETH |
| RC/UC RDMA WRITE Middle/Last | (none) |
| RC/UC RDMA WRITE Last with Immediate | ImmDt |
| RC/UC RDMA WRITE Only | RETH |
| RC/UC RDMA WRITE Only with Immediate | RETH + ImmDt |
| RC RDMA READ Request | RETH |
| RC RDMA READ Response First/Last/Only | AETH |
| RC RDMA READ Response Middle | (none) |
| RC ACK / NAK | AETH |
| RC CmpSwap / FetchAdd | AtomicETH |
| RC ATOMIC Acknowledge | AETH + AtomicAckETH |
| UD SEND Only | DETH |
| UD SEND Only with Immediate | DETH + ImmDt |
| RD any operation | RDETH + DETH + (op-specific) |
| XRC any operation | XRCETH + (RC-equivalent extended headers) |

For the management traffic in this dataset (UD SEND Only with `MgmtClass`-tagged MAD), the layout is:

```
LRH → BTH → DETH → MAD common header → MAD payload → ICRC → VCRC
```

For the RC SEND Only carrying IPoIB ICMP in `infiniband.pcap` frame 10:

```
LRH → BTH → IPoIB encap (4B) → IPv4 → ICMP → ICRC → VCRC
```

For a hypothetical RC RDMA READ Request → multi-packet response:

```
Request: LRH → BTH(RDMA READ Request) → RETH → ICRC → VCRC
First:   LRH → BTH(RDMA READ Response First)   → AETH → payload → ICRC → VCRC
Middle:  LRH → BTH(RDMA READ Response Middle)  → payload         → ICRC → VCRC
Last:    LRH → BTH(RDMA READ Response Last)    → AETH → payload → ICRC → VCRC
```

## See Also

- [Main analysis report](README.md) — observed evidence from the `ib-packets` dataset
- [ERF Capture Anatomy](README.md#erf-capture-anatomy) — what the ERF outer record exposes vs hides
- [RDMA Read/Write Packet Analysis Model](README.md#rdma-readwrite-packet-analysis-model) — operation flows and validation rules
- IBA Architecture Specification Volume 1, Release 1.5
- Wireshark IB dissector field list: `tshark -G fields | grep -i infiniband`
- [Tencent Cloud: RDMA - IB Specification Volume 1 Transport Layer](https://cloud.tencent.com/developer/article/2513460)
