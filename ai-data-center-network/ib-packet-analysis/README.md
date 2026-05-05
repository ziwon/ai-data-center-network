# InfiniBand Packet Analysis

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Networking-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/networking/)
[![RDMA](https://img.shields.io/badge/RDMA-Zero--Copy-0078D4)](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17)
[![TShark](https://img.shields.io/badge/TShark-CLI%20Analysis-1679A7?logo=wireshark&logoColor=white)](https://www.wireshark.org/docs/man-pages/tshark.html)
[![InfiniBand](https://img.shields.io/badge/InfiniBand-Fabric-111827)](https://www.infinibandta.org/about-infiniband/)
[![IPoIB](https://img.shields.io/badge/IPoIB-IP%20over%20IB-2563EB)](https://docs.nvidia.com/networking/display/ofed/infiniband%2Bnetwork)

This report analyzes the packet captures in the `ib-packets` directory using `tshark`. The goal is to connect the captured packets back to the Chapter 1 RDMA and InfiniBand notes: data path vs control path, InfiniBand packet structure, management traffic, IP over InfiniBand, Queue Pairs, and Reliable Connection behavior. 

<p align="center">
  <img src="ib-packet-analysis.png" alt="ib packet analysis">
</p>

## Table of Contents

- [Executive Summary](#executive-summary)
- [Capture Set](#capture-set)
- [How the Captures Were Analyzed](#how-the-captures-were-analyzed)
- [Inferred Capture Method](#inferred-capture-method)
- [InfiniBand Protocol Stack](#infiniband-protocol-stack)
- [InfiniBand Layers Visible in the Captures](#infiniband-layers-visible-in-the-captures)
- [InfiniBand Packet Structure](#infiniband-packet-structure)
- [Control Path vs Data Path](#control-path-vs-data-path)
- [Per-Capture Findings](#per-capture-findings)
  - [ib_initial_sniffer.pcap](#ib_initial_snifferpcap)
  - [ib_ibping_sniffer.pcap](#ib_ibping_snifferpcap)
  - [ib_ibtracert_sminfo_sniffer.pcap](#ib_ibtracert_sminfo_snifferpcap)
  - [ib_sniffer.pcap](#ib_snifferpcap)
  - [ib_ipping_sniffer.pcap](#ib_ipping_snifferpcap)
  - [ib_IPoIB.pcap](#ib_ipoibpcap)
  - [infiniband.pcap](#infinibandpcap)
- [Key Packet Examples](#key-packet-examples)
- [What These Captures Do Not Show](#what-these-captures-do-not-show)
- [Useful tshark Commands](#useful-tshark-commands)
- [Takeaways for Chapter 1](#takeaways-for-chapter-1)
- [References](#references)

## Executive Summary

The captures can be analyzed with `tshark` without superuser privileges because they are offline `.pcap` files. Root or special capture permissions are usually needed for live packet capture, not for reading existing capture files.

The packet set shows several important InfiniBand behaviors:

- InfiniBand management traffic is visible through MAD packets.
- Subnet Management traffic appears as `SubnGet`, `SubnGetResp`, `SubnSet`, and Subnet Administration records.
- Performance Management traffic appears as `PortCounters`, `PortCountersExtended`, and `ClassPortInfo`.
- IP over InfiniBand (IPoIB) is visible as normal IP, TCP, SSH, ARP, and ICMP traffic carried inside InfiniBand frames.
- One capture shows Reliable Connection (RC) behavior, including `ConnectRequest`, `ConnectReply`, `ReadyToUse`, `RC SEND Only`, and `RC Acknowledge`.

The captures are especially useful for understanding the distinction between:

- **Control path**: setup, discovery, management, path lookup, connection establishment, and performance queries.
- **Data path**: payload movement after the required resources and paths are ready.

Most captures are management or IPoIB examples. They do not show a complete RDMA Read or RDMA Write payload exchange with RETH fields, remote virtual addresses, or rkeys. The closest data-path example is `infiniband.pcap`, which shows RC SEND and AETH ACK behavior.

## Capture Set

| File | Packets | Duration | Main Observation |
| --- | ---: | ---: | --- |
| `ib_initial_sniffer.pcap` | 108 | 10.899693 s | Initial subnet discovery, SMP, SA, multicast membership, and performance queries |
| `ib_ibping_sniffer.pcap` | 65 | 10.181178 s | Vendor MAD request/response behavior plus performance counters |
| `ib_ibtracert_sminfo_sniffer.pcap` | 84 | 30.462705 s | Tracing and SMInfo-related control path traffic |
| `ib_sniffer.pcap` | 24 | 6.002367 s | Performance Management traffic only |
| `ib_ipping_sniffer.pcap` | 34 | 11.999439 s | ICMP ping over IPoIB plus a small amount of ARP and performance traffic |
| `ib_IPoIB.pcap` | 5,848 | 4.284584 s | SSH over TCP over IPoIB |
| `infiniband.pcap` | 43 | 250.573390 s | SMInfo, IPoIB, CM connection setup, RC SEND, and RC ACK behavior |

All files are pcap files with **Extensible Record Format** encapsulation. `tshark` decodes the ERF outer record and then the InfiniBand payload.

## How the Captures Were Analyzed

The analysis used Wireshark/TShark 4.2.2:

```sh
tshark -v
```

Basic capture metadata:

```sh
capinfos ../ib-packets/*.pcap
```

Protocol hierarchy:

```sh
tshark -r ../ib-packets/ib_IPoIB.pcap -q -z io,phs
```

InfiniBand field extraction:

```sh
tshark -r ../ib-packets/ib_initial_sniffer.pcap \
  -Y infiniband \
  -T fields \
  -e frame.number \
  -e frame.time_relative \
  -e infiniband.lrh.dlid \
  -e infiniband.lrh.slid \
  -e infiniband.bth.opcode \
  -e infiniband.bth.destqp \
  -e infiniband.mad.method \
  -e infiniband.mad.attributeid \
  -E header=y
```

Useful fields:

| Field | Meaning |
| --- | --- |
| `infiniband.lrh.dlid` | Destination Local ID from the Local Route Header |
| `infiniband.lrh.slid` | Source Local ID from the Local Route Header |
| `infiniband.bth.opcode` | Base Transport Header opcode |
| `infiniband.bth.destqp` | Destination Queue Pair |
| `infiniband.mad.mgmtclass` | MAD management class |
| `infiniband.mad.method` | MAD method, such as Get or GetResp |
| `infiniband.mad.attributeid` | MAD attribute ID |

## Inferred Capture Method

The exact capture commands cannot be proven from the pcap files alone. The following is an inference from file names, encapsulation type, protocol hierarchy, and decoded packet contents.

The captures are likely the result of running InfiniBand diagnostic or IPoIB workloads while a native InfiniBand sniffer was recording traffic. The files use **Extensible Record Format** encapsulation and expose InfiniBand LRH/BTH/MAD fields, which is more consistent with a native InfiniBand capture path than with a simple Ethernet-style `tcpdump` on an IP interface.

| File | Likely workload during capture | Evidence |
| --- | --- | --- |
| `ib_initial_sniffer.pcap` | Fabric initialization or subnet discovery | `SubnGet(NodeInfo)`, `NodeDescription`, `PortInfo`, `SMInfo`, QP0 traffic |
| `ib_ibping_sniffer.pcap` | `ibping` between two InfiniBand nodes | Repeated vendor MAD request/response traffic between LID 5 and LID 8 |
| `ib_ibtracert_sminfo_sniffer.pcap` | `ibtracert`, `sminfo`, and possibly counter queries | `SMInfo`, `LinearForwardingTable`, `PortCounters`, `PortCountersExtended` |
| `ib_sniffer.pcap` | Performance counter polling | Mostly `PERF (PortCounters)` and `PortCountersExtended` |
| `ib_ipping_sniffer.pcap` | IP ping over IPoIB | ICMP echo request/reply plus ARP over InfiniBand |
| `ib_IPoIB.pcap` | SSH/TCP session over IPoIB | TCP conversation `10.10.10.12:34826 <-> 10.10.10.11:22`, SSH payload |
| `infiniband.pcap` | Mixed InfiniBand sample workload | `SMInfo`, `PathRecord`, `ConnectRequest`, `ConnectReply`, `ReadyToUse`, `RC SEND`, and `RC ACK` |

A plausible collection workflow would have looked like this:

```text
Terminal 1:
  Start a native InfiniBand sniffer and write to a pcap file.

Terminal 2:
  Run one diagnostic or workload command, such as ibping, ibtracert,
  sminfo, perfquery, ping over IPoIB, or SSH over an IPoIB address.

Result:
  The sniffer records LRH/BTH/MAD/IPoIB traffic into a pcap file.
```

For example, the `ib_ibping_sniffer.pcap` name and decoded packets suggest this type of scenario:

```text
Start capture:
  native IB sniffer -> ib_ibping_sniffer.pcap

Run workload:
  ibping between two IB endpoints

Observed packets:
  VENDOR MAD request/response traffic between LIDs
```

The IPoIB captures likely came from ordinary IP tools running over an `ib0`-style interface:

```text
Start capture:
  native IB or IPoIB-aware capture -> ib_ipping_sniffer.pcap

Run workload:
  ping <remote IPoIB address>

Observed packets:
  ARP, ICMP Echo request, ICMP Echo reply over InfiniBand
```

and:

```text
Start capture:
  native IB or IPoIB-aware capture -> ib_IPoIB.pcap

Run workload:
  ssh <remote IPoIB address>

Observed packets:
  TCP handshake and SSH payload over IPoIB
```

If reproducing a similar analysis from existing files, no superuser privileges are required:

```sh
tshark -r ../ib-packets/ib_ibping_sniffer.pcap -c 10
tshark -r ../ib-packets/ib_ipping_sniffer.pcap -c 10
tshark -r ../ib-packets/ib_IPoIB.pcap -q -z conv,tcp
```

If reproducing the capture itself, permissions depend on the capture method. Live capture from a privileged interface or a vendor sniffer may require extra capabilities, group membership, or root privileges. Offline analysis of the resulting pcap does not.

## InfiniBand Protocol Stack

The InfiniBand protocol stack can be viewed at two complementary levels:

- **Protocol stack view**: how applications, upper-layer protocols, transport services, network routing, link behavior, and physical signaling fit together.
- **Packet structure view**: how an individual packet is encoded on the wire, including routing headers, transport headers, optional extended headers, payload, and integrity checks.

The following third-party diagrams are useful orientation material. They are included here as educational figures, while the packet-level interpretation in this report is based on the fields visible through `tshark` and the official NVIDIA, IBTA, and Wireshark references listed below.

![InfiniBand Protocol Stack](infiniband-protocol-stack.webp)

Conceptually, this stack explains why the captures include both **control path** protocols, such as Subnet Management and Connection Management, and **data path** traffic, such as IPoIB and Reliable Connection packets.

![InfiniBand Packet Encapsulation Format](infiniband-packet-encapsulation-format.webp)

The encapsulation figure aligns with the next two sections: `tshark` exposes packet fields such as `LRH`, `BTH`, `DETH`, `MAD`, `AETH`, and IP payloads, depending on the packet type.

Source: [What is InfiniBand? (A Complete Guide)](https://www.naddod.com/blog/what-is-infiniband)

## InfiniBand Layers Visible in the Captures

The packet structure visible in `tshark` maps well to the Chapter 1 InfiniBand Communication Stack.

```mermaid
flowchart TB
    ERF[ERF capture record]
    LRH[InfiniBand LRH<br/>LID routing inside the fabric]
    BTH[InfiniBand BTH<br/>transport opcode, destination QP, PSN]
    EXT[Extended transport headers<br/>DETH / AETH / CM / MAD fields]
    PAYLOAD[Payload<br/>MAD, IPoIB, ICMP, TCP, SSH, or data]

    ERF --> LRH --> BTH --> EXT --> PAYLOAD
```

Important visible headers:

| Header | Role | Example from captures |
| --- | --- | --- |
| LRH | Local routing inside the InfiniBand fabric | `slid`, `dlid`, packet length |
| BTH | Transport behavior and QP selection | opcode `100` for UD SEND Only, opcode `4` for RC SEND Only, opcode `17` for RC ACK |
| DETH | Datagram transport fields for UD traffic | QP0/QP1 management traffic |
| MAD | Management datagram | SubnGet, SubnGetResp, PortCounters |
| AETH | ACK Extended Transport Header | RC Acknowledge packets in `infiniband.pcap` |
| IP payload | IP over InfiniBand | TCP/SSH and ICMP over IPoIB |

## InfiniBand Packet Structure

Building on the encapsulation diagram above, an InfiniBand packet can be read from left to right as **fabric routing**, **transport selection**, **operation-specific metadata**, and **payload**. The exact extended header depends on the transport and opcode.

```mermaid
flowchart LR
    LRH["LRH<br/>Local Route Header<br/>DLID, SLID, VL, packet length"]
    GRH["GRH<br/>Global Route Header<br/>optional, GID-based routing"]
    BTH["BTH<br/>Base Transport Header<br/>opcode, P_Key, destination QP, PSN"]
    EXT["Extended Header<br/>DETH, RETH, AETH, Atomic, Immediate, or none"]
    PAYLOAD["Payload<br/>MAD, IPoIB packet, SEND data, RDMA data"]
    CRC["ICRC / VCRC<br/>integrity checks on the wire"]

    LRH --> GRH --> BTH --> EXT --> PAYLOAD --> CRC
```

`GRH` is optional, so many local-subnet packets are effectively `LRH -> BTH -> ...`. Some capture paths also hide or normalize link-level CRC details, so the CRC fields may be more important as an on-wire concept than as a visible field in every `tshark` decode.

Common packet shapes:

```mermaid
flowchart TB
    MGMT["Management traffic<br/>QP0/QP1 control path"]
    MGMT_SEQ["LRH -> BTH -> DETH -> MAD"]
    IPOIB["IP over InfiniBand<br/>normal IP payload over IB"]
    IPOIB_SEQ["LRH -> BTH -> DETH -> IP -> TCP/ICMP/SSH"]
    RC_SEND["Reliable Connection SEND<br/>message-style data path"]
    RC_SEND_SEQ["LRH -> BTH -> SEND payload"]
    RC_ACK["Reliable Connection ACK<br/>transport acknowledgement"]
    RC_ACK_SEQ["LRH -> BTH -> AETH"]
    RDMA_WRITE["RDMA WRITE<br/>one-sided push, not fully shown here"]
    RDMA_WRITE_SEQ["LRH -> BTH -> RETH -> data payload"]
    RDMA_READ["RDMA READ<br/>one-sided pull, not fully shown here"]
    RDMA_READ_SEQ["Request: LRH -> BTH -> RETH<br/>Response: LRH -> BTH -> AETH -> data payload"]

    MGMT --> MGMT_SEQ
    IPOIB --> IPOIB_SEQ
    RC_SEND --> RC_SEND_SEQ
    RC_ACK --> RC_ACK_SEQ
    RDMA_WRITE --> RDMA_WRITE_SEQ
    RDMA_READ --> RDMA_READ_SEQ
```

How this maps to the current captures:

| Packet family | Typical structure | Visible in this packet set? | Notes |
| --- | --- | --- | --- |
| Subnet Management | `LRH -> BTH -> DETH -> MAD` | Yes | Seen in `ib_initial_sniffer.pcap`, `ib_ibtracert_sminfo_sniffer.pcap`, and `infiniband.pcap` |
| Performance Management | `LRH -> BTH -> DETH -> MAD` | Yes | Seen as `PortCounters`, `PortCountersExtended`, and `ClassPortInfo` |
| IPoIB | `LRH -> BTH -> DETH -> IP payload` | Yes | Carries ICMP, TCP, SSH, and ARP-like behavior over InfiniBand |
| RC SEND | `LRH -> BTH -> payload` | Yes | `infiniband.pcap` shows `RC SEND Only` |
| RC ACK | `LRH -> BTH -> AETH` | Yes | `infiniband.pcap` shows `RC Acknowledge` |
| RDMA WRITE | `LRH -> BTH -> RETH -> payload` | No | This would show remote virtual address and `rkey` in `RETH` |
| RDMA READ | request with `RETH`, response with data | No | This would show the pull model described in Chapter 1 |

## Control Path vs Data Path

The captures make the Chapter 1 distinction between control path and data path concrete.

```mermaid
flowchart LR
    subgraph Control[Control path]
        SM[Subnet Management<br/>NodeInfo, PortInfo, SMInfo]
        SA[Subnet Administration<br/>PathRecord, MCMemberRecord]
        PM[Performance Management<br/>PortCounters]
        CM[Connection Management<br/>ConnectRequest, ConnectReply, ReadyToUse]
    end

    subgraph Data[Data path]
        IPoIB[IP over InfiniBand<br/>ICMP, TCP, SSH]
        RC[Reliable Connection traffic<br/>RC SEND and ACK]
    end

    Control --> Ready[QP/path/resources ready]
    Ready --> Data
```

### Control Path

Control path traffic is strongly represented in these captures. It includes:

- Subnet discovery through QP0.
- Subnet Administration through QP1.
- Path discovery and multicast membership.
- Performance counter queries.
- Connection Management messages.

This corresponds to Chapter 1's explanation that RDMA does not remove the kernel or control software from the system. The CPU, kernel driver, subnet manager, RDMA runtime, and NIC firmware still configure resources and paths.

### Data Path

Data path traffic appears in two forms:

- IPoIB traffic, where normal IP applications run over InfiniBand.
- RC SEND/ACK traffic, where InfiniBand transport behavior is visible below an IP payload.

The packet set does not show full RDMA Write or RDMA Read operations with RETH. Therefore, it is better to describe it as an InfiniBand and IPoIB packet set, not as a complete RDMA Read/Write capture set.

## Per-Capture Findings

### ib_initial_sniffer.pcap

This capture is the best example of InfiniBand control path initialization.

Protocol hierarchy:

```text
erf
  infiniband
    arp
```

Representative packets:

```text
UD Send Only QP=0x000000 SubnGet(NodeInfo)
UD Send Only QP=0x000000 SubnGetResp(NodeInfo)
UD Send Only QP=0x000000 SubnGet(NodeDescription)
UD Send Only QP=0x000000 SubnGetResp(NodeDescription)
UD Send Only QP=0x000000 SubnGet(PortInfo)
UD Send Only QP=0x000000 SubnGetResp(PortInfo)
```

Interpretation:

- QP0 is used for Subnet Management Packets.
- The node is being discovered and configured.
- `NodeInfo`, `NodeDescription`, `PortInfo`, `P_KeyTable`, and `SMInfo` are part of fabric discovery and setup.
- Subnet Administration records such as `MCMemberRecord` and `InformInfo` also appear.
- This is control path traffic, not user payload movement.

Why it matters for Chapter 1:

- It shows the setup work that must happen before the fast path can be used.
- It supports the point that "kernel bypass" does not mean "no setup or control path."
- It maps to PD/QP/MR/path readiness concepts in the RDMA Process section.

### ib_ibping_sniffer.pcap

This capture is centered on `ibping` behavior and vendor MAD messages.

Protocol hierarchy:

```text
erf
  infiniband
```

Representative packets:

```text
LID: 5 -> LID: 8   InfiniBand 290 VENDOR (Unknown Attribute)
LID: 8 -> LID: 5   InfiniBand 290 VENDOR (Unknown Attribute)
```

Top decoded items:

```text
VENDOR (Unknown Attribute)
PERF (PortCounters)
PERF (ClassPortInfo)
PERF (PortCountersExtended)
```

Interpretation:

- `ibping` uses InfiniBand management-style traffic rather than IP ping.
- The capture shows request/response behavior between LID 5 and LID 8.
- The periodic pattern is visible: requests and responses are roughly one second apart.
- Performance Management traffic is also present.

Why it matters for Chapter 1:

- It demonstrates that InfiniBand has its own management and diagnostic traffic independent of TCP/IP.
- LIDs are used directly at the fabric level.

### ib_ibtracert_sminfo_sniffer.pcap

This capture combines trace-related behavior, SMInfo, and performance management.

Protocol hierarchy:

```text
erf
  infiniband
```

Representative packets:

```text
PERF (PortCounters)
PERF (PortCountersExtended)
UD Send Only QP=0x000000 SubnGet(SMInfo)
UD Send Only QP=0x000000 SubnGetResp(SMInfo)
UD Send Only QP=0x000000 SubnGet(LinearForwardingTable)
UD Send Only QP=0x000000 SubnGetResp(LinearForwardingTable)
```

Interpretation:

- `ibtracert` needs fabric topology and forwarding information.
- `SMInfo` identifies subnet manager information.
- `LinearForwardingTable` points to switch forwarding behavior.
- Performance counters show management-plane visibility into port state.

Why it matters for Chapter 1:

- It shows the control plane objects behind the switched fabric model.
- It supports the Packet Relay / Fabric section: switches forward packets, while management traffic discovers and programs fabric behavior.

### ib_sniffer.pcap

This is a small Performance Management capture.

Protocol hierarchy:

```text
erf
  infiniband
```

Top decoded items:

```text
PERF (PortCounters)
PERF (PortCountersExtended)
```

Interpretation:

- The capture is dominated by performance counter queries.
- It is useful for observing monitoring traffic, but it does not show application payloads or RDMA Read/Write operations.

Why it matters for Chapter 1:

- It connects to the monitoring side of the control path.
- Fabric health is not inferred only from data packets. It is also queried through management traffic.

### ib_ipping_sniffer.pcap

This capture shows ICMP over IPoIB.

Protocol hierarchy:

```text
erf
  infiniband
    ip
      icmp
    arp
```

Representative packets:

```text
203.0.113.17 -> 203.0.113.18   ICMP Echo request
203.0.113.18 -> 203.0.113.17   ICMP Echo reply
```

Interpretation:

- This is not `ibping`; it is IP ping carried over InfiniBand.
- It shows normal IP packets mapped onto InfiniBand.
- ARP appears because IPoIB still needs address resolution for IP communication.

Why it matters for Chapter 1:

- It demonstrates the difference between native InfiniBand management tools and IP over InfiniBand.
- It shows how normal IP applications can run above InfiniBand transport.

### ib_IPoIB.pcap

This is the largest capture and shows SSH over TCP over IPoIB.

Protocol hierarchy:

```text
erf
  infiniband
    ip
      tcp
        ssh
```

TCP conversation:

```text
10.10.10.12:34826 <-> 10.10.10.11:22
Total frames: 5848
Total bytes: 10 MB
Duration: 4.2846 s
```

Representative packets:

```text
10.10.10.12 -> 10.10.10.11   TCP 34826 -> 22 [SYN]
10.10.10.11 -> 10.10.10.12   TCP 22 -> 34826 [SYN, ACK]
10.10.10.12 -> 10.10.10.11   SSH Client: Protocol
```

Interpretation:

- This is an IP workload carried over InfiniBand.
- The application is SSH, not RDMA Read/Write.
- `tshark` decodes the upper layers just like ordinary IP traffic once it gets past the InfiniBand encapsulation.
- The high frame count and 10 MB size make this the best sample for IPoIB throughput-style traffic.

Why it matters for Chapter 1:

- It shows that InfiniBand can carry ordinary IP workloads.
- It should not be confused with RDMA semantics. IPoIB is not the same as one-sided RDMA.

### infiniband.pcap

This is the most useful capture for seeing multiple InfiniBand concepts in one place.

Protocol hierarchy:

```text
erf
  infiniband
    ip
      udp
      icmp
    arp
    ipv6
      icmpv6
```

Important decoded items:

```text
UD Send Only QP=0x000000 SubnGet(SMInfo)
UD Send Only QP=0x000000 SubnGetResp(SMInfo)
CM: ConnectRequest
CM: ConnectReply
CM: ReadyToUse
RC Acknowledge
ICMP Echo request
ICMPv6 Neighbor Solicitation / Advertisement
```

Condensed flow:

```text
1-2      SMInfo query and response
3-6      IPoIB UDP/ARP traffic
7-9      Connection Management: request, reply, ready
10-23    RC SEND Only carrying ICMP and RC ACK packets
24-25    More UDP over IPoIB
26-31    IPv6 neighbor discovery and RC ACK
32-33    Subnet Administration PathRecord lookup
34-40    Second CM setup plus ICMPv6 traffic and ACKs
41-42    SMInfo query and response
43       ICMPv6 echo request
```

The most important pair is frame 10 and frame 11:

```text
Frame 10:
  Opcode: Reliable Connection (RC) - SEND Only (4)
  Destination QP: 0xfc0407
  Acknowledge Request: True
  Payload: IPv4 ICMP Echo request

Frame 11:
  Opcode: Reliable Connection (RC) - Acknowledge (17)
  Destination QP: 0x870408
  AETH: Ack
```

Interpretation:

- Connection Management prepares communication.
- RC SEND carries an IP payload.
- RC ACK confirms reliable delivery.
- AETH is visible in the ACK packet.
- This is close to the Chapter 1 discussion of two-sided reliable transport, even though it is not a full RDMA Write or Read example.

Why it matters for Chapter 1:

- It visibly connects QP, BTH opcode, PSN, ACK, and AETH.
- It shows how a ready connection can carry payload and receive transport-level acknowledgments.
- It helps explain why InfiniBand reliability is handled by the HCA/RNIC rather than by the CPU for each packet.

## Key Packet Examples

### Subnet Management over QP0

From `ib_initial_sniffer.pcap`:

```text
UD Send Only QP=0x000000 SubnGet(NodeInfo)
UD Send Only QP=0x000000 SubnGetResp(NodeInfo)
```

Meaning:

- QP0 is used for Subnet Management.
- The traffic is part of fabric discovery and configuration.
- This is control path behavior.

### Performance Management

From `ib_sniffer.pcap` and related captures:

```text
PERF (PortCounters)
PERF (PortCountersExtended)
```

Meaning:

- Fabric components expose counters through management traffic.
- These counters support monitoring and troubleshooting.
- This is not application data movement.

### IPoIB

From `ib_ipping_sniffer.pcap`:

```text
203.0.113.17 -> 203.0.113.18   ICMP Echo request
203.0.113.18 -> 203.0.113.17   ICMP Echo reply
```

From `ib_IPoIB.pcap`:

```text
10.10.10.12:34826 <-> 10.10.10.11:22   TCP/SSH
```

Meaning:

- IP packets can be transported over InfiniBand.
- Upper-layer tools may look familiar, but the L2/L3 underlay is InfiniBand rather than Ethernet.

### Reliable Connection SEND and ACK

From `infiniband.pcap`:

```text
RC SEND Only QP=0xfc0407
RC Acknowledge QP=0x870408
```

Meaning:

- The BTH opcode distinguishes the transport operation.
- The destination QP identifies the queue pair endpoint.
- The ACK includes AETH.
- This shows reliable InfiniBand transport behavior.

## What These Captures Do Not Show

These captures are not full RDMA Read/Write examples.

Missing from the packet set:

- RDMA Write packets with RETH.
- RDMA Read request and response flows.
- Remote virtual address and rkey fields in RETH.
- A complete one-sided RDMA data movement example.
- NCCL collective traffic.

Therefore, the correct interpretation is:

> These captures demonstrate InfiniBand fabric management, IPoIB, and some reliable connection transport behavior. They support the RDMA/IB concepts in Chapter 1, but they do not fully demonstrate RDMA Read or RDMA Write data movement.

## Useful tshark Commands

List basic packet summaries:

```sh
tshark -r ../ib-packets/infiniband.pcap -c 20
```

Show protocol hierarchy:

```sh
tshark -r ../ib-packets/ib_IPoIB.pcap -q -z io,phs
```

Extract LRH, BTH, and MAD fields:

```sh
tshark -r ../ib-packets/ib_initial_sniffer.pcap \
  -Y infiniband \
  -T fields \
  -e frame.number \
  -e frame.time_relative \
  -e infiniband.lrh.slid \
  -e infiniband.lrh.dlid \
  -e infiniband.bth.opcode \
  -e infiniband.bth.destqp \
  -e infiniband.mad.mgmtclass \
  -e infiniband.mad.method \
  -e infiniband.mad.attributeid \
  -E header=y
```

Count BTH opcodes:

```sh
tshark -r ../ib-packets/infiniband.pcap \
  -Y "infiniband.bth.opcode" \
  -T fields \
  -e infiniband.bth.opcode | sort | uniq -c
```

Show a detailed packet decode:

```sh
tshark -r ../ib-packets/infiniband.pcap \
  -Y "frame.number==10 || frame.number==11" \
  -V
```

Find IPoIB TCP conversations:

```sh
tshark -r ../ib-packets/ib_IPoIB.pcap -q -z conv,tcp
```

## Takeaways for Chapter 1

1. **InfiniBand has a visible control path.**
   The captures show Subnet Management, Subnet Administration, Performance Management, and Connection Management traffic. This reinforces the Chapter 1 point that RDMA kernel bypass mainly applies to the data path.

2. **LIDs and QPs are real packet-level identifiers.**
   LRH fields show source and destination LIDs. BTH fields show destination QPs and opcodes. These are not just abstract API concepts.

3. **QP0 and QP1 matter for management.**
   QP0 appears in Subnet Management traffic. QP1 appears in Subnet Administration and Connection Management traffic.

4. **IPoIB is different from one-sided RDMA.**
   IPoIB carries normal IP traffic over InfiniBand. It can show TCP, SSH, ARP, and ICMP, but that does not mean it is showing RDMA Read or RDMA Write.

5. **Reliable Connection behavior is visible.**
   `infiniband.pcap` shows RC SEND and RC ACK packets. The ACK includes AETH, which matches the Chapter 1 discussion of ACK Extended Transport Header behavior.

6. **The data path/control path distinction should stay explicit.**
   The management captures are mostly control path. IPoIB and RC SEND/ACK are closer to data path. Full RDMA Read/Write would require additional captures that include RETH and one-sided operation fields.


## References

### NVIDIA official documentation
- [NVIDIA Introduction to InfiniBand™](https://network.nvidia.com/pdf/whitepapers/IB_Intro_WP_190.pdf)
- [NVIDIA RDMA Aware Networks Programming User Manual: Key Concepts](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/key%2Bconcepts)
- [NVIDIA MLNX_OFED: InfiniBand Network](https://docs.nvidia.com/networking/display/ofed/infiniband%2Bnetwork)
- [NVIDIA MLNX_OFED Documentation v24.04-0.7.0.0](https://docs.nvidia.com/networking/display/mlnxofedv24040700)
- [NVIDIA Quantum InfiniBand Networking Solutions](https://www.nvidia.com/en-us/networking/products/infiniband/)
- [NVIDIA Quantum-X800 InfiniBand Platform](https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/)

### Protocol and tooling references

- [InfiniBand Trade Association: InfiniBand Architecture Specification FAQ](https://www.infinibandta.org/ibta-specification/)
- [InfiniBand Trade Association: About InfiniBand](https://www.infinibandta.org/about-infiniband/)
- [Wireshark Display Filter Reference: InfiniBand](https://www.wireshark.org/docs/dfref/i/infiniband.html)
- [TShark Manual Page](https://www.wireshark.org/docs/man-pages/tshark.html)
- [linux-rdma rdma-core](https://github.com/linux-rdma/rdma-core)
- [OpenFabrics Alliance Overview](https://www.openfabrics.org/ofa-overview/)
- [OpenFabrics Alliance Advanced Network Software](https://www.openfabrics.org/advanced-network-software/)

### Technical articles and background

- [NADDOD Blog: What is InfiniBand?](https://www.naddod.com/blog/what-is-infiniband)
- [O’Reilly InfiniBand Network Architecture](https://learning.oreilly.com/library/view/infiniband-network-architecture/0321117654/)
- [NVIDIA Technical Blog: Simplifying Network Operations for AI with NVIDIA Quantum InfiniBand](https://developer.nvidia.com/blog/simplifying-network-operations-for-ai-with-nvidia-quantum-infiniband/)
- [NVIDIA Technical Blog: InfiniBand Multilayered Security Protects Data Centers and AI Workloads](https://developer.nvidia.com/blog/infiniband-multilayered-security-protects-data-centers-and-ai-workloads/)
- [NVIDIA Technical Blog: Powering the Next Frontier of Networking for AI Platforms with NVIDIA DOCA 3.0](https://developer.nvidia.com/blog/powering-the-next-frontier-of-networking-for-ai-platforms-with-nvidia-doca-3-0/)
- [NVIDIA Technical Blog: New MLPerf Inference Network Division Showcases NVIDIA InfiniBand and GPUDirect RDMA Capabilities](https://developer.nvidia.com/blog/new-mlperf-inference-network-division-showcases-infiniband-and-gpudirect-rdma-capabilities/)

