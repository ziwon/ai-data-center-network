# AI Data Center Systems

AI data center networking, LLM inference, training, storage, and AI systems performance engineering study notes.

<table align="center">
  <tr>
    <td align="center">
      <p>
        <a href="https://adcs.restack.tech/"><strong>Open the Web Wiki</strong></a>
      </p>
      <p>
        <a href="https://adcs.restack.tech/"><code>adcs.restack.tech</code></a>
      </p>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://adcs.restack.tech/">
    <img width="1024" alt="Animated AI performance fabric" src="./fabric.svg" />
  </a>
</p>

## Core Infrastructure

- [Network](./network/README.md): AI 데이터센터 네트워크, RDMA, InfiniBand, RoCE, Clos fabric ([스터디 홈](https://app.notion.com/p/gasidaseo/AI-Data-Center-Network-Study-34a50aec5edf8097b1d0ec9c499b3913))
- [GPU & Accelerator Systems](./gpu/README.md): GPU architecture, CUDA, profiling, and kernel analysis
- [Storage](./storage/README.md): AI workload storage, ZFS, checkpoint/data pipeline

## AI Workloads

- [Training](./training/README.md): MLPerf Training, distributed training, LLM/MoE/LoRA workload
- [Inference](./inference/README.md): LLM inference 성능, KV cache, batching, GPU profiling

## Cross-Layer Engineering

- [Systems Performance](./systems-performance/README.md): GPU, CUDA, PyTorch 기반 AI 시스템 성능 엔지니어링

## Courses

- [CME295 Lecture Notes](./courses/cme295/README.md): Transformer/LLM 강의 노트
- [Deep Learning for Network Engineers](./courses/deep-learning-for-network-engineers/README.md): Deep learning model, training process, network engineering 기초

## Talks

- [ziwon - Making DGX B200 RDMA-ready](./talks/sr-iov-with-dgx-b200/making-dgx-b200-rdma-ready.pdf): 온라인 스터디 공유용

## Labs

- [Clos Fabric Lab Series](./network/clos-ebgp-lab/README.md)
- [InfiniBand Packet Analysis](./network/ib-packet-analysis/README.md)
- [RDMA Read/Write Examples](./network/rdma-examples/README.md)

## News
- [Ask this docs](https://adcs.restack.tech/): AI Q&A panel with logs in Cloudflare D1 ([admin](https://adcs.restack.tech/admin/qa-logs/))

  <img width="422" height="320" alt="adcs-ask-docs" src="https://github.com/user-attachments/assets/0bf83a9d-ef55-40f1-8016-3b178fb15def" />

- [Knowledge graph](https://adcs.restack.tech/): per-page and global document-concept graphs with multilingual (KO/EN) semantic links (Cloudflare Workers AI `@cf/baai/bge-m3`), rendered with d3-force ([3D view](https://adcs.restack.tech/knowledge-graph-3d/))
  <img width="640" height="373" alt="adcs-3d-graph" src="https://github.com/user-attachments/assets/20c7850e-b013-473f-b87a-ab049fb959c7" />
