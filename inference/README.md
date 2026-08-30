# Inference

LLM 추론의 모델 구조와 시스템 최적화 자료를 주제별로 분리해 정리합니다.

## Efficient LLM Inference Systems

성능 지표, 하드웨어, KV cache, 양자화, speculative decoding을 주차별 실습과 함께 다룹니다.

- [Efficient LLM Inference Systems](efficient-llm-inference-systems/README.md)

### 관련 도서

- [Inference Engineering](https://www.baseten.co/inference-engineering/) — 모델 아키텍처, GPU 하드웨어, inference engine, 최적화 기법, production serving을 아우르는 실무 안내서
- **Hands-On LLM Serving and Optimization** (Chi Wang, Peiheng Hu) — KV cache, batching, quantization, speculative decoding, 분산 serving과 vLLM 최적화를 실습 중심으로 다룹니다.
  - [Book](https://orca3.github.io/llm-model-inference/) · [Video](https://www.oreilly.com/library/view/hands-on-llm-serving/9798341621480/) · [Code and Notebooks](https://github.com/orca3/llm-model-inference)

## FlashAttention from First Principles

Dense attention의 수학적 연산은 그대로 유지하면서, online softmax와 tiling으로 중간 score·probability 행렬을 HBM에 materialize하지 않는 원리를 설명합니다. 이어서 GPU 세대가 바뀔 때마다 최적화의 중심이 어떻게 이동했는지 추적합니다.

| 세대 | 주요 병목 | 핵심 대응 |
| --- | --- | --- |
| **FlashAttention-1** | $N^2$ 중간 텐서의 HBM 트래픽 | tiling, kernel fusion, online softmax, backward recomputation |
| **FlashAttention-2** | 부족한 병렬성, non-matmul 연산, warp 간 통신 | query-block 병렬화, 단순화한 상태 갱신, query-row 단위 warp 분할 |
| **FlashAttention-3** | load·Tensor Core·softmax 단계의 직렬 실행 | Hopper TMA/WGMMA, warp specialization, ping-pong pipeline |
| **FlashAttention-4** | Tensor Core 대비 느린 exponential과 shared-memory 경로 | Blackwell TMEM, software-assisted exponential, conditional rescaling, 2-CTA 협력 |

- [FlashAttention from First Principles 전체 글](flashattention-from-first-principles/README.md) ([한국어 번역](flashattention-from-first-principles/README.ko.md))

> [!NOTE]
> FlashAttention은 training과 긴 prompt prefill에서 특히 유용합니다. 한 번에 처리하는 query가 하나 또는 소수인 autoregressive decode에서는 $N\times N$ 행렬을 만들지 않으므로, KV cache bandwidth·paging·batching과 decode 전용 kernel이 더 큰 병목일 수 있습니다.

## Model Architecture and Serving Profiles

모델별 아키텍처, attention과 MoE 설계, 메모리·통신 특성, 자체 호스팅 전략을 다룹니다.

- [Models](models/README.md)
  - [How We Scaled Kimi K2.5 강의 노트](models/kimi-k2-5-scaling.md)
  - [Kimi K3 기술 해부](models/kimi-k3.md)
