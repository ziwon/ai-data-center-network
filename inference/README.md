# Inference

LLM 추론의 모델 구조와 시스템 최적화 자료를 주제별로 분리해 정리합니다.

## Layout

| 주제 | 성격 | 다루는 범위 |
|---|---|---|
| [Efficient LLM Inference Systems](efficient-llm-inference-systems/README.md) | Course — **primary spine** | 성능 지표, 하드웨어, KV cache, 양자화, speculative decoding을 주차별 실습과 함께 |
| [FlashAttention from First Principles](flashattention-from-first-principles/README.md) | Deep dive | online softmax와 tiling, FA-1~4의 세대별 병목 이동 ([한국어](flashattention-from-first-principles/README.ko.md)) |
| [SGLang in 2026](sglang-production-practices/README.md) | Case study | RadixAttention, Model Gateway, HiCache, prefill-decode 분리, production rollout (**영문**) |
| [Models](models/README.md) | Reference | 모델별 아키텍처와 serving profile — [Kimi K2.5](models/kimi-k2-5-scaling.md), [Kimi K3](models/kimi-k3.md) |

## Suggested path

1. **기초:** [Efficient LLM Inference Systems](efficient-llm-inference-systems/README.md)로 지표·하드웨어·KV cache를 먼저 잡습니다.
2. **Kernel 레벨:** [FlashAttention](flashattention-from-first-principles/README.md)으로 중간 행렬을 HBM에 materialize하지 않는 원리와, GPU 세대마다 최적화의 중심이 어디로 옮겨갔는지 봅니다.
3. **시스템 레벨:** [SGLang](sglang-production-practices/README.md)에서 개별 kernel의 속도가 아니라 SLO를 만족하는 goodput 관점으로 올라갑니다.
4. **모델별 확인:** [Models](models/README.md)에서 대상 모델의 메모리·통신 특성을 조회합니다.

## Resources

### Books

- [Inference Engineering](https://www.baseten.co/inference-engineering/) — 모델 아키텍처, GPU 하드웨어, inference engine, 최적화 기법, production serving을 아우르는 실무 안내서
- **Hands-On LLM Serving and Optimization** (Chi Wang, Peiheng Hu) — KV cache, batching, quantization, speculative decoding, 분산 serving과 vLLM 최적화를 실습 중심으로 다룹니다.
  - [Book](https://orca3.github.io/llm-model-inference/) · [Video](https://www.oreilly.com/library/view/hands-on-llm-serving/9798341621480/) · [Code and Notebooks](https://github.com/orca3/llm-model-inference)
