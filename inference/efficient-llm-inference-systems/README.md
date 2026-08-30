# Efficient LLM Inference Systems

LLM 추론의 성능 지표와 하드웨어 특성을 이해하고, KV cache, 양자화, speculative decoding을 실제 측정 결과와 연결해 학습합니다. 이 문서는 과정의 전체 구성과 부록을 안내하며, 세부 설명과 참고 논문은 각 주차 문서에서 다룹니다.

## Curriculum

- [Week 1: Understanding Performance Metrics](week01/README.md) — TTFT, TPOT, batch throughput과 메모리 한계를 측정합니다.
- [Week 2: Hardware Foundations for Inference](week02/README.md) — 메모리 계층, Tensor Core, roofline과 GPU 간 통신을 다룹니다.
- [Week 3: Transformer Inference and the KV Cache](week03/README.md) — MHA·MQA·GQA·MLA와 장문 context의 메모리 비용을 분석합니다.
- [Week 4: Quantization](week04/README.md) — 정밀도, 양자화 알고리즘, kernel 지원과 품질·성능 trade-off를 비교합니다.
- [Week 5: Speculative Decoding](week05/README.md) — acceptance rate와 draft 전략을 바탕으로 latency 개선 조건을 검토합니다.

## Appendix

- [Hardware Architectures for LLM Inference](appendix/hardware-architectures/README.md)
- [LLM Inference](appendix/llm-inference/README.md)
- [Transformer](appendix/transformer/README.md)

## Source Material

- [Efficient LLM Inference Systems, Algorithms & Production Engineering — Interview Pocket Notes](https://drive.google.com/file/d/1mfTzOnwn8yx4eKObjPvpd-B_toGkQ_tu/view) (2026)

## Related Topics

- [SGLang Production Practices](../sglang-production-practices/README.md) — KV cache, speculative decoding, routing과 분산 topology를 production SLO와 연결합니다.
- [Model Architecture and Serving Profiles](../models/README.md) — 모델별 architecture와 serving 특성을 정리합니다.
- [FlashAttention from First Principles](../flashattention-from-first-principles/README.md) ([한국어 번역](../flashattention-from-first-principles/README.ko.md)) — exact attention의 I/O-aware kernel 최적화를 설명합니다.
