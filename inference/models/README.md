# Model Architecture and Serving Profiles

LLM별 핵심 아키텍처와 실제 추론·서빙 시스템에서 확인해야 할 메모리, 통신, 캐시, 병렬화 특성을 정리합니다. 시스템 최적화 기법 자체보다 모델 설계가 serving profile에 미치는 영향에 집중합니다.

## Model Notes

- [How We Scaled Kimi K2.5: 토큰 효율, 장문 컨텍스트, 에이전트 스웜](kimi-k2-5-scaling.md)
- [Kimi K3 기술 해부: 2.8T MoE, KDA, Attention Residuals, 그리고 64-GPU 서빙](kimi-k3.md)

## Architecture Foundations

- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) — Transformer와 LLM 구성 요소를 구현 관점에서 설명합니다.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020.01) — model size, dataset size, compute 사이의 scaling 관계를 정리합니다.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024.05) — MLA와 MoE 설계가 학습 및 추론 비용에 미치는 영향을 다룹니다.

## Architecture Comparisons and References

- [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [The Big LLM Architecture Comparison](https://www.youtube.com/watch?v=rNlULI-zGcw)
- [The Big LLM Architecture Comparison Blog](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
