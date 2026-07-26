# Inference

LLM 추론의 모델 구조와 시스템 최적화 자료를 주제별로 분리해 정리합니다.

## Model Architecture and Serving Profiles

모델별 아키텍처, attention과 MoE 설계, 메모리·통신 특성, 자체 호스팅 전략을 다룹니다.

- [Models](models/README.md)
  - [How We Scaled Kimi K2.5 강의 노트](models/kimi-k2-5-scaling.md)
  - [Kimi K3 기술 해부](models/kimi-k3.md)

## Efficient LLM Inference Systems

성능 지표, 하드웨어, KV cache, 양자화, speculative decoding을 주차별 실습과 함께 다룹니다.

- [Efficient LLM Inference Systems](efficient-llm-inference-systems/README.md)
