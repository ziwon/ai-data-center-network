# Efficient LLM Inference Systems

- [Week 1: Understanding Performance Metrics](week01/README.md)
- [Week 2: Hardware Foundations for Inference](week02/README.md)
- [Week 3: Transformer Inference and the KV Cache](week03/README.md)
- [Week 4: Quantization](week04/README.md)
- [Week 5: Speculative Decoding](week05/README.md)

## Appendix

- [Hardware Architectures for LLM Inference](appendix/hardware-architectures/README.md)
- [LLM Inference](appendix/llm-inference/README.md)
- [Transformer](appendix/transformer/README.md)

## Resources

### Books

- [Efficient LLM Inference Systems, Algorithms & Production Engineering - Interview Pocket Notes](https://drive.google.com/file/d/1mfTzOnwn8yx4eKObjPvpd-B_toGkQ_tu/view) (2026)
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)

### Papers

#### Inference Systems

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (2022.11)
- [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) (2023.11)
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) (2022)

#### Attention, KV Cache, and Long Context

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019.11)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021.04)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023.05)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (2023.09)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024.05)

#### Quantization

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022.08)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022.10)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438) (2022.11)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2023.06)
- [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) (2024.01)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) (2024.02)

#### Speculative Decoding

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022.11)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023.02)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) (2024.01)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) (2024.06)

#### General Scaling

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020.01)

### LLM Architecture

- [Kimi K3 기술 해부: 2.8T MoE, KDA, Attention Residuals, 그리고 64-GPU 서빙](../models/kimi-k3.md)
- [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- [The Big LLM Architecture Comparison](https://www.youtube.com/watch?v=rNlULI-zGcw)
- [The Big LLM Architecture Comparison Blog](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
