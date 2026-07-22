# Deep Learning for Network Engineers

A six-week engineering study of how deep learning workloads shape GPU communication, distributed training, and AI data center backend networks.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/c4f68d13-1942-436a-bbe0-35f360da5456" />

* [Week 1: From Artificial Neuron to AI Training Fabric](./week01/README.md)
  Revisit artificial neurons, forward pass, backward pass, gradients, and weight updates from an infrastructure perspective. The key focus is understanding why training eventually leads to GPU memory pressure and distributed synchronization.

* [Week 2: Introduction to Large Language Models and Training](./week02/README.md)
  Study tokenization, embeddings, Transformer basics, attention, feed-forward layers, and next-token prediction. The goal is to understand why LLMs require large GPU memory, high compute throughput, and scalable distributed training.

* [Week 3: Parallelism Strategy and Communication Primitives](./week03/README.md)
  Study data parallelism, tensor parallelism, pipeline parallelism, and the communication patterns behind them. The focus is on how AllReduce, ReduceScatter, AllGather, and activation transfers affect GPU cluster performance.

* [Week 4: RDMA, RoCE, and NCCL Communication](./week04/README.md)
  Connect distributed training with RDMA-based GPU-to-GPU communication. The goal is to understand how NCCL uses high-speed backend networks and why low latency, high throughput, and lossless transport matter.

* Week 5: Congestion Control and AI Fabric Reliability
  Study the operational challenges of AI fabrics, including packet loss, congestion, PFC, ECN, DCQCN, head-of-line blocking, and hash polarization. The focus is on why AI training networks require careful congestion control and validation

* Week 6: Backend Network Topology and Design Review
  Study backend topology choices such as shared NIC, NIC-per-GPU, single-rail, dual-rail, and rail-optimized designs. The final goal is to produce an AI training fabric design review from the perspective of an MLOps/LLOps engineer.
