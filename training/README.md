# MLPerf Training

[MLCommons Training](https://github.com/mlcommons/training)은 MLPerf Training 벤치마크의 reference implementation 모음입니다. 실제 제출용 최적화 코드라기보다, AI 데이터센터에서 대규모 학습 워크로드가 어떤 compute, network, storage, orchestration 요구사항을 만드는지 읽고 실험하기 좋은 기준 코드입니다.

2026년 기준으로는 [MLPerf Training v6.0](https://github.com/mlcommons/training#mlperf-training-v60-submission-deadline-may-15-2026)이 최신 벤치마크 표에 올라와 있으며, LLM pretraining, MoE, LoRA fine-tuning, text-to-image, recommendation 모델이 포함되어 있습니다.

## 읽을 자료

| 주제 | 링크 | 시스템 관점에서 볼 내용 |
| --- | --- | --- |
| 전체 개요 | [mlcommons/training](https://github.com/mlcommons/training) | 벤치마크 목록, 실행 공통 절차, 데이터셋/컨테이너/타깃 품질 기준 |
| MLPerf Training paper | [arXiv:1910.01500](https://arxiv.org/abs/1910.01500) | benchmark design, time-to-train, target quality, closed/open division |
| v6.0 benchmark table | [MLPerf Training v6.0](https://github.com/mlcommons/training#mlperf-training-v60-submission-deadline-may-15-2026) | 2026년 제출 기준 모델, framework, dataset, parameter count |
| Llama 3.1 8B pretraining | [small_llm_pretraining/nemo](https://github.com/mlcommons/training/tree/master/small_llm_pretraining/nemo) | NeMo 기반 LLM pretraining, C4 dataset, Slurm job, checkpoint resume |
| Llama 3.1 405B pretraining | [large_language_model_pretraining/nemo](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo) | 초대형 dense LLM 학습, distributed checkpoint, multi-node Slurm |
| DeepSeek V3 671B MoE | [llm_moe_pretraining/nemo](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo) | MoE expert parallelism, GBS 제약, GB300급 multi-node 학습 |
| GPT-OSS 20B MoE | [small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus) | Primus 기반 MoE 학습, AMD/NVIDIA 단일 노드 예제, expert parallelism |
| Llama2 70B LoRA | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | PEFT/LoRA fine-tuning, 8k sequence, Accelerate, FlashAttention |
| FLUX.1 text-to-image | [text_to_image](https://github.com/mlcommons/training/tree/master/text_to_image) | TorchTitan 기반 diffusion/flow model 학습, preprocessed embedding, HSDP/DDP |
| DLRM DCNv2 recommendation | [recommendation_v2/torchrec_dlrm](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | TorchRec embedding sharding, model parallel embedding table, Criteo multi-hot data |
| MLCommons R2 Downloader | [mlcommons/r2-downloader](https://github.com/mlcommons/r2-downloader) | C4, tokenizer, Criteo, FLUX dataset 다운로드 자동화 |
| MLCommons training storage | [training.mlcommons-storage.org](https://training.mlcommons-storage.org) | 벤치마크용 공개/회원 제한 데이터셋과 checkpoint metadata |

## 2026년 v6.0 벤치마크 요약

| 모델 | Reference implementation | Framework | Dataset | 시스템 포인트 |
| --- | --- | --- | --- | --- |
| FLUX.1 | [text_to_image](https://github.com/mlcommons/training/tree/master/text_to_image) | TorchTitan | CC12M subset | GPU memory, image/text embedding preprocessing, HSDP/DDP |
| Llama 3.1 8B | [small_llm_pretraining/nemo](https://github.com/mlcommons/training/tree/master/small_llm_pretraining/nemo) | NeMo | C4 | tensor/data parallel, C4 dataloader, checkpoint resume |
| Llama2 70B LoRA | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | PyTorch | SCROLLS GovReport | PEFT, 8k context, FlashAttention, 8 GPU fine-tuning |
| Llama 3.1 405B | [large_language_model_pretraining/nemo](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo) | NeMo | C4 | multi-node Slurm, checkpoint conversion/resume, high-bandwidth fabric |
| DLRM DCNv2 | [recommendation_v2/torchrec_dlrm](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | TorchRec | Criteo 3.5TB multi-hot | embedding all-to-all, host memory, mmap, storage throughput |
| GPT-OSS 20B MoE | [small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus) | Primus | C4 | expert parallelism, MoE routing, AMD/NVIDIA portability |
| DeepSeek V3 671B | [llm_moe_pretraining/nemo](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo) | NeMo | C4 | 256 GPU급 MoE, expert parallel, large GBS, frequent evaluation |

## 예제 실행

### 1. Llama 3.1 8B: NeMo 기반 분산 pretraining

작은 LLM pretraining reference입니다. `config.sh`에 Slurm, container, dataset, checkpoint 경로를 채운 뒤 job을 제출합니다.

```bash
git clone https://github.com/mlcommons/training.git mlcommons-training
cd mlcommons-training/small_llm_pretraining/nemo

docker build -t mlperf-llama31-8b -f Dockerfile .

# C4 pre-tokenized dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/llama3_1_8b/c4 \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Llama 3.1 8B tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/llama3_1_8b/tokenizer \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

# config.sh에서 Slurm partition/account, container image, PREPROCESSED_PATH, TOKENIZER_PATH 등을 설정합니다.
source config.sh
bash run_llama31.sh
```

관찰 포인트:

- GPU utilization이 dataloader, checkpoint write/read, NCCL collective 중 어디에서 막히는지 확인합니다.
- `config.sh`의 tensor/pipeline/data parallel 설정과 실제 node/GPU topology가 맞는지 봅니다.
- checkpoint resume을 켜면 shared storage와 network fabric tail latency가 학습 시간에 미치는 영향을 볼 수 있습니다.

### 2. DLRM DCNv2: TorchRec embedding sharding과 all-to-all

추천 모델은 embedding table이 커서 GPU 간 all-to-all, host memory, mmap I/O 병목을 보기 좋습니다.

```bash
cd mlcommons-training/recommendation_v2/torchrec_dlrm
pip install -r requirements.txt
pip install torchx

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/dlrmv2-preprocessed-criteo-click-logs.uri

export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=65536
export WORLD_SIZE=8
export CRITEO_MULTI_HOT=/data/criteo/multi_hot

torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
  --embedding_dim 128 \
  --dense_arch_layer_sizes 512,256,128 \
  --over_arch_layer_sizes 1024,1024,512,256,1 \
  --synthetic_multi_hot_criteo_path "$CRITEO_MULTI_HOT" \
  --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
  --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
  --epochs 1 \
  --pin_memory \
  --mmap_mode \
  --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
  --interaction_type=dcn \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --learning_rate 0.005
```

관찰 포인트:

- `--mmap_mode` on/off로 startup time, storage read pattern, steady-state throughput을 비교합니다.
- embedding table sharding이 all-to-all traffic을 얼마나 만드는지 NCCL/IB/RoCE counter와 함께 봅니다.
- local batch size를 바꾸며 GPU compute, network, host memory bandwidth 중 병목이 어디로 이동하는지 확인합니다.

### 3. FLUX.1: TorchTitan HSDP/DDP text-to-image training

FLUX.1 예제는 image/text encoder preprocessing과 distributed training mode를 함께 보기 좋습니다.

```bash
cd mlcommons-training/text_to_image
export MLCOMMONS_TRAINING_ROOT="$(pwd)/.."

cd torchtitan
docker build -t mlperf-flux -f Dockerfile .
cd ..

mkdir -p /data/flux /models/flux /logs/flux

# Preprocessed embedding을 쓰면 frozen encoder를 매 step 실행하지 않아도 됩니다.
cd /data/flux
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

cd "$MLCOMMONS_TRAINING_ROOT/text_to_image"

export DATAROOT=/data/flux
export MODELROOT=/models/flux
export LOGDIR=/logs/flux
export CONFIG_FILE=torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml
export CONT=mlperf-flux
export SEED=1234

sbatch -N 2 -t 04:00:00 run.sub \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=8
```

관찰 포인트:

- 기본은 node 내부 sharding과 node 간 DDP를 섞는 HSDP 구조입니다.
- preprocessed embedding을 쓰는 경우 storage capacity와 read bandwidth가 커지고, encoder compute 부담은 줄어듭니다.
- checkpointing을 켜려면 `ENABLE_CHECKPOINTING=True`와 `--checkpoint.interval=<steps>`를 같이 설정합니다.

### 4. GPT-OSS 20B MoE: 단일 노드 MoE 학습

Primus 기반 MoE 예제는 B200/MI355X처럼 서로 다른 accelerator에서 expert parallelism을 비교하기 좋습니다.

```bash
cd mlcommons-training/small_llm_moe_pretraining/primus

# NVIDIA B200이면 Dockerfile.nvidia, AMD MI355X이면 Dockerfile을 기준으로 빌드합니다.
docker build -t mlperf-gpt-oss-20b -f Dockerfile.nvidia .

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/gpt_oss_20b/data \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=mlperf-gpt-oss-20b

source config_B200_1x8x1.sh
export NEXP=1
bash run_with_docker.sh
```

관찰 포인트:

- MoE는 dense LLM보다 all-to-all과 expert load balancing이 중요합니다.
- 같은 8 GPU 단일 노드에서도 NVLink, PCIe, NUMA 배치에 따라 expert parallel 성능이 달라질 수 있습니다.
- Kubernetes에서 실행한다면 [AI Systems Performance Engineering Chapter 3](../ai-system-performance-engineering/chap03/README.md)의 NUMA-local GPU workload case study처럼 CPU pinning, Topology Manager, GPU-local memory path를 함께 확인해야 합니다.
- AMD/NVIDIA config를 나눠 accelerator별 software stack 차이를 비교할 수 있습니다.

### 5. DeepSeek V3 671B: 대규모 MoE multi-node Slurm

이 예제는 일반 로컬 실험보다는 대규모 클러스터 설계 참고용입니다. reference는 Slurm 기반 multi-node 실행을 전제로 합니다.

```bash
cd mlcommons-training/llm_moe_pretraining/nemo

docker build -t mlperf-deepseek-v3 -f Dockerfile .

python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/NVIDIA-NeMo/Run.git

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/deepseekv3/c4 \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/deepseekv3/tokenizer \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

# config 파일에서 Slurm, IMAGE, DATA_DIR, MODEL_CKPT, parallelism 값을 환경에 맞게 수정합니다.
source config_GB300_64x4x256xtp1pp4cp1.sh
bash run_deepseek_v3_671b.sh
```

관찰 포인트:

- DeepSeek V3 671B는 total parameter 671B, active parameter 37B인 MoE 구조입니다.
- benchmark는 큰 GBS를 강제하므로 fabric bandwidth, all-to-all latency, evaluation overhead가 중요합니다.
- checkpoint는 별도 다운로드/변환 절차가 필요하며, multi-TB급 shared filesystem 설계가 선행되어야 합니다.

## 실험 아이디어

- 같은 C4 dataset을 Llama 3.1 8B dense pretraining과 GPT-OSS 20B MoE pretraining에 넣고 NCCL collective pattern을 비교합니다.
- DLRM DCNv2에서 `--mmap_mode`, local batch size, embedding sharding 설정을 바꿔 storage/network/compute 병목 전환점을 찾습니다.
- FLUX.1을 raw image pipeline과 preprocessed embedding pipeline으로 나눠 CPU preprocessing, storage capacity, GPU utilization 차이를 봅니다.
- checkpoint interval을 바꾸며 shared filesystem, object storage, local NVMe의 write burst와 recovery time을 비교합니다.
- Slurm job log, MLPerf log, GPU profiler, NCCL debug log, fabric counter를 같은 타임라인으로 맞춰 time-to-train 병목을 재구성합니다.
