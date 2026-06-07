FROM nvcr.io/nvidia/pytorch:26.05-py3 AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    numactl \
    libnuma-dev \
    libjemalloc2 \
    google-perftools \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA allocator tuning.
# Workload-specific. Validate with memory_stats / memory_snapshot / Nsight Systems.
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:256,backend:cudaMallocAsync

# Stable GPU ordering.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Keep CPU thread defaults conservative.
# Override at runtime for DataLoader/tokenizer/video preprocessing-heavy workloads.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# CPU allocator selection:
#   none      -> default glibc malloc
#   jemalloc  -> jemalloc
#   tcmalloc  -> tcmalloc
ENV CPU_ALLOCATOR=none

# jemalloc tuning profile
ENV JEMALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000,narenas:8

# tcmalloc tuning profile
ENV TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=536870912
ENV TCMALLOC_RELEASE_RATE=16

WORKDIR /app

# Better Docker layer cache behavior
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

# Runtime launcher that selects CPU allocator dynamically.
RUN cat > /usr/local/bin/gpu-entrypoint.sh <<'EOF' && \
    chmod +x /usr/local/bin/gpu-entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

case "${CPU_ALLOCATOR:-none}" in
  none|"")
    echo "[gpu-entrypoint] CPU allocator: glibc malloc (default)"
    ;;

  jemalloc)
    JEMALLOC_LIB="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"

    if [[ ! -f "${JEMALLOC_LIB}" ]]; then
      echo "[gpu-entrypoint] ERROR: jemalloc library not found: ${JEMALLOC_LIB}" >&2
      exit 1
    fi

    export LD_PRELOAD="${JEMALLOC_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
    export MALLOC_CONF="${JEMALLOC_CONF:-background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000,narenas:8}"

    echo "[gpu-entrypoint] CPU allocator: jemalloc"
    echo "[gpu-entrypoint] MALLOC_CONF=${MALLOC_CONF}"
    ;;

  tcmalloc)
    TCMALLOC_LIB=""

    for candidate in \
      /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
      /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
    do
      if [[ -f "${candidate}" ]]; then
        TCMALLOC_LIB="${candidate}"
        break
      fi
    done

    if [[ -z "${TCMALLOC_LIB}" ]]; then
      echo "[gpu-entrypoint] ERROR: tcmalloc library not found" >&2
      exit 1
    fi

    export LD_PRELOAD="${TCMALLOC_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"

    echo "[gpu-entrypoint] CPU allocator: tcmalloc"
    echo "[gpu-entrypoint] TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=${TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES:-}"
    echo "[gpu-entrypoint] TCMALLOC_RELEASE_RATE=${TCMALLOC_RELEASE_RATE:-}"
    ;;

  *)
    echo "[gpu-entrypoint] ERROR: invalid CPU_ALLOCATOR='${CPU_ALLOCATOR}'" >&2
    echo "[gpu-entrypoint] valid values: none, jemalloc, tcmalloc" >&2
    exit 1
    ;;
esac

exec "$@"
EOF

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app

ENTRYPOINT ["/usr/local/bin/gpu-entrypoint.sh"]
CMD ["python", "train.py"]