#!/usr/bin/env bash
set -euo pipefail

# gpu-node-baseline-tuning.sh
#
# Baseline OS tuning for GPU training/inference nodes.
#
# Design goals:
# - Safe by default: dry-run unless --apply is given
# - Idempotent: write dedicated config files instead of appending repeatedly
# - Profile-aware: training vs inference
# - Topology-aware enough to report, but does not blindly pin IRQs
#
# Usage:
#   sudo ./gpu-node-baseline-tuning.sh --profile training --apply
#   sudo ./gpu-node-baseline-tuning.sh --profile inference --apply
#   ./gpu-node-baseline-tuning.sh --profile training
#
# Profiles:
#   training   : throughput-oriented defaults
#   inference  : latency/jitter-oriented defaults

PROFILE="training"
APPLY="false"
DISABLE_SWAP="false"
SET_CPU_GOVERNOR="true"
SET_GPU_PERSISTENCE="true"
WRITE_PERSISTENT_CONFIG="true"

SYSCTL_CONF="/etc/sysctl.d/99-gpu-performance.conf"
LIMITS_CONF="/etc/security/limits.d/99-gpu-performance.conf"
SYSTEMD_CONF_DIR="/etc/systemd/system.conf.d"
SYSTEMD_CONF="${SYSTEMD_CONF_DIR}/99-gpu-performance.conf"
TMPFILES_CONF="/etc/tmpfiles.d/99-gpu-performance.conf"
REPORT_DIR="/var/log/gpu-node-tuning"
REPORT_FILE="${REPORT_DIR}/topology-report.txt"

usage() {
  cat <<EOF
Usage:
  $0 [--profile training|inference] [--apply] [--disable-swap]

Options:
  --profile        tuning profile: training or inference. default: training
  --apply          actually apply changes. default: dry-run
  --disable-swap   disable swap immediately and comment swap entries in /etc/fstab
  --no-governor    do not set CPU governor to performance
  --no-gpu-pm      do not enable NVIDIA persistence mode
  -h, --help       show this help

Examples:
  $0 --profile training
  sudo $0 --profile training --apply
  sudo $0 --profile inference --apply --disable-swap
EOF
}

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

run() {
  if [[ "${APPLY}" == "true" ]]; then
    log "RUN: $*"
    "$@"
  else
    log "DRY-RUN: $*"
  fi
}

write_file() {
  local path="$1"
  local content="$2"

  if [[ "${APPLY}" == "true" ]]; then
    log "Writing ${path}"
    mkdir -p "$(dirname "${path}")"
    printf "%s\n" "${content}" > "${path}"
  else
    log "DRY-RUN: write ${path}"
    echo "----- ${path} -----"
    printf "%s\n" "${content}"
    echo "-------------------"
  fi
}

require_root_if_apply() {
  if [[ "${APPLY}" == "true" && "${EUID}" -ne 0 ]]; then
    die "--apply requires root. Run with sudo."
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        PROFILE="${2:-}"
        shift 2
        ;;
      --apply)
        APPLY="true"
        shift
        ;;
      --disable-swap)
        DISABLE_SWAP="true"
        shift
        ;;
      --no-governor)
        SET_CPU_GOVERNOR="false"
        shift
        ;;
      --no-gpu-pm)
        SET_GPU_PERSISTENCE="false"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  case "${PROFILE}" in
    training|inference) ;;
    *) die "--profile must be either training or inference" ;;
  esac
}

collect_topology_report() {
  log "Collecting topology report"

  if [[ "${APPLY}" == "true" ]]; then
    mkdir -p "${REPORT_DIR}"
    : > "${REPORT_FILE}"
  fi

  {
    echo "==== GPU Node Tuning Topology Report ===="
    date
    echo

    echo "==== Host ===="
    hostname || true
    uname -a || true
    echo

    echo "==== CPU / NUMA ===="
    lscpu || true
    echo
    numactl -H || true
    echo

    echo "==== NVIDIA GPUs ===="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
      echo
      nvidia-smi topo -m || true
    else
      echo "nvidia-smi not found"
    fi
    echo

    echo "==== PCI Tree ===="
    lspci -tv || true
    echo

    echo "==== Network / RDMA Devices ===="
    ip -br link || true
    echo
    if command -v ibstat >/dev/null 2>&1; then
      ibstat || true
    else
      echo "ibstat not found"
    fi
    echo
    if command -v ibv_devinfo >/dev/null 2>&1; then
      ibv_devinfo || true
    else
      echo "ibv_devinfo not found"
    fi
    echo

    echo "==== Interrupts: NVIDIA / mlx / ib ===="
    grep -Ei "nvidia|mlx|ib" /proc/interrupts || true
    echo

    echo "==== Current sysctl sample ===="
    sysctl vm.swappiness kernel.numa_balancing vm.dirty_ratio vm.dirty_background_ratio 2>/dev/null || true
    sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem 2>/dev/null || true
    echo

    echo "==== THP ===="
    cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
    echo

    echo "==== Limits ===="
    ulimit -a || true
  } | if [[ "${APPLY}" == "true" ]]; then
        tee -a "${REPORT_FILE}" >/dev/null
      else
        cat
      fi

  if [[ "${APPLY}" == "true" ]]; then
    log "Topology report saved to ${REPORT_FILE}"
  fi
}

configure_sysctl() {
  log "Preparing sysctl configuration for profile=${PROFILE}"

  local dirty_ratio
  local dirty_background_ratio
  local thp_mode
  local thp_defrag

  if [[ "${PROFILE}" == "training" ]]; then
    # Training is usually throughput-oriented.
    # Keep dirty ratios moderate to avoid huge writeback bursts during checkpointing.
    dirty_ratio="20"
    dirty_background_ratio="10"

    # Safer than "always"; allows applications/runtime to opt in.
    thp_mode="madvise"
    thp_defrag="madvise"
  else
    # Inference is usually latency/jitter-oriented.
    # Lower dirty ratios reduce large writeback stalls.
    dirty_ratio="10"
    dirty_background_ratio="5"

    # Avoid THP compaction/collapse jitter for p95/p99-sensitive workloads.
    thp_mode="never"
    thp_defrag="never"
  fi

  local sysctl_content
  sysctl_content=$(cat <<EOF
# Managed by gpu-node-baseline-tuning.sh
# Profile: ${PROFILE}
#
# Memory behavior
vm.swappiness = 0
kernel.numa_balancing = 0

# Filesystem writeback behavior
vm.dirty_ratio = ${dirty_ratio}
vm.dirty_background_ratio = ${dirty_background_ratio}

# High-throughput TCP defaults.
# These help TCP-based storage, HTTP object store traffic, Ray/gRPC/HTTP serving,
# and NCCL socket fallback. They are not the main tuning knobs for native RDMA verbs.
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456

# Useful for high connection fan-in inference frontends.
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192

# Larger backlog may help high packet-rate environments, but validate with NIC counters.
net.core.netdev_max_backlog = 250000
EOF
)

  write_file "${SYSCTL_CONF}" "${sysctl_content}"

  if [[ "${APPLY}" == "true" ]]; then
    run sysctl --system
  fi

  configure_thp "${thp_mode}" "${thp_defrag}"
}

configure_thp() {
  local mode="$1"
  local defrag="$2"

  log "Configuring Transparent Huge Pages: enabled=${mode}, defrag=${defrag}"

  run bash -c "echo '${mode}' > /sys/kernel/mm/transparent_hugepage/enabled || true"
  run bash -c "echo '${defrag}' > /sys/kernel/mm/transparent_hugepage/defrag || true"

  if [[ "${WRITE_PERSISTENT_CONFIG}" == "true" ]]; then
    local tmpfiles_content
    tmpfiles_content=$(cat <<EOF
# Managed by gpu-node-baseline-tuning.sh
# Persist THP settings across reboot.
w /sys/kernel/mm/transparent_hugepage/enabled - - - - ${mode}
w /sys/kernel/mm/transparent_hugepage/defrag - - - - ${defrag}
EOF
)
    write_file "${TMPFILES_CONF}" "${tmpfiles_content}"
  fi
}

configure_limits() {
  log "Configuring memlock and nofile limits"

  local limits_content
  limits_content=$(cat <<EOF
# Managed by gpu-node-baseline-tuning.sh
#
# RDMA, UCX, NCCL, GPUDirect RDMA, and pinned-memory-heavy workloads
# may require large or unlimited locked memory.
* soft memlock unlimited
* hard memlock unlimited

# Large dataloaders, inference servers, Ray clusters, and sharded datasets
# may open many files/sockets.
* soft nofile 1048576
* hard nofile 1048576

root soft memlock unlimited
root hard memlock unlimited
root soft nofile 1048576
root hard nofile 1048576
EOF
)
  write_file "${LIMITS_CONF}" "${limits_content}"

  local systemd_content
  systemd_content=$(cat <<EOF
# Managed by gpu-node-baseline-tuning.sh
[Manager]
DefaultLimitMEMLOCK=infinity
DefaultLimitNOFILE=1048576
EOF
)
  write_file "${SYSTEMD_CONF}" "${systemd_content}"

  if [[ "${APPLY}" == "true" ]]; then
    run systemctl daemon-reexec
  fi
}

configure_cpu_governor() {
  if [[ "${SET_CPU_GOVERNOR}" != "true" ]]; then
    log "Skipping CPU governor tuning"
    return
  fi

  log "Setting CPU governor to performance"

  if command -v cpupower >/dev/null 2>&1; then
    run cpupower frequency-set -g performance
  elif command -v tuned-adm >/dev/null 2>&1; then
    # throughput-performance is a reasonable baseline, but validate on inference nodes.
    run tuned-adm profile throughput-performance
  else
    warn "Neither cpupower nor tuned-adm found. Skipping CPU governor tuning."
  fi
}

configure_swap() {
  if [[ "${DISABLE_SWAP}" != "true" ]]; then
    log "Skipping swap disable. Use --disable-swap to disable it."
    return
  fi

  log "Disabling swap"

  run swapoff -a

  if [[ "${APPLY}" == "true" ]]; then
    if grep -qE '^[^#].*\sswap\s' /etc/fstab; then
      log "Commenting swap entries in /etc/fstab"
      cp /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"
      sed -i '/^[^#].*\sswap\s/s/^/# gpu-node-tuning disabled swap: /' /etc/fstab
    fi
  else
    log "DRY-RUN: would comment active swap entries in /etc/fstab"
  fi
}

configure_gpu_persistence() {
  if [[ "${SET_GPU_PERSISTENCE}" != "true" ]]; then
    log "Skipping GPU persistence mode"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found. Skipping GPU persistence mode."
    return
  fi

  log "Enabling NVIDIA persistence mode"
  run nvidia-smi -pm 1
}

print_irq_guidance() {
  cat <<'EOF'

[INFO] IRQ affinity was intentionally NOT changed.

Do not blindly pin all NVIDIA or mlx interrupts to one CPU.
That often creates a new bottleneck.

Recommended next step:
  1. Inspect topology:
       nvidia-smi topo -m
       numactl -H
       lspci -tv
       grep -Ei "nvidia|mlx|ib" /proc/interrupts

  2. Map each GPU/NIC PCI device to NUMA:
       cat /sys/bus/pci/devices/<PCI_ADDR>/numa_node

  3. Pin GPU/NIC IRQs to CPUs local to the same NUMA domain.

  4. If irqbalance is enabled, configure banned CPUs or policy carefully:
       systemctl status irqbalance
       cat /etc/default/irqbalance 2>/dev/null || true

For DGX/HGX/B200/H100 class nodes, IRQ affinity should be generated from
the actual GPU/NIC/NUMA topology, not hardcoded.
EOF
}

print_kubernetes_guidance() {
  cat <<'EOF'

[INFO] Kubernetes-specific follow-up checklist

For GPU Kubernetes nodes, host tuning alone is not enough.
Check these separately:

  kubelet:
    --cpu-manager-policy=static
    --topology-manager-policy=single-numa-node or restricted
    --reserved-cpus=<housekeeping CPU set>
    --kube-reserved / --system-reserved

  Pod QoS:
    Use Guaranteed Pods for CPU-pinned training/inference workloads.

  NVIDIA:
    NVIDIA GPU Operator / device plugin topology hints
    DCGM exporter for GPU/NVLink/PCIe telemetry

  RDMA / SR-IOV:
    RDMA device plugin or SR-IOV device plugin
    Multus NAD topology
    UCX/NCCL env validation

  NCCL validation:
    NCCL_DEBUG=INFO
    NCCL_IB_DISABLE=0
    NCCL_SOCKET_IFNAME=<expected interface>
    NCCL_IB_HCA=<expected HCA>
    nccl-tests all_reduce_perf / all_gather_perf
EOF
}

print_validation_plan() {
  cat <<'EOF'

[INFO] Validate before/after with reproducible benchmarks.

Training:
  - step time p50/p95
  - GPU utilization
  - dataloader wait time
  - H2D copy time
  - NCCL all-reduce bandwidth
  - checkpoint latency

Inference:
  - TTFT
  - TPOT
  - tokens/sec
  - p95/p99 latency
  - CPU utilization
  - GPU utilization
  - KV cache memory pressure

Useful commands:
  nvidia-smi dmon -s pucvmet -d 1
  dcgmi dmon
  mpstat -P ALL 1
  pidstat -t -p <PID> 1
  numastat -p <PID>
  iostat -x 1
  vmstat 1
  perf stat -p <PID>
  grep -Ei "nvidia|mlx|ib" /proc/interrupts
  nvidia-smi topo -m

NCCL:
  NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8
EOF
}

main() {
  parse_args "$@"
  require_root_if_apply

  log "GPU node baseline tuning"
  log "profile=${PROFILE}"
  log "apply=${APPLY}"
  log "disable_swap=${DISABLE_SWAP}"

  collect_topology_report
  configure_sysctl
  configure_limits
  configure_cpu_governor
  configure_swap
  configure_gpu_persistence

  print_irq_guidance
  print_kubernetes_guidance
  print_validation_plan

  if [[ "${APPLY}" == "true" ]]; then
    log "Tuning applied."
    log "Some settings may require logout/login or reboot to fully take effect."
    log "Topology report: ${REPORT_FILE}"
  else
    log "Dry-run complete. Re-run with --apply to make changes."
  fi
}

main "$@"