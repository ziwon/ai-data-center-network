#!/usr/bin/env bash
################################################################################
# GB200/GB300 Grace-Blackwell Baseline NUMA and Runtime Tuning Script
################################################################################
#
# Safe baseline tuning for Grace-Blackwell GPU nodes.
#
# What this script does:
# - Detects Grace-Blackwell-like systems before applying changes
# - Collects a topology report for GPU/CPU/NUMA/NIC/IRQ review
# - Applies conservative OS/runtime tuning through sysctl.d, limits.d, tmpfiles.d
# - Enables NVIDIA persistence mode
# - Sets CPU governor to performance when available
# - Creates a NUMA-aware helper for single-GPU process launches
# - Saves original settings for reset
#
# What this script intentionally does NOT do:
# - It does not reset GPUs
# - It does not stop irqbalance by default
# - It does not hardcode IRQ affinity
# - It does not blindly set application clocks
# - It does not append repeatedly to /etc/sysctl.conf or /etc/security/limits.conf
#
# Usage:
#   sudo ./gb_grace_blackwell_numa_tuning.sh --apply --profile training
#   sudo ./gb_grace_blackwell_numa_tuning.sh --apply --profile inference --disable-swap
#   ./gb_grace_blackwell_numa_tuning.sh --check
#   sudo ./gb_grace_blackwell_numa_tuning.sh --reset
#
################################################################################

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
MODE="check"
PROFILE="training"
FORCE="false"
DISABLE_SWAP="false"
SET_CPU_GOVERNOR="true"
SET_GPU_PERSISTENCE="true"
CREATE_BOOT_SERVICE="true"
SET_APP_CLOCKS="false"
APP_CLOCKS="auto"

STATE_DIR="/var/lib/gb-node-tuning"
REPORT_DIR="/var/log/gb-node-tuning"
REPORT_FILE="${REPORT_DIR}/topology-report.txt"

SYSCTL_CONF="/etc/sysctl.d/99-gb-gpu-performance.conf"
LIMITS_CONF="/etc/security/limits.d/99-gb-gpu-performance.conf"
SYSTEMD_CONF_DIR="/etc/systemd/system.conf.d"
SYSTEMD_CONF="${SYSTEMD_CONF_DIR}/99-gb-gpu-performance.conf"
TMPFILES_CONF="/etc/tmpfiles.d/99-gb-thp.conf"
RUNTIME_SCRIPT="/usr/local/sbin/gb-node-runtime-tuning.sh"
RUNTIME_SERVICE="/etc/systemd/system/gb-node-runtime-tuning.service"
NUMA_HELPER="/usr/local/bin/numa_bind_gpu.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
  cat <<EOF_USAGE
GB200/GB300 Grace-Blackwell baseline NUMA/runtime tuning

Usage:
  ${SCRIPT_NAME} --check [--profile training|inference]
  sudo ${SCRIPT_NAME} --apply [--profile training|inference] [options]
  sudo ${SCRIPT_NAME} --reset

Modes:
  --check                 Show topology and current settings. No changes. Default.
  --apply                 Apply baseline tuning. Requires root.
  --reset                 Restore settings saved during first apply. Requires root.

Profiles:
  --profile training      Throughput-oriented baseline. Default.
  --profile inference     Latency/jitter-oriented baseline.

Apply options:
  --disable-swap          Disable swap now and comment swap entries in /etc/fstab.
  --no-governor           Do not set CPU governor to performance.
  --no-gpu-pm             Do not enable NVIDIA persistence mode.
  --no-boot-service       Do not create systemd service for runtime settings.
  --force                 Apply even if Grace-Blackwell detection is not confirmed.

Advanced option, disabled by default:
  --set-app-clocks [auto|MEM,SM]
                          Set NVIDIA application clocks. Use only after validating
                          power, thermal, scheduler, and cluster policy.

Examples:
  ${SCRIPT_NAME} --check
  sudo ${SCRIPT_NAME} --apply --profile training
  sudo ${SCRIPT_NAME} --apply --profile inference --disable-swap
  sudo ${SCRIPT_NAME} --apply --force --no-boot-service
  sudo ${SCRIPT_NAME} --reset
EOF_USAGE
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "This mode requires root. Run with sudo."
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --check)
        MODE="check"
        shift
        ;;
      --apply)
        MODE="apply"
        shift
        ;;
      --reset)
        MODE="reset"
        shift
        ;;
      --profile)
        PROFILE="${2:-}"
        shift 2
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
      --no-boot-service)
        CREATE_BOOT_SERVICE="false"
        shift
        ;;
      --force)
        FORCE="true"
        shift
        ;;
      --set-app-clocks)
        SET_APP_CLOCKS="true"
        if [[ $# -gt 1 && "${2:-}" != --* ]]; then
          APP_CLOCKS="$2"
          shift 2
        else
          APP_CLOCKS="auto"
          shift
        fi
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

  case "${MODE}" in
    check|apply|reset) ;;
    *) die "Invalid mode: ${MODE}" ;;
  esac

  case "${PROFILE}" in
    training|inference) ;;
    *) die "--profile must be training or inference" ;;
  esac
}

run() {
  log "RUN: $*"
  "$@"
}

write_file() {
  local path="$1"
  local content="$2"
  mkdir -p "$(dirname "${path}")"
  log "Writing ${path}"
  printf "%s\n" "${content}" > "${path}"
}

read_or_empty() {
  local path="$1"
  if [[ -r "${path}" ]]; then
    cat "${path}"
  fi
}

current_thp_mode() {
  local path="$1"
  if [[ -r "${path}" ]]; then
    grep -o '\[[^]]*\]' "${path}" | tr -d '[]' || true
  fi
}

normalize_pci_bus_id() {
  local raw="$1"
  local bus
  bus="$(echo "${raw}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"

  # nvidia-smi often returns 00000000:xx:yy.z, while sysfs typically uses 0000:xx:yy.z.
  if [[ "${bus}" =~ ^([0-9a-f]{8}):(.+)$ ]]; then
    local domain="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    bus="${domain:4}:$(echo "${rest}" | tr '[:upper:]' '[:lower:]')"
  fi

  echo "${bus}"
}

get_gpu_indices() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true
  fi
}

get_gpu_pci_bus() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-gpu=pci.bus_id --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || true
}

get_gpu_numa_node() {
  local gpu="$1"
  local pci_raw pci_sysfs numa_path node

  pci_raw="$(get_gpu_pci_bus "${gpu}")"
  [[ -n "${pci_raw}" ]] || { echo "N/A"; return 0; }

  pci_sysfs="$(normalize_pci_bus_id "${pci_raw}")"
  numa_path="/sys/bus/pci/devices/${pci_sysfs}/numa_node"

  if [[ -r "${numa_path}" ]]; then
    node="$(cat "${numa_path}" 2>/dev/null || true)"
    if [[ "${node}" =~ ^[0-9]+$ ]]; then
      echo "${node}"
    else
      echo "N/A"
    fi
  else
    echo "N/A"
  fi
}

get_gpu_name() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || true
}

detect_grace_blackwell() {
  log "Detecting Grace-Blackwell-like system"

  local ok="true"
  local arch gpu_count gpu_names

  arch="$(uname -m)"
  echo "Architecture: ${arch}"
  if [[ "${arch}" != "aarch64" ]]; then
    warn "Not ARM64. Grace CPU systems should report aarch64."
    ok="false"
  fi

  if grep -qiE 'grace|neoverse' /proc/cpuinfo 2>/dev/null || lscpu 2>/dev/null | grep -qiE 'grace|neoverse'; then
    echo "CPU hint: Grace/Neoverse-like CPU detected"
  else
    warn "Grace CPU was not confirmed from /proc/cpuinfo or lscpu."
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found."
    ok="false"
  else
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    echo "NVIDIA GPU count: ${gpu_count}"
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    echo "GPU names:"
    echo "${gpu_names}"

    if echo "${gpu_names}" | grep -qiE 'b200|b300|gb200|gb300|blackwell'; then
      echo "GPU hint: Blackwell/GB200/GB300-like GPU detected"
    else
      warn "Blackwell/GB200/GB300 GPU was not confirmed from nvidia-smi names."
      ok="false"
    fi
  fi

  if ! command -v numactl >/dev/null 2>&1; then
    warn "numactl not found. Install numactl before using NUMA binding."
  fi

  if [[ "${ok}" == "true" ]]; then
    log "Grace-Blackwell detection confirmed"
    return 0
  fi

  warn "Grace-Blackwell detection was not fully confirmed"
  return 1
}

print_gpu_numa_table() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found"
    return 0
  fi

  echo "GPU NUMA table:"
  printf "%-6s %-40s %-20s %-8s\n" "GPU" "Name" "PCI" "NUMA"
  printf "%-6s %-40s %-20s %-8s\n" "---" "----" "---" "----"

  local gpu name pci numa
  while read -r gpu; do
    [[ -n "${gpu}" ]] || continue
    name="$(get_gpu_name "${gpu}")"
    pci="$(get_gpu_pci_bus "${gpu}")"
    numa="$(get_gpu_numa_node "${gpu}")"
    printf "%-6s %-40s %-20s %-8s\n" "${gpu}" "${name}" "${pci}" "${numa}"
  done < <(get_gpu_indices)
}

collect_topology_report() {
  if [[ "${MODE}" == "apply" ]]; then
    mkdir -p "${REPORT_DIR}"
    : > "${REPORT_FILE}"
  fi

  {
    echo "==== GB Node Topology Report ===="
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
      print_gpu_numa_table || true
      echo
      nvidia-smi topo -m || true
    else
      echo "nvidia-smi not found"
    fi
    echo

    echo "==== PCI Tree ===="
    lspci -tv 2>/dev/null || true
    echo

    echo "==== Network / RDMA Devices ===="
    ip -br link 2>/dev/null || true
    echo
    ibstat 2>/dev/null || echo "ibstat not found"
    echo
    ibv_devinfo 2>/dev/null || echo "ibv_devinfo not found"
    echo

    echo "==== Interrupts: NVIDIA / mlx / ib / nvme ===="
    grep -Ei 'nvidia|mlx|ib|nvme' /proc/interrupts || true
    echo

    echo "==== Current sysctl sample ===="
    sysctl kernel.numa_balancing vm.swappiness vm.zone_reclaim_mode vm.dirty_ratio vm.dirty_background_ratio 2>/dev/null || true
    sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem 2>/dev/null || true
    echo

    echo "==== Transparent Huge Pages ===="
    cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
    echo

    echo "==== Current limits ===="
    ulimit -a || true
    echo

    echo "==== irqbalance ===="
    systemctl is-active irqbalance 2>/dev/null || true
    systemctl is-enabled irqbalance 2>/dev/null || true
  } | if [[ "${MODE}" == "apply" ]]; then
        tee -a "${REPORT_FILE}" >/dev/null
      else
        cat
      fi

  if [[ "${MODE}" == "apply" ]]; then
    log "Topology report saved to ${REPORT_FILE}"
  fi
}

check_settings() {
  log "Current settings"

  if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    echo "CPU governor cpu0: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  else
    echo "CPU governor: not available"
  fi

  [[ -f /proc/sys/kernel/numa_balancing ]] && echo "NUMA balancing: $(cat /proc/sys/kernel/numa_balancing)"
  [[ -f /proc/sys/vm/swappiness ]] && echo "Swappiness: $(cat /proc/sys/vm/swappiness)"
  [[ -f /proc/sys/vm/zone_reclaim_mode ]] && echo "Zone reclaim mode: $(cat /proc/sys/vm/zone_reclaim_mode)"

  if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    echo "THP enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  fi
  if [[ -f /sys/kernel/mm/transparent_hugepage/defrag ]]; then
    echo "THP defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA persistence mode:"
    nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader,nounits 2>/dev/null || true
  fi

  echo "Managed config files:"
  for f in "${SYSCTL_CONF}" "${LIMITS_CONF}" "${SYSTEMD_CONF}" "${TMPFILES_CONF}" "${RUNTIME_SERVICE}" "${NUMA_HELPER}"; do
    if [[ -e "${f}" ]]; then
      echo "  present: ${f}"
    else
      echo "  absent : ${f}"
    fi
  done
}

save_snapshot() {
  mkdir -p "${STATE_DIR}"

  if [[ -f "${STATE_DIR}/snapshot.created" ]]; then
    log "Snapshot already exists at ${STATE_DIR}; not overwriting original state"
    return 0
  fi

  log "Saving original settings to ${STATE_DIR}"
  date > "${STATE_DIR}/snapshot.created"

  [[ -r /proc/sys/kernel/numa_balancing ]] && cat /proc/sys/kernel/numa_balancing > "${STATE_DIR}/kernel_numa_balancing"
  [[ -r /proc/sys/vm/swappiness ]] && cat /proc/sys/vm/swappiness > "${STATE_DIR}/vm_swappiness"
  [[ -r /proc/sys/vm/zone_reclaim_mode ]] && cat /proc/sys/vm/zone_reclaim_mode > "${STATE_DIR}/vm_zone_reclaim_mode"
  [[ -r /proc/sys/vm/dirty_ratio ]] && cat /proc/sys/vm/dirty_ratio > "${STATE_DIR}/vm_dirty_ratio"
  [[ -r /proc/sys/vm/dirty_background_ratio ]] && cat /proc/sys/vm/dirty_background_ratio > "${STATE_DIR}/vm_dirty_background_ratio"

  current_thp_mode /sys/kernel/mm/transparent_hugepage/enabled > "${STATE_DIR}/thp_enabled" || true
  current_thp_mode /sys/kernel/mm/transparent_hugepage/defrag > "${STATE_DIR}/thp_defrag" || true

  : > "${STATE_DIR}/cpu_governors.tsv"
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -r "${gov}" ]] || continue
    printf "%s\t%s\n" "${gov}" "$(cat "${gov}")" >> "${STATE_DIR}/cpu_governors.tsv"
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader,nounits > "${STATE_DIR}/gpu_persistence.csv" 2>/dev/null || true
  fi

  if command -v systemctl >/dev/null 2>&1; then
    systemctl is-active irqbalance > "${STATE_DIR}/irqbalance_active" 2>/dev/null || true
    systemctl is-enabled irqbalance > "${STATE_DIR}/irqbalance_enabled" 2>/dev/null || true
  fi
}

configure_sysctl() {
  local dirty_ratio dirty_background_ratio swappiness

  if [[ "${PROFILE}" == "training" ]]; then
    dirty_ratio="20"
    dirty_background_ratio="10"
    swappiness="0"
  else
    dirty_ratio="10"
    dirty_background_ratio="5"
    swappiness="0"
  fi

  local content
  content="$(cat <<EOF_SYSCTL
# Managed by ${SCRIPT_NAME}
# Profile: ${PROFILE}

# Avoid OS-driven memory migration and swapping jitter on GPU nodes.
kernel.numa_balancing = 0
vm.swappiness = ${swappiness}

# Prefer remote NUMA access over expensive local reclaim stalls.
vm.zone_reclaim_mode = 0

# Keep writeback bursts moderate. Validate with checkpoint and storage benchmarks.
vm.dirty_ratio = ${dirty_ratio}
vm.dirty_background_ratio = ${dirty_background_ratio}

# High-throughput TCP defaults.
# Helpful for TCP storage, object-store traffic, serving frontends, and NCCL socket fallback.
# Native RDMA verbs performance must be validated separately with ib_*_bw and nccl-tests.
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.core.netdev_max_backlog = 250000
EOF_SYSCTL
)"

  write_file "${SYSCTL_CONF}" "${content}"
  run sysctl --system
}

configure_thp() {
  local thp_mode thp_defrag

  if [[ "${PROFILE}" == "training" ]]; then
    thp_mode="madvise"
    thp_defrag="madvise"
  else
    thp_mode="never"
    thp_defrag="never"
  fi

  log "Configuring Transparent Huge Pages: enabled=${thp_mode}, defrag=${thp_defrag}"

  if [[ -w /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    echo "${thp_mode}" > /sys/kernel/mm/transparent_hugepage/enabled || true
  fi
  if [[ -w /sys/kernel/mm/transparent_hugepage/defrag ]]; then
    echo "${thp_defrag}" > /sys/kernel/mm/transparent_hugepage/defrag || true
  fi

  local content
  content="$(cat <<EOF_THP
# Managed by ${SCRIPT_NAME}
w /sys/kernel/mm/transparent_hugepage/enabled - - - - ${thp_mode}
w /sys/kernel/mm/transparent_hugepage/defrag - - - - ${thp_defrag}
EOF_THP
)"
  write_file "${TMPFILES_CONF}" "${content}"
}

configure_limits() {
  local limits_content systemd_content

  limits_content="$(cat <<'EOF_LIMITS'
# Managed by gb_grace_blackwell_numa_tuning.sh
# Useful for RDMA/UCX/NCCL/GPUDirect RDMA and pinned-memory-heavy workloads.
* soft memlock unlimited
* hard memlock unlimited
root soft memlock unlimited
root hard memlock unlimited

# Useful for high fan-in inference servers, many dataset shards, Ray, and logging-heavy jobs.
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576
EOF_LIMITS
)"
  write_file "${LIMITS_CONF}" "${limits_content}"

  systemd_content="$(cat <<'EOF_SYSTEMD'
# Managed by gb_grace_blackwell_numa_tuning.sh
[Manager]
DefaultLimitMEMLOCK=infinity
DefaultLimitNOFILE=1048576
EOF_SYSTEMD
)"
  write_file "${SYSTEMD_CONF}" "${systemd_content}"

  run systemctl daemon-reexec || true
}

set_cpu_governor_performance() {
  if [[ "${SET_CPU_GOVERNOR}" != "true" ]]; then
    log "Skipping CPU governor tuning"
    return 0
  fi

  log "Setting CPU governor to performance"

  local changed="false"
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -w "${gov}" ]] || continue
    echo performance > "${gov}" || true
    changed="true"
  done

  if [[ "${changed}" == "false" ]]; then
    if command -v cpupower >/dev/null 2>&1; then
      run cpupower frequency-set -g performance || true
    elif command -v tuned-adm >/dev/null 2>&1; then
      run tuned-adm profile throughput-performance || true
    else
      warn "CPU governor interface, cpupower, and tuned-adm are unavailable. Skipping."
    fi
  fi
}

configure_swap() {
  if [[ "${DISABLE_SWAP}" != "true" ]]; then
    log "Swap disable was not requested. Keeping current swap devices."
    return 0
  fi

  log "Disabling swap"
  run swapoff -a

  if grep -qE '^[^#].*\sswap\s' /etc/fstab 2>/dev/null; then
    cp /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"
    sed -i '/^[^#].*\sswap\s/s/^/# gb-node-tuning disabled swap: /' /etc/fstab
    log "Swap entries commented in /etc/fstab"
  fi
}

configure_gpu_persistence() {
  if [[ "${SET_GPU_PERSISTENCE}" != "true" ]]; then
    log "Skipping NVIDIA persistence mode"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found. Skipping persistence mode."
    return 0
  fi

  log "Enabling NVIDIA persistence mode"
  run nvidia-smi -pm 1 || true
}

configure_app_clocks_if_requested() {
  if [[ "${SET_APP_CLOCKS}" != "true" ]]; then
    log "Application clocks not requested. Skipping."
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found. Skipping application clocks."
    return 0
  fi

  warn "Setting application clocks can affect power, thermal behavior, fairness, and scheduler policy."
  warn "Proceeding because --set-app-clocks was explicitly requested."

  mkdir -p "${STATE_DIR}"
  date > "${STATE_DIR}/app_clocks_applied"

  local gpu mem sm clocks
  while read -r gpu; do
    [[ -n "${gpu}" ]] || continue

    if [[ "${APP_CLOCKS}" == "auto" ]]; then
      clocks="$(nvidia-smi -i "${gpu}" --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    else
      clocks="${APP_CLOCKS}"
    fi

    if [[ "${clocks}" =~ ^([0-9]+),([0-9]+)$ ]]; then
      mem="${BASH_REMATCH[1]}"
      sm="${BASH_REMATCH[2]}"
      log "Setting GPU ${gpu} application clocks to memory=${mem}, sm=${sm}"
      nvidia-smi -i "${gpu}" -ac "${mem},${sm}" || warn "Failed to set app clocks on GPU ${gpu}"
    else
      warn "Could not determine valid app clocks for GPU ${gpu}: '${clocks}'"
    fi
  done < <(get_gpu_indices)
}

create_runtime_boot_service() {
  if [[ "${CREATE_BOOT_SERVICE}" != "true" ]]; then
    log "Boot-time runtime tuning service not requested. Skipping."
    return 0
  fi

  local thp_mode thp_defrag
  if [[ "${PROFILE}" == "training" ]]; then
    thp_mode="madvise"
    thp_defrag="madvise"
  else
    thp_mode="never"
    thp_defrag="never"
  fi

  local runtime_content service_content
  runtime_content="$(cat <<EOF_RUNTIME
#!/usr/bin/env bash
set -euo pipefail

# Managed by ${SCRIPT_NAME}
# Applies runtime-only settings that do not persist through sysctl.d/limits.d.

if [[ "${SET_CPU_GOVERNOR}" == "true" ]]; then
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -w "\${gov}" ]] || continue
    echo performance > "\${gov}" || true
  done
fi

if [[ -w /sys/kernel/mm/transparent_hugepage/enabled ]]; then
  echo ${thp_mode} > /sys/kernel/mm/transparent_hugepage/enabled || true
fi
if [[ -w /sys/kernel/mm/transparent_hugepage/defrag ]]; then
  echo ${thp_defrag} > /sys/kernel/mm/transparent_hugepage/defrag || true
fi

if [[ "${SET_GPU_PERSISTENCE}" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -pm 1 || true
fi
EOF_RUNTIME
)"
  write_file "${RUNTIME_SCRIPT}" "${runtime_content}"
  chmod +x "${RUNTIME_SCRIPT}"

  service_content="$(cat <<EOF_SERVICE
# Managed by ${SCRIPT_NAME}
[Unit]
Description=Grace-Blackwell GPU node runtime tuning
After=multi-user.target nvidia-persistenced.service
ConditionPathExists=${RUNTIME_SCRIPT}

[Service]
Type=oneshot
ExecStart=${RUNTIME_SCRIPT}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF_SERVICE
)"
  write_file "${RUNTIME_SERVICE}" "${service_content}"

  run systemctl daemon-reload
  run systemctl enable --now gb-node-runtime-tuning.service || true
}

create_numa_helper() {
  local helper_content
  helper_content="$(cat <<'EOF_HELPER'
#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <gpu_id> <command> [args...]" >&2
  echo "Example: $(basename "$0") 0 python train.py" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

GPU_ID="$1"
shift

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; running without NUMA binding" >&2
  exec "$@"
fi

if ! command -v numactl >/dev/null 2>&1; then
  echo "numactl not found; running without NUMA binding" >&2
  exec "$@"
fi

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "Invalid GPU ID: ${GPU_ID}" >&2
  usage
  exit 2
fi

normalize_pci_bus_id() {
  local raw="$1"
  local bus
  bus="$(echo "${raw}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  if [[ "${bus}" =~ ^([0-9a-f]{8}):(.+)$ ]]; then
    local domain="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    bus="${domain:4}:${rest}"
  fi
  echo "${bus}"
}

PCI_RAW="$(nvidia-smi -i "${GPU_ID}" --query-gpu=pci.bus_id --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"

if [[ -z "${PCI_RAW}" ]]; then
  echo "Could not query PCI bus for GPU ${GPU_ID}; running without NUMA binding" >&2
  exec "$@"
fi

PCI_SYSFS="$(normalize_pci_bus_id "${PCI_RAW}")"
NUMA_PATH="/sys/bus/pci/devices/${PCI_SYSFS}/numa_node"
NUMA_NODE=""

if [[ -r "${NUMA_PATH}" ]]; then
  NUMA_NODE="$(cat "${NUMA_PATH}" 2>/dev/null || true)"
fi

if [[ "${NUMA_NODE}" =~ ^[0-9]+$ ]]; then
  echo "Binding command to NUMA node ${NUMA_NODE} for GPU ${GPU_ID} (${PCI_SYSFS})" >&2
  exec numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" "$@"
fi

echo "GPU ${GPU_ID} NUMA node unavailable from ${NUMA_PATH}; running without NUMA binding" >&2
exec "$@"
EOF_HELPER
)"

  write_file "${NUMA_HELPER}" "${helper_content}"
  chmod +x "${NUMA_HELPER}"
}

apply_tuning() {
  require_root

  if ! detect_grace_blackwell; then
    if [[ "${FORCE}" != "true" ]]; then
      die "Grace-Blackwell detection failed. Re-run with --force only if you intentionally want to apply this baseline."
    fi
    warn "Proceeding because --force was provided."
  fi

  save_snapshot
  collect_topology_report

  configure_sysctl
  configure_thp
  configure_limits
  set_cpu_governor_performance
  configure_swap
  configure_gpu_persistence
  configure_app_clocks_if_requested
  create_runtime_boot_service
  create_numa_helper

  log "Tuning applied"
  log "Topology report: ${REPORT_FILE}"
  warn "Log out/in or reboot may be required for PAM/systemd limits to apply to all services."
  warn "IRQ affinity was not changed. Generate a topology-aware IRQ plan before changing IRQ affinity."
}

restore_value_to_path() {
  local state_name="$1"
  local target_path="$2"

  if [[ -f "${STATE_DIR}/${state_name}" && -w "${target_path}" ]]; then
    log "Restoring ${target_path} from ${state_name}"
    cat "${STATE_DIR}/${state_name}" > "${target_path}" || true
  fi
}

reset_tuning() {
  require_root

  log "Resetting settings from ${STATE_DIR} where possible"

  if [[ -f "${RUNTIME_SERVICE}" ]]; then
    systemctl disable --now gb-node-runtime-tuning.service 2>/dev/null || true
    rm -f "${RUNTIME_SERVICE}"
  fi

  rm -f "${SYSCTL_CONF}" "${LIMITS_CONF}" "${SYSTEMD_CONF}" "${TMPFILES_CONF}" "${RUNTIME_SCRIPT}" "${NUMA_HELPER}"

  systemctl daemon-reload 2>/dev/null || true
  systemctl daemon-reexec 2>/dev/null || true
  sysctl --system >/dev/null 2>&1 || true

  restore_value_to_path kernel_numa_balancing /proc/sys/kernel/numa_balancing
  restore_value_to_path vm_swappiness /proc/sys/vm/swappiness
  restore_value_to_path vm_zone_reclaim_mode /proc/sys/vm/zone_reclaim_mode
  restore_value_to_path vm_dirty_ratio /proc/sys/vm/dirty_ratio
  restore_value_to_path vm_dirty_background_ratio /proc/sys/vm/dirty_background_ratio

  if [[ -f "${STATE_DIR}/thp_enabled" && -w /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    local thp_enabled
    thp_enabled="$(cat "${STATE_DIR}/thp_enabled")"
    [[ -n "${thp_enabled}" ]] && echo "${thp_enabled}" > /sys/kernel/mm/transparent_hugepage/enabled || true
  fi
  if [[ -f "${STATE_DIR}/thp_defrag" && -w /sys/kernel/mm/transparent_hugepage/defrag ]]; then
    local thp_defrag
    thp_defrag="$(cat "${STATE_DIR}/thp_defrag")"
    [[ -n "${thp_defrag}" ]] && echo "${thp_defrag}" > /sys/kernel/mm/transparent_hugepage/defrag || true
  fi

  if [[ -f "${STATE_DIR}/cpu_governors.tsv" ]]; then
    while IFS=$'\t' read -r path value; do
      [[ -n "${path:-}" && -n "${value:-}" ]] || continue
      [[ -w "${path}" ]] || continue
      echo "${value}" > "${path}" || true
    done < "${STATE_DIR}/cpu_governors.tsv"
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && [[ -f "${STATE_DIR}/gpu_persistence.csv" ]]; then
    while IFS=',' read -r idx mode; do
      idx="$(echo "${idx}" | tr -d '[:space:]')"
      mode="$(echo "${mode}" | tr -d '[:space:]')"
      [[ "${idx}" =~ ^[0-9]+$ ]] || continue
      case "${mode}" in
        Enabled) nvidia-smi -i "${idx}" -pm 1 >/dev/null 2>&1 || true ;;
        Disabled) nvidia-smi -i "${idx}" -pm 0 >/dev/null 2>&1 || true ;;
      esac
    done < "${STATE_DIR}/gpu_persistence.csv"
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && [[ -f "${STATE_DIR}/app_clocks_applied" ]]; then
    warn "Resetting NVIDIA application clocks because this script previously applied them."
    nvidia-smi -rac >/dev/null 2>&1 || true
  fi

  log "Reset complete. Original snapshot kept at ${STATE_DIR} for audit."
}

print_recommendations() {
  cat <<'EOF_RECS'

Recommended validation plan:

Training:
  - step time p50/p95
  - DataLoader wait time
  - H2D copy time
  - GPU utilization and HBM bandwidth
  - NCCL all-reduce/all-gather bandwidth
  - checkpoint latency and writeback stalls

Inference:
  - TTFT
  - TPOT
  - tokens/sec
  - p95/p99 latency
  - CPU utilization and scheduling jitter
  - GPU utilization
  - KV cache memory pressure

Useful commands:
  nvidia-smi topo -m
  nvidia-smi dmon -s pucvmet -d 1
  dcgmi dmon
  mpstat -P ALL 1
  pidstat -t -p <PID> 1
  numastat -p <PID>
  iostat -x 1
  vmstat 1
  grep -Ei 'nvidia|mlx|ib|nvme' /proc/interrupts

NCCL/RDMA validation:
  ibstat
  ibv_devinfo
  ib_write_bw / ib_read_bw
  NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 4G -f 2 -g <num_gpus>

NUMA helper example:
  /usr/local/bin/numa_bind_gpu.sh 0 python train.py

Do not change IRQ affinity until you have a GPU/NIC/NUMA/CPU-set mapping.
EOF_RECS
}

main() {
  parse_args "$@"

  case "${MODE}" in
    check)
      detect_grace_blackwell || true
      echo
      collect_topology_report
      echo
      check_settings
      print_recommendations
      ;;
    apply)
      apply_tuning
      echo
      check_settings
      print_recommendations
      ;;
    reset)
      reset_tuning
      echo
      check_settings
      ;;
  esac
}

main "$@"
