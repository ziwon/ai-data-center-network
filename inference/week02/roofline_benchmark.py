import torch
import time
import json
from statistics import median

device = torch.device('cuda')
dtype = torch.bfloat16

# RTX 5080 reference specs.
# This benchmark does not enable 2:4 sparsity, so the dense BF16 peak is the
# relevant roofline. The sparse peak is kept for comparison with NVIDIA specs.
PEAK_TFLOPS_BF16_DENSE = 112.6
PEAK_TFLOPS_BF16_SPARSE = 225.1
PEAK_BANDWIDTH_GBS = 960.0


def benchmark_matmul(M, N, K, n_iter=200, n_warmup=20, n_repeat=5):
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(n_warmup):
        C = A @ B
    torch.cuda.synchronize()

    gpu_times_us = []
    wall_times_us = []
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(n_iter):
            C = A @ B
        end.record()
        torch.cuda.synchronize()
        wall_elapsed = time.perf_counter() - wall_start

        gpu_times_us.append(start.elapsed_time(end) * 1000 / n_iter)
        wall_times_us.append(wall_elapsed * 1e6 / n_iter)

    flops = 2 * M * N * K
    analytic_bytes = 2 * (M*K + K*N + M*N)  # BF16 = 2 bytes

    gpu_time_us = median(gpu_times_us)
    achieved_tflops = flops / (gpu_time_us * 1e-6) / 1e12
    effective_bw = analytic_bytes / (gpu_time_us * 1e-6) / 1e9
    arithmetic_intensity = flops / analytic_bytes

    return {
        'M': M, 'N': N, 'K': K,
        'flops': flops,
        'analytic_bytes_bf16': analytic_bytes,
        'gpu_time_us_median': gpu_time_us,
        'gpu_time_us_min': min(gpu_times_us),
        'gpu_time_us_max': max(gpu_times_us),
        'wall_time_us_median': median(wall_times_us),
        'wall_time_us_min': min(wall_times_us),
        'wall_time_us_max': max(wall_times_us),
        # Backward-compatible aliases used by plot_roofline.py and README tables.
        'time_us': gpu_time_us,
        'tflops': achieved_tflops,
        'effective_bw_gbs': effective_bw,
        'bw_gbs': effective_bw,
        'ai': arithmetic_intensity,
        'tflops_pct_dense': achieved_tflops / PEAK_TFLOPS_BF16_DENSE * 100,
        'tflops_pct_sparse': achieved_tflops / PEAK_TFLOPS_BF16_SPARSE * 100,
        'tflops_pct': achieved_tflops / PEAK_TFLOPS_BF16_DENSE * 100,
        'effective_bw_pct': effective_bw / PEAK_BANDWIDTH_GBS * 100,
        'bw_pct': effective_bw / PEAK_BANDWIDTH_GBS * 100,
        'n_iter': n_iter,
        'n_repeat': n_repeat,
    }

# Qwen2.5-3B의 실제 FFN shape: d_model=2048, d_ffn=11008 (대략)
# 정확한 값은 model.config 확인
D_MODEL = 2048
D_FFN = 11008

# Batch (= M) sweep
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
results = []

print(
    f"{'M':>6} {'gpu_us':>10} {'wall_us':>10} {'TFLOPS':>8} "
    f"{'%dense':>7} {'eff GB/s':>9} {'%mem':>7} {'AI':>8}"
)
for M in batches:
    r = benchmark_matmul(M, D_FFN, D_MODEL)  # gate_proj/up_proj shape
    results.append(r)
    print(
        f"{M:>6d} {r['gpu_time_us_median']:>10.1f} "
        f"{r['wall_time_us_median']:>10.1f} {r['tflops']:>8.1f} "
        f"{r['tflops_pct_dense']:>6.1f}% {r['effective_bw_gbs']:>9.1f} "
        f"{r['effective_bw_pct']:>6.1f}% {r['ai']:>8.1f}"
    )

# Save for plotting
with open('roofline_data.json', 'w') as f:
    json.dump({
        'metadata': {
            'device': torch.cuda.get_device_name(0),
            'dtype': str(dtype).replace('torch.', ''),
            'timing': 'CUDA event median over repeated PyTorch matmul loops',
            'peak_tflops_bf16_dense': PEAK_TFLOPS_BF16_DENSE,
            'peak_tflops_bf16_sparse': PEAK_TFLOPS_BF16_SPARSE,
            'peak_bandwidth_gbs': PEAK_BANDWIDTH_GBS,
            'd_model': D_MODEL,
            'd_ffn': D_FFN,
        },
        'results': results,
    }, f, indent=2)
