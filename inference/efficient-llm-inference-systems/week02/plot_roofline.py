import json
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

with open('roofline_data.json') as f:
    data = json.load(f)

if isinstance(data, dict):
    metadata = data.get('metadata', {})
    results = data['results']
else:
    metadata = {}
    results = data

PEAK_TFLOPS = metadata.get('peak_tflops_bf16_dense', 112.6)
SPARSE_PEAK_TFLOPS = metadata.get('peak_tflops_bf16_sparse', 225.1)
PEAK_BANDWIDTH_GBS = metadata.get('peak_bandwidth_gbs', 960.0)
RIDGE_AI = PEAK_TFLOPS * 1000 / PEAK_BANDWIDTH_GBS  # FLOP/byte

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 7.5))

# Roofline ceilings
ai_range = np.logspace(0, 4, 100)
memory_bound = PEAK_BANDWIDTH_GBS * ai_range / 1000  # TFLOPS
compute_bound = np.full_like(ai_range, PEAK_TFLOPS)
roofline = np.minimum(memory_bound, compute_bound)
memory_mask = ai_range <= RIDGE_AI
compute_mask = ai_range >= RIDGE_AI

ax.plot(
    ai_range[memory_mask],
    roofline[memory_mask],
    color='#2563eb',
    linewidth=3.0,
    label=f'Memory ceiling ({PEAK_BANDWIDTH_GBS:.0f} GB/s)',
)
ax.plot(
    ai_range[compute_mask],
    compute_bound[compute_mask],
    color='#7c3aed',
    linewidth=3.0,
    label=f'BF16 dense peak ({PEAK_TFLOPS:.1f} TFLOPS)',
)
ax.axhline(
    SPARSE_PEAK_TFLOPS,
    color='#a78bfa',
    linestyle=':',
    linewidth=1.8,
    alpha=0.85,
    label=f'BF16 sparse spec ({SPARSE_PEAK_TFLOPS:.1f} TFLOPS)',
)
ax.axvline(
    RIDGE_AI,
    color='#6b7280',
    linestyle='--',
    linewidth=1.5,
    alpha=0.75,
    label=f'Ridge point (AI={RIDGE_AI:.0f})',
)

# Measurements
ais = [r['ai'] for r in results]
tflops = [r['tflops'] for r in results]
labels = [f"M={r['M']}" for r in results]

ax.plot(ais, tflops, color='#dc2626', linewidth=1.8, alpha=0.75, zorder=4)
ax.scatter(
    ais,
    tflops,
    s=84,
    color='#dc2626',
    edgecolor='white',
    linewidth=1.4,
    zorder=5,
    label='Measured BF16 matmul',
)

label_offsets = {
    1: (8, 8),
    2: (8, 7),
    4: (8, 6),
    8: (8, 6),
    16: (8, 6),
    32: (8, 7),
    64: (8, 8),
    128: (8, 8),
    256: (8, 7),
    512: (8, 8),
    1024: (8, -18),
    2048: (8, 8),
}
for r, ai, tf, lbl in zip(results, ais, tflops, labels):
    ax.annotate(
        lbl,
        (ai, tf),
        textcoords='offset points',
        xytext=label_offsets.get(r['M'], (6, 6)),
        fontsize=9,
        color='#111827',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
    )

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
ax.set_ylabel('Achieved Performance (TFLOPS)')
ax.set_title(
    'Roofline: BF16 Matmul for Qwen2.5-3B FFN Shape on RTX 5080',
    fontsize=15,
    pad=14,
)
ax.set_xlim(0.75, 1.6e4)
ax.set_ylim(0.35, 330)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which='major', color='#9ca3af', alpha=0.32, linewidth=0.8)
ax.grid(True, which='minor', color='#9ca3af', alpha=0.18, linewidth=0.5)
ax.legend(loc='lower right', frameon=True, framealpha=0.94)
for spine in ax.spines.values():
    spine.set_color('#374151')
    spine.set_linewidth(0.9)

plt.tight_layout()
plt.savefig('roofline.png', dpi=200)
print("Saved roofline.png")
