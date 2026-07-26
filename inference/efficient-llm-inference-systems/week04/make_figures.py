"""Generate SVG figures for the Week 4 quantization results.

Reads the result CSVs in week04/results/ and emits hand-built SVGs in the same
visual language as the Week 3 charts (Inter font, slate palette, 960px wide).
Figures:
  1. kernel_decides.svg   - speedup vs BF16: bnb INT8/NF4 (slower) vs AWQ-INT4 (faster)
  2. bnb_mem_vs_latency.svg - Lab 1 bnb tradeoff: memory falls, latency rises
  3. orin_projection.svg   - projected Orin decode ms/token by precision
"""

import csv
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

FONT = "Inter, Arial, sans-serif"
INK = "#0f172a"
SUB = "#475569"
GRID = "#e5e7eb"
AXIS = "#0f172a"
GREEN = "#16a34a"
GREEN_D = "#14532d"
RED = "#dc2626"
RED_D = "#7f1d1d"
BLUE = "#2563eb"
BLUE_D = "#1e3a8a"
AMBER = "#d97706"


def read_csv(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def header(svg, w, title, subtitle):
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="540" viewBox="0 0 {w} 540">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<rect x="0" y="0" width="{w}" height="540" fill="#f8fafc"/>')
    svg.append(f'<text x="60" y="40" font-size="22" font-family="{FONT}" font-weight="700" fill="{INK}">{title}</text>')
    svg.append(f'<text x="60" y="64" font-size="13" font-family="{FONT}" fill="{SUB}">{subtitle}</text>')


def vbar_chart(values, *, w=960, title, subtitle, ylabel, ymax, baseline=None,
               fmt="{:.2f}", note=None):
    """values: list of (label, value, fill, stroke, caption)."""
    svg = []
    header(svg, w, title, subtitle)
    plot_l, plot_r, plot_t, plot_b = 90, w - 60, 100, 440
    pw, ph = plot_r - plot_l, plot_b - plot_t

    def y_of(v):
        return plot_b - (v / ymax) * ph

    # y gridlines
    steps = 5
    for i in range(steps + 1):
        v = ymax * i / steps
        y = y_of(v)
        svg.append(f'<line x1="{plot_l}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="{GRID}"/>')
        svg.append(f'<text x="{plot_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="{FONT}" fill="{SUB}">{fmt.format(v)}</text>')

    n = len(values)
    slot = pw / n
    bw = slot * 0.5
    for i, (label, val, fill, stroke, cap) in enumerate(values):
        cx = plot_l + slot * (i + 0.5)
        x = cx - bw / 2
        y = y_of(val)
        h = plot_b - y
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="3"/>')
        svg.append(f'<text x="{cx:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="14" font-weight="700" font-family="{FONT}" fill="{INK}">{fmt.format(val)}</text>')
        svg.append(f'<text x="{cx:.1f}" y="{plot_b+20:.1f}" text-anchor="middle" font-size="12" font-weight="600" font-family="{FONT}" fill="{INK}">{label}</text>')
        if cap:
            svg.append(f'<text x="{cx:.1f}" y="{plot_b+38:.1f}" text-anchor="middle" font-size="11" font-family="{FONT}" fill="{SUB}">{cap}</text>')

    # baseline line (e.g. 1.0x)
    if baseline is not None:
        yb = y_of(baseline)
        svg.append(f'<line x1="{plot_l}" y1="{yb:.1f}" x2="{plot_r}" y2="{yb:.1f}" stroke="{AMBER}" stroke-width="1.6" stroke-dasharray="7 4"/>')
        svg.append(f'<text x="{plot_r}" y="{yb-6:.1f}" text-anchor="end" font-size="11" font-family="{FONT}" fill="{AMBER}">BF16 baseline = {fmt.format(baseline)}</text>')

    # axes
    svg.append(f'<line x1="{plot_l}" y1="{plot_t}" x2="{plot_l}" y2="{plot_b}" stroke="{AXIS}" stroke-width="1.4"/>')
    svg.append(f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="{AXIS}" stroke-width="1.4"/>')
    svg.append(f'<text transform="translate(28 {(plot_t+plot_b)/2:.0f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="{FONT}" fill="{INK}">{ylabel}</text>')
    if note:
        svg.append(f'<text x="60" y="500" font-size="12" font-family="{FONT}" fill="{SUB}">{note}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def fig_kernel_decides():
    bnb = {r["variant"]: r for r in read_csv("quant_compare.csv")}
    vllm = {r["label"]: r for r in read_csv("vllm_quant_bench.csv")}
    int8 = float(bnb["int8"]["speedup_vs_bf16"])
    nf4 = float(bnb["nf4"]["speedup_vs_bf16"])
    awq = float(vllm["bf16"]["ms_per_gen"]) / float(vllm["awq_int4"]["ms_per_gen"])
    vals = [
        ("bnb INT8", int8, RED, RED_D, "HF generate"),
        ("bnb NF4 (4-bit)", nf4, RED, RED_D, "HF generate"),
        ("AWQ-INT4 (4-bit)", round(awq, 2), GREEN, GREEN_D, "vLLM Marlin"),
    ]
    return vbar_chart(
        vals, title="Same bits, opposite speed: the kernel decides",
        subtitle="Qwen2.5-3B on RTX 5080 (Blackwell) - decode speedup vs BF16, batch 1, 32 tokens. >1 = faster.",
        ylabel="Speedup vs BF16 (x)", ymax=2.5, baseline=1.0, fmt="{:.2f}x",
        note="Both 4-bit. bitsandbytes is a memory path (dequant overhead); AWQ+Marlin is a fused low-bit GEMM. Speedups are within each engine.",
    )


def fig_bnb_tradeoff():
    rows = read_csv("quant_compare.csv")
    w = 960
    svg = []
    header(svg, w, "bitsandbytes: memory drops, latency rises",
           "Qwen2.5-3B on RTX 5080. Fewer weight bytes do NOT mean faster decode without a fused kernel.")
    plot_l, plot_r, plot_t, plot_b = 90, w - 90, 100, 430
    pw, ph = plot_r - plot_l, plot_b - plot_t
    lat_max, mem_max = 2400.0, 7.0

    for i in range(6):
        v = lat_max * i / 5
        y = plot_b - (v / lat_max) * ph
        svg.append(f'<line x1="{plot_l}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="{GRID}"/>')
        svg.append(f'<text x="{plot_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="{FONT}" fill="{RED}">{v:.0f}</text>')
        mv = mem_max * i / 5
        svg.append(f'<text x="{plot_r+10}" y="{y+4:.1f}" text-anchor="start" font-size="11" font-family="{FONT}" fill="{BLUE}">{mv:.1f}</text>')

    n = len(rows)
    slot = pw / n
    bw = slot * 0.22
    lat_pts, mem_pts = [], []
    for i, r in enumerate(rows):
        cx = plot_l + slot * (i + 0.5)
        lat = float(r["ms_per_gen"]); mem = float(r["peak_mem_gb"])
        ly = plot_b - (lat / lat_max) * ph
        my = plot_b - (mem / mem_max) * ph
        # latency bar (red)
        svg.append(f'<rect x="{cx-bw-3:.1f}" y="{ly:.1f}" width="{bw:.1f}" height="{plot_b-ly:.1f}" fill="{RED}" stroke="{RED_D}" stroke-width="1.2" rx="2"/>')
        svg.append(f'<text x="{cx-bw/2-3:.1f}" y="{ly-8:.1f}" text-anchor="middle" font-size="11" font-weight="700" font-family="{FONT}" fill="{RED_D}">{lat:.0f}</text>')
        # memory bar (blue)
        svg.append(f'<rect x="{cx+3:.1f}" y="{my:.1f}" width="{bw:.1f}" height="{plot_b-my:.1f}" fill="{BLUE}" stroke="{BLUE_D}" stroke-width="1.2" rx="2"/>')
        svg.append(f'<text x="{cx+bw/2+3:.1f}" y="{my-8:.1f}" text-anchor="middle" font-size="11" font-weight="700" font-family="{FONT}" fill="{BLUE_D}">{mem:.2f}</text>')
        svg.append(f'<text x="{cx:.1f}" y="{plot_b+20:.1f}" text-anchor="middle" font-size="12" font-weight="600" font-family="{FONT}" fill="{INK}">{r["variant"]}</text>')
        lat_pts.append((cx, ly)); mem_pts.append((cx, my))

    svg.append(f'<line x1="{plot_l}" y1="{plot_t}" x2="{plot_l}" y2="{plot_b}" stroke="{AXIS}" stroke-width="1.4"/>')
    svg.append(f'<line x1="{plot_r}" y1="{plot_t}" x2="{plot_r}" y2="{plot_b}" stroke="{AXIS}" stroke-width="1.4"/>')
    svg.append(f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="{AXIS}" stroke-width="1.4"/>')
    svg.append(f'<text transform="translate(28 {(plot_t+plot_b)/2:.0f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="{FONT}" fill="{RED}">Latency (ms / 32-token gen)</text>')
    svg.append(f'<text transform="translate({w-24} {(plot_t+plot_b)/2:.0f}) rotate(90)" text-anchor="middle" font-size="13" font-family="{FONT}" fill="{BLUE}">Peak memory (GB)</text>')
    # legend
    svg.append(f'<rect x="{plot_l+10}" y="{plot_t+6}" width="14" height="14" fill="{RED}"/><text x="{plot_l+30}" y="{plot_t+18}" font-size="12" font-family="{FONT}" fill="{SUB}">latency (lower=better)</text>')
    svg.append(f'<rect x="{plot_l+200}" y="{plot_t+6}" width="14" height="14" fill="{BLUE}"/><text x="{plot_l+220}" y="{plot_t+18}" font-size="12" font-family="{FONT}" fill="{SUB}">memory (lower=better)</text>')
    svg.append(f'<text x="60" y="490" font-size="12" font-family="{FONT}" fill="{SUB}">Memory shrinks 5.76 -> 1.98 GB as predicted, but latency moves the wrong way: INT8 +394%, NF4 +74%.</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def fig_orin():
    rows = read_csv("orin_projection.csv")
    vals = []
    fills = {"BF16": (BLUE, BLUE_D), "INT8": (AMBER, "#92400e"), "INT4": (GREEN, GREEN_D)}
    for r in rows:
        f, s = fills[r["precision"]]
        cap = f'{r["weight_gb"]}GB weights  -  {r["speedup_vs_bf16"]}x'
        vals.append((r["precision"], float(r["ms_per_token"]), f, s, cap))
    return vbar_chart(
        vals, title="Orin edge projection: weight bytes -> decode latency",
        subtitle="AGX Orin 64GB, 7B GQA, bandwidth-bound model calibrated to the Week 3 BF16 measurement (94 ms/token).",
        ylabel="Decode latency (ms / token)", ymax=100, baseline=None, fmt="{:.0f}",
        note="Bandwidth-bound regime: halving weight bytes ~halves decode latency. INT4 -> 4x faster decode, the regime where quantization pays off.",
    )


def main():
    figs = {
        "kernel_decides.svg": fig_kernel_decides(),
        "bnb_mem_vs_latency.svg": fig_bnb_tradeoff(),
        "orin_projection.svg": fig_orin(),
    }
    for name, content in figs.items():
        out = RESULTS / name
        out.write_text(content, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
