import argparse
import csv
import gc
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TEST_POINTS = "128:768,256:768,128:1536,128:2048,64:4096,384:1024,256:2048"


def parse_test_points(value: str) -> list[tuple[int, int]]:
    points = []
    for item in value.split(","):
        if not item.strip():
            continue
        batch, seq_len = item.split(":")
        points.append((int(batch), int(seq_len)))
    return points


def parse_args():
    parser = argparse.ArgumentParser(description="Measure KV cache OOM boundary over batch x sequence length.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--test-points", default=DEFAULT_TEST_POINTS, help="Comma-separated batch:seq_len pairs.")
    parser.add_argument(
        "--mode",
        choices=["allocate", "forward"],
        default="allocate",
        help=(
            "allocate: directly allocate synthetic K/V tensors to isolate KV capacity. "
            "forward: run full model prefill, which also includes attention/temp allocations."
        ),
    )
    parser.add_argument("--out", default="week03/results/oom_boundary.csv")
    parser.add_argument("--svg", default="week03/results/oom_boundary.svg")
    return parser.parse_args()


def make_input_ids(tokenizer, batch: int, seq_len: int, device: str) -> dict[str, torch.Tensor]:
    seed = (
        "KV cache memory grows with batch size and sequence length. "
        "This synthetic prompt is repeated to create a fixed-length input. "
    )
    token_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = max(1, (seq_len + len(token_ids) - 1) // len(token_ids))
    token_ids = (token_ids * repeats)[:seq_len]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device).repeat(batch, 1)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def model_kv_config(model_id: str) -> dict[str, int]:
    config = AutoConfig.from_pretrained(model_id)
    n_layer = config.num_hidden_layers
    n_q_head = config.num_attention_heads
    n_kv_head = getattr(config, "num_key_value_heads", n_q_head)
    d_head = getattr(config, "head_dim", None) or config.hidden_size // n_q_head
    return {
        "layers": n_layer,
        "query_heads": n_q_head,
        "kv_heads": n_kv_head,
        "head_dim": d_head,
        "bytes_per_token": 2 * n_layer * n_kv_head * d_head * 2,
    }


def iter_cache_tensors(cache):
    """Yield tensors from both legacy tuple caches and Transformers DynamicCache."""
    if cache is None:
        return
    if isinstance(cache, torch.Tensor):
        yield cache
        return

    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if isinstance(tensor, torch.Tensor):
                    yield tensor
        return

    if isinstance(cache, dict):
        for value in cache.values():
            yield from iter_cache_tensors(value)
        return

    if isinstance(cache, (list, tuple)):
        for value in cache:
            yield from iter_cache_tensors(value)


def measure_allocate_point(
    batch: int,
    seq_len: int,
    kv_config: dict[str, int],
) -> dict[str, int | float | str]:
    gc.collect()
    torch.cuda.empty_cache()

    tensors = []
    bytes_per_token = kv_config["bytes_per_token"]
    theoretical_mib = bytes_per_token * batch * seq_len / 1024**2

    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(kv_config["layers"]):
            tensors.append(
                torch.empty(
                    (batch, kv_config["kv_heads"], seq_len, kv_config["head_dim"]),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            )
            tensors.append(
                torch.empty(
                    (batch, kv_config["kv_heads"], seq_len, kv_config["head_dim"]),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            )
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        kv_total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return {
            "batch": batch,
            "seq_len": seq_len,
            "token_positions": batch * seq_len,
            "mode": "allocate",
            "status": "ok",
            "peak_mib": peak_mib,
            "kv_mib": kv_total / 1024**2,
            "theoretical_kv_mib": theoretical_mib,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "batch": batch,
            "seq_len": seq_len,
            "token_positions": batch * seq_len,
            "mode": "allocate",
            "status": "oom",
            "peak_mib": 0.0,
            "kv_mib": 0.0,
            "theoretical_kv_mib": theoretical_mib,
        }
    finally:
        del tensors
        torch.cuda.empty_cache()


def measure_forward_point(model, tokenizer, batch: int, seq_len: int, bytes_per_token: int) -> dict[str, int | float | str]:
    gc.collect()
    torch.cuda.empty_cache()

    inputs = None
    out = None
    actual_seq_len = seq_len
    theoretical_mib = bytes_per_token * batch * seq_len / 1024**2

    try:
        inputs = make_input_ids(tokenizer, batch, seq_len, "cuda")
        actual_seq_len = inputs["input_ids"].shape[1]
        theoretical_mib = bytes_per_token * batch * actual_seq_len / 1024**2

        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            out = model(**inputs, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2

        kv_total = sum(tensor.numel() * tensor.element_size() for tensor in iter_cache_tensors(out.past_key_values))
        return {
            "batch": batch,
            "seq_len": actual_seq_len,
            "token_positions": batch * actual_seq_len,
            "mode": "forward",
            "status": "ok",
            "peak_mib": peak_mib,
            "kv_mib": kv_total / 1024**2,
            "theoretical_kv_mib": theoretical_mib,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "batch": batch,
            "seq_len": actual_seq_len,
            "token_positions": batch * actual_seq_len,
            "mode": "forward",
            "status": "oom",
            "peak_mib": 0.0,
            "kv_mib": 0.0,
            "theoretical_kv_mib": theoretical_mib,
        }
    finally:
        del out, inputs
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, int | float | str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict[str, int | float | str]]):
    width, height = 960, 640
    left, right, top, bottom = 100, 170, 76, 92
    batches = [float(r["batch"]) for r in rows]
    seqs = [float(r["seq_len"]) for r in rows]
    x_min, x_max = min(batches), max(batches)
    y_min, y_max = min(seqs), max(seqs)
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min = max(0, x_min - x_pad)
    x_max = x_max + x_pad
    y_min = max(0, y_min - y_pad)
    y_max = y_max + y_pad
    plot_right = width - right
    plot_bottom = height - bottom
    plot_width = plot_right - left
    plot_height = plot_bottom - top

    def scale(value, src_min, src_max, dst_min, dst_max):
        if src_max == src_min:
            return (dst_min + dst_max) / 2
        return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)

    def x_scale(value: float) -> float:
        return scale(value, x_min, x_max, left, plot_right)

    def y_scale(value: float) -> float:
        return scale(value, y_min, y_max, plot_bottom, top)

    def fmt_mib(value: float) -> str:
        if value >= 1024:
            return f"{value / 1024:.1f} GiB"
        return f"{value:.0f} MiB"

    x_ticks = sorted(set(int(v) for v in batches))
    y_ticks = sorted(set(int(v) for v in seqs))

    grid = []
    for tick in x_ticks:
        x = x_scale(float(tick))
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 24}" text-anchor="middle" '
            f'font-size="12" font-family="Inter, Arial, sans-serif" fill="#374151">{tick}</text>'
        )
    for tick in y_ticks:
        y = y_scale(float(tick))
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" font-family="Inter, Arial, sans-serif" fill="#374151">{tick}</text>'
        )

    bytes_per_token = rows[0]["theoretical_kv_mib"] * 1024**2 / rows[0]["token_positions"]
    contour_values = sorted({int(r["token_positions"]) for r in rows if r["status"] == "ok"})
    contours = []
    for token_positions in contour_values:
        x_start = max(x_min, token_positions / y_max)
        x_end = min(x_max, token_positions / max(y_min, 1))
        if x_start >= x_end:
            continue
        y_start = token_positions / x_start
        y_end = token_positions / x_end
        x1, y1 = x_scale(x_start), y_scale(y_start)
        x2, y2 = x_scale(x_end), y_scale(y_end)
        kv_mib = bytes_per_token * token_positions / 1024**2
        contours.append(
            f'<path d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="6 5" fill="none"/>'
        )
        label_x = x_scale(min(x_end, x_max - (x_max - x_min) * 0.10))
        label_y = y_scale(token_positions / min(x_end, x_max - (x_max - x_min) * 0.10))
        contours.append(
            f'<text x="{label_x + 4:.1f}" y="{label_y - 4:.1f}" '
            f'font-size="11" font-family="Inter, Arial, sans-serif" fill="#64748b">'
            f'{fmt_mib(kv_mib)}</text>'
        )

    points = []
    for row in sorted(rows, key=lambda r: (r["status"] != "ok", r["token_positions"])):
        x = x_scale(float(row["batch"]))
        y = y_scale(float(row["seq_len"]))
        ok = row["status"] == "ok"
        color = "#16a34a" if ok else "#dc2626"
        stroke = "#14532d" if ok else "#7f1d1d"
        label = (
            f'b={row["batch"]}, s={row["seq_len"]}\\n'
            f'KV={fmt_mib(float(row["theoretical_kv_mib"]))}'
        )
        marker = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>'
            if ok
            else (
                f'<path d="M {x - 8:.1f} {y - 8:.1f} L {x + 8:.1f} {y + 8:.1f} '
                f'M {x + 8:.1f} {y - 8:.1f} L {x - 8:.1f} {y + 8:.1f}" '
                f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
            )
        )
        points.append(
            f'<g><title>{label}</title>{marker}'
            f'<text x="{x + 12:.1f}" y="{y - 10:.1f}" '
            f'font-size="11" font-family="Inter, Arial, sans-serif" fill="#111827">'
            f'b{row["batch"]}/s{row["seq_len"]}</text></g>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1"/>
<text x="{left}" y="32" font-size="22" font-family="Inter, Arial, sans-serif" font-weight="700" fill="#0f172a">KV Cache OOM Boundary</text>
<text x="{left}" y="55" font-size="13" font-family="Inter, Arial, sans-serif" fill="#475569">Qwen2.5-3B BF16 synthetic KV allocation. Dashed lines are constant KV-memory contours.</text>
{''.join(grid)}
{''.join(contours)}
<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#0f172a" stroke-width="1.4"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="#0f172a" stroke-width="1.4"/>
<text x="{(left + plot_right) / 2}" y="{height - 34}" text-anchor="middle" font-size="14" font-family="Inter, Arial, sans-serif" fill="#111827">Batch size</text>
<text transform="translate(28 {(top + plot_bottom) / 2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Inter, Arial, sans-serif" fill="#111827">Sequence length</text>
{''.join(points)}
<g transform="translate({plot_right + 24} {top + 8})" font-family="Inter, Arial, sans-serif">
  <text x="0" y="0" font-size="13" font-weight="700" fill="#0f172a">Legend</text>
  <circle cx="8" cy="24" r="7" fill="#16a34a" stroke="#14532d" stroke-width="1.5"/>
  <text x="24" y="28" font-size="12" fill="#334155">OK</text>
  <path d="M 1 49 L 15 63 M 15 49 L 1 63" stroke="#dc2626" stroke-width="3" stroke-linecap="round"/>
  <text x="24" y="60" font-size="12" fill="#334155">OOM</text>
  <line x1="0" y1="85" x2="40" y2="85" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="6 5"/>
  <text x="0" y="105" font-size="12" fill="#334155">constant KV</text>
  <text x="0" y="138" font-size="12" fill="#475569">Rule of thumb:</text>
  <text x="0" y="157" font-size="12" fill="#475569">KV grows with</text>
  <text x="0" y="176" font-size="12" fill="#475569">batch x seq_len.</text>
</g>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OOM boundary measurement.")

    kv_config = model_kv_config(args.model)
    bytes_per_token = kv_config["bytes_per_token"]
    tokenizer = None
    model = None
    if args.mode == "forward":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        model.eval()

    rows = []
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"KV/token: {bytes_per_token / 1024:.1f} KiB BF16")
    print(f"{'batch':>6} {'seq':>7} {'mode':>9} {'status':>7} {'peak MiB':>10} {'kv MiB':>10} {'theory MiB':>11}")

    for batch, seq_len in parse_test_points(args.test_points):
        if args.mode == "allocate":
            row = measure_allocate_point(batch, seq_len, kv_config)
        else:
            row = measure_forward_point(model, tokenizer, batch, seq_len, bytes_per_token)
        rows.append(row)
        print(
            f"{row['batch']:>6} {row['seq_len']:>7} {row['mode']:>9} {row['status']:>7} "
            f"{row['peak_mib']:>10.0f} {row['kv_mib']:>10.0f} {row['theoretical_kv_mib']:>11.0f}"
        )

    write_csv(Path(args.out), rows)
    write_svg(Path(args.svg), rows)
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()
