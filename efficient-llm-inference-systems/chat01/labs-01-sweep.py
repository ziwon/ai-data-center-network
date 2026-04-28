import argparse
import csv
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_LENGTHS = [16, 256, 1024, 4096]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


class DmonSampler:
    def __init__(self, output_path: Path, interval: int = 1):
        self.output_path = output_path
        self.interval = interval
        self.proc = None

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", str(self.interval)],
            stdout=self.file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.2)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
        self.file.close()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_prompt_ids(tokenizer, target_len: int, device: str) -> torch.Tensor:
    seed = (
        "Explain why LLM decoding is often memory bandwidth bound. "
        "Discuss KV cache, arithmetic intensity, prefill, decode, batching, "
        "GPU utilization, and practical production inference tradeoffs. "
    )
    token_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = max(1, math.ceil(target_len / len(token_ids)))
    token_ids = (token_ids * repeats)[:target_len]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def make_batch(tokenizer, prompt_len: int, batch_size: int, device: str) -> dict[str, torch.Tensor]:
    input_ids = make_prompt_ids(tokenizer, prompt_len, device).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def synchronize_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_decode(model, tokenizer, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> dict[str, float]:
    timestamps = []
    batch_size = inputs["input_ids"].shape[0]

    with torch.inference_mode():
        past_key_values = None
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        synchronize_if_needed()
        t_start = time.perf_counter()

        for _ in range(max_new_tokens):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )

            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            synchronize_if_needed()
            timestamps.append(time.perf_counter())

            input_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )

    ttft_ms = (timestamps[0] - t_start) * 1000
    tpots_ms = np.array([timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]) * 1000
    generated_tokens = len(timestamps)
    total_tps = batch_size * generated_tokens / (timestamps[-1] - t_start)
    aggregate_tps = batch_size * (generated_tokens - 1) / (timestamps[-1] - timestamps[0]) if generated_tokens > 1 else 0.0

    return {
        "generated_tokens": generated_tokens,
        "ttft_ms": ttft_ms,
        "tpot_mean_ms": float(tpots_ms.mean()) if len(tpots_ms) else 0.0,
        "tpot_p50_ms": float(np.percentile(tpots_ms, 50)) if len(tpots_ms) else 0.0,
        "tpot_p95_ms": float(np.percentile(tpots_ms, 95)) if len(tpots_ms) else 0.0,
        "tpot_p99_ms": float(np.percentile(tpots_ms, 99)) if len(tpots_ms) else 0.0,
        "aggregate_tps": aggregate_tps,
        "total_tps_including_ttft": total_tps,
    }


def read_dmon_averages(path: Path) -> dict[str, float]:
    samples = {"pwr_w": [], "gpu_util_pct": [], "mem_util_pct": [], "enc_pct": [], "dec_pct": []}
    if not path.exists():
        return {key: 0.0 for key in samples}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            samples["pwr_w"].append(float(parts[1]))
            samples["gpu_util_pct"].append(float(parts[4]))
            samples["mem_util_pct"].append(float(parts[5]))
            samples["enc_pct"].append(float(parts[6]))
            samples["dec_pct"].append(float(parts[7]))
        except ValueError:
            continue

    return {key: (statistics.fmean(values) if values else 0.0) for key, values in samples.items()}


def write_csv(path: Path, rows: list[dict[str, float | int | str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "\n".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" />' for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="2.5" />\n{circles}'


def write_prompt_svg(path: Path, rows: list[dict[str, float | int | str]]):
    width, height = 980, 480
    left, right, top, bottom = 80, 40, 40, 70
    xs = [float(row["prompt_tokens"]) for row in rows]
    ttft = [float(row["ttft_ms"]) for row in rows]
    tpot = [float(row["tpot_mean_ms"]) for row in rows]
    x_min, x_max = math.log2(min(xs)), math.log2(max(xs))
    y_max = max(max(ttft), max(tpot)) * 1.15

    def x_pos(x):
        return scale(math.log2(x), x_min, x_max, left, width - right)

    def y_pos(y):
        return scale(y, 0, y_max, height - bottom, top)

    ttft_points = [(x_pos(x), y_pos(y)) for x, y in zip(xs, ttft)]
    tpot_points = [(x_pos(x), y_pos(y)) for x, y in zip(xs, tpot)]
    x_ticks = "\n".join(
        f'<text x="{x_pos(x):.1f}" y="{height - 35}" text-anchor="middle">{int(x)}</text>'
        f'<line x1="{x_pos(x):.1f}" y1="{height - bottom}" x2="{x_pos(x):.1f}" y2="{height - bottom + 6}" stroke="#555" />'
        for x in xs
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">Prompt length sweep</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" />
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-family="sans-serif">prompt tokens, log2 scale</text>
<text transform="translate(18,{height / 2}) rotate(-90)" text-anchor="middle" font-size="13" font-family="sans-serif">milliseconds</text>
{x_ticks}
{polyline(ttft_points, "#2563eb")}
{polyline(tpot_points, "#dc2626")}
<text x="{width - 210}" y="58" font-size="13" font-family="sans-serif" fill="#2563eb">TTFT ms</text>
<text x="{width - 210}" y="80" font-size="13" font-family="sans-serif" fill="#dc2626">TPOT mean ms</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_batch_svg(path: Path, rows: list[dict[str, float | int | str]]):
    width, height = 1100, 520
    left, right, top, bottom = 80, 90, 40, 70
    xs = [float(row["batch_size"]) for row in rows]
    series = [
        ("TPOT mean ms", [float(row["tpot_mean_ms"]) for row in rows], "#dc2626"),
        ("Throughput tok/s", [float(row["aggregate_tps"]) for row in rows], "#16a34a"),
        ("GPU util %", [float(row["gpu_util_pct"]) for row in rows], "#2563eb"),
    ]
    x_min, x_max = math.log2(min(xs)), math.log2(max(xs))

    def x_pos(x):
        return scale(math.log2(x), x_min, x_max, left, width - right)

    x_ticks = "\n".join(
        f'<text x="{x_pos(x):.1f}" y="{height - 35}" text-anchor="middle">{int(x)}</text>'
        f'<line x1="{x_pos(x):.1f}" y1="{height - bottom}" x2="{x_pos(x):.1f}" y2="{height - bottom + 6}" stroke="#555" />'
        for x in xs
    )
    lines = []
    legend = []
    for idx, (name, values, color) in enumerate(series):
        y_max = max(values) * 1.15 if max(values) > 0 else 1

        def y_pos(y, local_max=y_max):
            return scale(y, 0, local_max, height - bottom, top)

        points = [(x_pos(x), y_pos(y)) for x, y in zip(xs, values)]
        lines.append(polyline(points, color))
        legend.append(f'<text x="{width - 230}" y="{58 + idx * 22}" font-size="13" font-family="sans-serif" fill="{color}">{name}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">Batch size sweep, independently normalized axes</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" />
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-family="sans-serif">batch size, log2 scale</text>
<text transform="translate(18,{height / 2}) rotate(-90)" text-anchor="middle" font-size="13" font-family="sans-serif">normalized per metric</text>
{x_ticks}
{''.join(lines)}
{''.join(legend)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt-lengths", default=",".join(map(str, PROMPT_LENGTHS)))
    parser.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    parser.add_argument("--batch-prompt-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-new-tokens", type=int, default=8)
    parser.add_argument("--out-dir", default="results/labs-01-sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    dmon_dir = out_dir / "dmon"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    warmup_inputs = make_batch(tokenizer, 16, 1, device)
    _ = run_decode(model, tokenizer, warmup_inputs, args.warmup_new_tokens)

    prompt_rows = []
    for prompt_len in parse_int_list(args.prompt_lengths):
        inputs = make_batch(tokenizer, prompt_len, 1, device)
        metrics = run_decode(model, tokenizer, inputs, args.max_new_tokens)
        row = {"experiment": "prompt_length", "prompt_tokens": prompt_len, "batch_size": 1, **metrics}
        prompt_rows.append(row)
        print(row)

    batch_rows = []
    for batch_size in parse_int_list(args.batch_sizes):
        inputs = make_batch(tokenizer, args.batch_prompt_len, batch_size, device)
        dmon_path = dmon_dir / f"batch_{batch_size}.log"
        with DmonSampler(dmon_path):
            metrics = run_decode(model, tokenizer, inputs, args.max_new_tokens)
        dmon_metrics = read_dmon_averages(dmon_path)
        row = {
            "experiment": "batch_size",
            "prompt_tokens": args.batch_prompt_len,
            "batch_size": batch_size,
            **metrics,
            **dmon_metrics,
            "dmon_log": str(dmon_path),
        }
        batch_rows.append(row)
        print(row)

    write_csv(out_dir / "prompt_length_sweep.csv", prompt_rows)
    write_csv(out_dir / "batch_size_sweep.csv", batch_rows)
    write_prompt_svg(out_dir / "prompt_length_sweep.svg", prompt_rows)
    write_batch_svg(out_dir / "batch_size_sweep.svg", batch_rows)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
