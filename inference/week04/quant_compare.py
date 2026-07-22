"""Lab 1 - Measure the same model across BF16 / INT8 / NF4 quantization.

The README sketches an AWQ INT4 path, but on Blackwell (RTX 5080, sm_120) the
prebuilt AWQ kernels are unreliable, so we use bitsandbytes for the low-bit
paths: 8-bit (LLM.int8()) and 4-bit NF4 (the QLoRA format). Both run natively
on Blackwell with a recent bitsandbytes + cu128 PyTorch.

Each variant reports generation latency for a fixed token budget and the peak
device memory, so we can confirm the Week 2 claim that decode is bandwidth-bound
and that fewer weight bytes -> faster decode.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PROMPT = "Explain memory bandwidth bottleneck in LLM decoding."


def benchmark(model, inputs, max_new_tokens: int, n_warmup: int, n_iter: int):
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / n_iter
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return elapsed, peak_gb


def load_model(model_id: str, variant: str):
    if variant == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
    if variant == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda"
        )
    if variant == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda"
        )
    raise ValueError(f"unknown variant: {variant}")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Compare quantization variants of one model.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--variants", default="bf16,int8,nf4")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", default="week04/results/quant_compare.csv")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    inputs = tok(args.prompt, return_tensors="pt").to("cuda")

    rows = []
    baseline_ms = None
    for variant in args.variants.split(","):
        variant = variant.strip()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = load_model(args.model, variant)
        elapsed, peak_gb = benchmark(
            model, inputs, args.max_new_tokens, args.warmup, args.iters
        )
        ms = elapsed * 1000
        if variant == "bf16":
            baseline_ms = ms
        speedup = baseline_ms / ms if baseline_ms else float("nan")

        row = {
            "variant": variant,
            "ms_per_gen": round(ms, 1),
            "peak_mem_gb": round(peak_gb, 2),
            "speedup_vs_bf16": round(speedup, 2),
            "max_new_tokens": args.max_new_tokens,
        }
        rows.append(row)
        print(
            f"{variant:<6} {ms:>7.1f} ms   {peak_gb:>5.2f} GB   "
            f"{speedup:>4.2f}x vs bf16"
        )

        del model
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
