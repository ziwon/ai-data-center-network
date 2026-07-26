"""Lab 1b - Fused-kernel quantization on Blackwell via vLLM.

Lab 1 used bitsandbytes and found INT8/NF4 *slower* than BF16, because bnb is a
memory-optimization path, not a fused decode kernel. This script tests whether a
fused INT4 path (AWQ via the Marlin kernel) reproduces the README's decode
speedup on RTX 5080 (sm_120).

Run one variant per process so vLLM fully releases GPU memory between configs:

    python vllm_quant_bench.py --model Qwen/Qwen2.5-3B-Instruct --label bf16
    python vllm_quant_bench.py --model Qwen/Qwen2.5-3B-Instruct-AWQ \
        --quantization awq_marlin --label awq_int4

Each call appends a row to results/vllm_quant_bench.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


DEFAULT_PROMPT = "Explain memory bandwidth bottleneck in LLM decoding."


def parse_args():
    p = argparse.ArgumentParser(description="vLLM decode-latency benchmark per quant variant.")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="row label, e.g. bf16 / awq_int4")
    p.add_argument("--quantization", default=None, help="e.g. awq_marlin, gptq_marlin, fp8")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.55)
    p.add_argument("--out", default="week04/results/vllm_quant_bench.csv")
    return p.parse_args()


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # warmup (also triggers CUDA graph capture on first real runs)
    for _ in range(args.warmup):
        llm.generate([args.prompt], sp, use_tqdm=False)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        out = llm.generate([args.prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) / args.iters * 1000
    gen_tokens = len(out[0].outputs[0].token_ids)
    tok_per_s = gen_tokens / (elapsed_ms / 1000)

    row = {
        "label": args.label,
        "model": args.model,
        "quantization": args.quantization or "none",
        "ms_per_gen": round(elapsed_ms, 1),
        "gen_tokens": gen_tokens,
        "tok_per_s": round(tok_per_s, 1),
    }
    append_csv(Path(args.out), row)
    print(
        f"\n{args.label:<10} {elapsed_ms:>7.1f} ms/gen   "
        f"{gen_tokens} tok   {tok_per_s:>6.1f} tok/s"
    )


if __name__ == "__main__":
    main()
