import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute KV cache bytes per token from model configs.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated Hugging Face model IDs.")
    parser.add_argument("--out", default="week03/results/kv_cache_sizes.csv", help="CSV output path.")
    parser.add_argument(
        "--measure-model",
        default="",
        help="Optional model ID to load and measure actual past_key_values size. Leave empty to skip.",
    )
    parser.add_argument("--prompt-repeat", type=int, default=50, help="Prompt repeat count for measurement.")
    return parser.parse_args()


def config_row(model_id: str) -> dict[str, str | int | float]:
    config = AutoConfig.from_pretrained(model_id)
    n_layer = config.num_hidden_layers
    n_q_head = config.num_attention_heads
    n_kv_head = getattr(config, "num_key_value_heads", n_q_head)
    d_head = getattr(config, "head_dim", None) or config.hidden_size // n_q_head
    bytes_per_token_bf16 = 2 * n_layer * n_kv_head * d_head * 2

    return {
        "model": model_id,
        "layers": n_layer,
        "query_heads": n_q_head,
        "kv_heads": n_kv_head,
        "gqa_ratio": n_q_head / n_kv_head,
        "head_dim": d_head,
        "kv_per_token_bytes_bf16": bytes_per_token_bf16,
        "kv_per_token_kib_bf16": bytes_per_token_bf16 / 1024,
        "kv_per_token_kib_int8": bytes_per_token_bf16 / 1024 / 2,
        "kv_per_token_kib_int4": bytes_per_token_bf16 / 1024 / 4,
    }


def measure_actual_kv(model_id: str, prompt_repeat: int) -> dict[str, float | int | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual KV measurement.")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    prompt = "Hello world " * prompt_repeat
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model(**inputs, use_cache=True)

    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in iter_cache_tensors(out.past_key_values))
    seq_len = inputs.input_ids.shape[1]

    return {
        "model": model_id,
        "measured_seq_len": seq_len,
        "measured_total_bytes": total_bytes,
        "measured_kv_per_token_kib": total_bytes / seq_len / 1024,
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


def write_csv(path: Path, rows: list[dict[str, str | int | float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    model_ids = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = []
    failures = []
    for model_id in model_ids:
        try:
            rows.append(config_row(model_id))
        except Exception as exc:  # noqa: BLE001 - keep the lab usable with partial local HF caches.
            failures.append((model_id, str(exc)))

    if not rows:
        raise RuntimeError("No model configs could be loaded. Check HF_HOME, HF_HUB_OFFLINE, or --models.")

    print(f"{'Model':<42} {'L':>3} {'Q':>4} {'KV':>4} {'GQA':>5} {'d':>4} {'KV/token':>10}")
    print("-" * 82)
    for row in rows:
        print(
            f"{row['model']:<42} {row['layers']:>3} {row['query_heads']:>4} "
            f"{row['kv_heads']:>4} {row['gqa_ratio']:>5.1f} {row['head_dim']:>4} "
            f"{row['kv_per_token_kib_bf16']:>8.1f} KiB"
        )

    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"\nWrote {out_path}")

    if failures:
        print("\nSkipped models:")
        for model_id, error in failures:
            print(f"  {model_id}: {error.splitlines()[0]}")

    if args.measure_model:
        measured = measure_actual_kv(args.measure_model, args.prompt_repeat)
        print(
            "\nMeasured actual KV: "
            f"{measured['model']}, seq_len={measured['measured_seq_len']}, "
            f"KV/token={measured['measured_kv_per_token_kib']:.1f} KiB"
        )


if __name__ == "__main__":
    main()
