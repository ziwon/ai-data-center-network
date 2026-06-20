"""Lab 2 - Quality measurement via WikiText-2 perplexity.

Runs the same BF16 / INT8 / NF4 variants from Lab 1 and reports perplexity on a
WikiText-2 sample, so we can quantify the quality cost of each quantization
path. Rule of thumb from the README: if the low-bit perplexity stays within ~5%
of BF16, it is generally production-viable.
"""

import argparse
import csv
from pathlib import Path

import torch
from datasets import load_dataset

from quant_compare import load_model
from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def compute_perplexity(model, tok, texts, max_length: int) -> float:
    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to("cuda")
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        nll_sum += out.loss.item() * input_ids.shape[1]
        n_tokens += input_ids.shape[1]
    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Perplexity per quantization variant.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--variants", default="bf16,int8,nf4")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--out", default="week04/results/perplexity.csv")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)

    # datasets>=5 requires a namespaced repo id; the bare "wikitext" alias
    # fails to resolve under the current huggingface_hub.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][: args.n_samples]
    print(f"Evaluating on {len(texts)} WikiText-2 samples\n")

    rows = []
    baseline_ppl = None
    for variant in args.variants.split(","):
        variant = variant.strip()
        torch.cuda.empty_cache()
        model = load_model(args.model, variant)
        ppl = compute_perplexity(model, tok, texts, args.max_length)
        if variant == "bf16":
            baseline_ppl = ppl
        delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100 if baseline_ppl else 0.0

        rows.append(
            {
                "variant": variant,
                "perplexity": round(ppl, 3),
                "delta_pct_vs_bf16": round(delta_pct, 2),
            }
        )
        print(f"{variant:<6} ppl={ppl:>7.3f}   {delta_pct:+5.2f}% vs bf16")

        del model
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
