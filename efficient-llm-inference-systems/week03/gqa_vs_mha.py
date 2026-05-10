import csv
from pathlib import Path


CONFIGS = {
    "Llama-2-7B (MHA)": {"L": 32, "n_q": 32, "n_kv": 32, "d_head": 128},
    "Llama-3-8B (GQA-4)": {"L": 32, "n_q": 32, "n_kv": 8, "d_head": 128},
    "Llama-2-70B (MHA)": {"L": 80, "n_q": 64, "n_kv": 64, "d_head": 128},
    "Llama-3-70B (GQA-8)": {"L": 80, "n_q": 64, "n_kv": 8, "d_head": 128},
}


def row_for(name: str, config: dict[str, int]) -> dict[str, str | int | float]:
    kv_per_token = 2 * config["L"] * config["n_kv"] * config["d_head"] * 2
    return {
        "model": name,
        "layers": config["L"],
        "query_heads": config["n_q"],
        "kv_heads": config["n_kv"],
        "head_dim": config["d_head"],
        "gqa_ratio": config["n_q"] / config["n_kv"],
        "kv_per_token_kib": kv_per_token / 1024,
        "kv_8k_mib": kv_per_token * 8192 / 1024**2,
        "kv_128k_mib": kv_per_token * 131072 / 1024**2,
    }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [row_for(name, config) for name, config in CONFIGS.items()]
    print(f"{'Model':<25} {'KV/token':<12} {'8K context':<15} {'128K context':<15}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['model']:<25} {row['kv_per_token_kib']:>5.0f} KiB   "
            f"{row['kv_8k_mib']:>6.1f} MiB      {row['kv_128k_mib']:>7.1f} MiB"
        )

    out_path = Path("week03/results/gqa_vs_mha.csv")
    write_csv(out_path, rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
