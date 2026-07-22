"""Lab 3 - Edge simulation: project AGX Orin decode latency under quantization.

Pure arithmetic, no GPU required. We take the Week 3 Orin measurement
(BF16 7B: ~94 ms/token decode at batch 1) and project INT8 / INT4 decode under a
bandwidth-bound model: decode_step_time ~= weight_bytes / memory_bandwidth.

The projection is calibrated so the BF16 row reproduces the measured 94 ms/token,
then INT8 / INT4 follow from their reduced weight footprint.
"""

import argparse
import csv
from pathlib import Path


ORIN_BANDWIDTH_GBS = 200  # AGX Orin LPDDR5
WEIGHT_GB = {"BF16": 14.0, "INT8": 7.0, "INT4": 3.5}
BF16_MEASURED_DECODE_MS = 94  # Week 3: batch=1, ~94 ms/token


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Project Orin decode latency by precision.")
    p.add_argument("--bandwidth", type=float, default=ORIN_BANDWIDTH_GBS)
    p.add_argument("--decode-tokens", type=int, default=16)
    p.add_argument("--out", default="week04/results/orin_projection.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # Calibration: theoretical BF16 step vs measured, to absorb real-world
    # overhead (kernel launch, non-weight traffic) into a single scale factor.
    bf16_theoretical_ms = WEIGHT_GB["BF16"] / args.bandwidth * 1000
    scale = BF16_MEASURED_DECODE_MS / bf16_theoretical_ms
    print(f"Calibration scale (measured / theoretical): {scale:.2f}\n")

    rows = []
    for precision, weight_gb in WEIGHT_GB.items():
        theoretical_ms = weight_gb / args.bandwidth * 1000
        decode_ms = theoretical_ms * scale
        total_s = decode_ms * args.decode_tokens / 1000
        rows.append(
            {
                "precision": precision,
                "weight_gb": weight_gb,
                "ms_per_token": round(decode_ms, 1),
                f"decode_{args.decode_tokens}tok_s": round(total_s, 2),
                "speedup_vs_bf16": round(WEIGHT_GB["BF16"] / weight_gb, 2),
            }
        )
        tag = "(matches measurement)" if precision == "BF16" else "(projection)"
        print(
            f"{precision}: weight {weight_gb}GB -> decode {decode_ms:.0f} ms/token, "
            f"{args.decode_tokens} token decode {total_s:.1f}s {tag}"
        )

    out_path = Path(args.out)
    write_csv(out_path, rows)
    print(f"\nWrote {out_path}")
    print(
        "\nNote: W4A16 reduces weight-read bytes (decode/bandwidth win) but "
        "dequantizes to FP16 for compute, so prefill compute is unchanged."
    )


if __name__ == "__main__":
    main()
