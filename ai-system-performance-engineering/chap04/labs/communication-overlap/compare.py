from __future__ import annotations

import baseline
import optimized
from common import make_buckets, measure, print_report


def main() -> None:
    buckets = make_buckets()
    baseline_result = measure("baseline", baseline.run, buckets)
    optimized_result = measure("optimized", optimized.run, buckets)
    print_report(baseline_result, optimized_result)


if __name__ == "__main__":
    main()
