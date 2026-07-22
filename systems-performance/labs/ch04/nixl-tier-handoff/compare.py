from __future__ import annotations

import baseline
import optimized
from common import make_workload, measure, print_report


def main() -> None:
    workload = make_workload()
    baseline_result = measure("baseline", baseline.run, workload)
    optimized_result = measure("optimized", optimized.run, workload)
    print_report(baseline_result, optimized_result)


if __name__ == "__main__":
    main()
