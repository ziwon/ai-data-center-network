from __future__ import annotations

import baseline
import optimized
from common import make_gradients, measure, print_report


def main() -> None:
    gradients = make_gradients()
    baseline_result = measure("baseline", baseline.run, gradients)
    optimized_result = measure("optimized", optimized.run, gradients)
    print_report(baseline_result, optimized_result)


if __name__ == "__main__":
    main()
