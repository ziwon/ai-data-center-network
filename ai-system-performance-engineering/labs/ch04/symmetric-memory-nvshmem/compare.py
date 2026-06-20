from __future__ import annotations

import baseline
import optimized
from common import ExchangeWorkload, measure, print_report


def main() -> None:
    workload = ExchangeWorkload()
    print_report(
        measure("baseline", baseline.run, workload),
        measure("optimized", optimized.run, workload),
    )


if __name__ == "__main__":
    main()
