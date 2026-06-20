from __future__ import annotations

import baseline
import optimized
from common import make_topology, measure, print_report


def main() -> None:
    topology = make_topology()
    print_report(
        measure("baseline", baseline.run, topology),
        measure("optimized", optimized.run, topology),
    )


if __name__ == "__main__":
    main()
