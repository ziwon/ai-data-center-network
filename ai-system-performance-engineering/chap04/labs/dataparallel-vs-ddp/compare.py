from __future__ import annotations

import baseline
import optimized
from common import TrainingShape, measure, print_report


def main() -> None:
    shape = TrainingShape()
    print_report(
        measure("baseline", baseline.run, shape),
        measure("optimized", optimized.run, shape),
    )


if __name__ == "__main__":
    main()
