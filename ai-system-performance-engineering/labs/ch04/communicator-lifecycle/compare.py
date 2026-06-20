from __future__ import annotations

import baseline
import optimized
from common import measure, print_report


def main() -> None:
    print_report(
        measure("baseline", baseline.run),
        measure("optimized", optimized.run),
    )


if __name__ == "__main__":
    main()
