"""Continue GAP PageRank from first instruction through EOF without resetting state."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from run import FULL, run

CASE = "pr_heldout_entire"
POLICIES = ("nopf", "stride", "multi", "spp", "temporal")
WARM = 50_000_000


def execute(policy):
    with (FULL / "intake" / (CASE + ".validation.csv")).open() as f:
        row = next(csv.DictReader(f))
    count = int(row["source_records"])
    if int(row["materialized_start"]) or int(row["materialized_records"]) != count:
        raise ValueError("Whole-recording validation and EOF required")
    p = FULL / "raw/traces" / (CASE + ".champsimtrace")
    run(CASE, policy, [p], [0], WARM, count - WARM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(execute, POLICIES))
