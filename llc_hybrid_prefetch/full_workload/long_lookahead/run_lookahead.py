"""Evaluate the existing look2 mechanism on the two frozen program holdouts."""

import csv
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
sys.path.insert(0, str(FULL / "native"))
run_native = importlib.import_module("run").run

CASES = ("pr_heldout", "soplex06_heldout")
WARMUP = 50_000_000
MEASURED = 200_000_000


def run_case(case):
    view = FULL / "intake" / (case + ".validation.csv")
    with view.open() as f:
        (record,) = csv.DictReader(f)
    if (
        int(record["materialized_start"]) != 0
        or int(record["materialized_records"]) < WARMUP + MEASURED
    ):
        raise ValueError(f"Invalid fixed interval for {case}")
    trace = FULL / "raw/traces" / (case + ".champsimtrace")
    return run_native(case, "look2", [trace], [0], WARMUP, MEASURED)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_case, CASES))
