"""Run two unchanged arithmetic controls across the three fixed MCF17 views."""

import csv
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from itertools import product
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FULL = HERE.parents[1]
sys.path.insert(0, str(FULL / "native"))
native = import_module("run")

CASES = ("mcf17_discovery_far", "mcf17_discovery", "mcf17_transfer")
POLICIES = ("shared_increment", "local")
RAW = FULL / "raw/mcf_arithmetic"
WARMUP = 50_000_000
ROI = 200_000_000


def execute(job):
    case, policy = job
    with (FULL / "intake" / (case + ".validation.csv")).open() as f:
        (source,) = csv.DictReader(f)
    if int(source["materialized_records"]) < WARMUP + ROI:
        raise ValueError(f"Short view {case}")
    origin = int(source["materialized_start"])
    if origin != (1_750_000_000 if case == "mcf17_discovery_far" else 0):
        raise ValueError(f"Unexpected source origin for {case}")
    inputs = FULL / "raw/traces"
    trace = inputs / (case + ".champsimtrace")
    seed = inputs / (case + ".prefix_vpns.u64le") if origin else None
    return native.run(
        case, policy, [trace], [0], WARMUP, ROI, origin=origin, prefix_pages=seed
    )


if __name__ == "__main__":
    native.RAW = RAW
    native.BIN = FULL / "raw/arithmetic/build/pythia/bin/champsim"
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(execute, product(CASES, POLICIES)))
