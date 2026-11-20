"""Validate all four aligned outcomes and write comparable metrics for each complete panel."""

import argparse
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
NATIVE = STUDY.parent / "full_workload/native"
sys.path.insert(0, str(NATIVE))
from summarize import summarize

POLICIES = ("nopf", "stride", "multi", "spp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    with (HERE / "sources.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    cases += ["cc5_slot1", "gcc734_slot1", "snap_stanford", "snap_google"]
    if args.cases and not set(args.cases).issubset(cases):
        parser.error("unknown recording or placement")
    summarize(args.out, False, args.cases or cases, STUDY / "raw/expansion", POLICIES)


if __name__ == "__main__":
    main()
