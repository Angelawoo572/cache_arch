"""Validate the Google and AI four-policy outcomes with the native dynamic operand join."""

import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "full_workload/native"))
from summarize import summarize

if __name__ == "__main__":
    with (HERE / "sources.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    summarize(
        HERE / "results",
        False,
        cases,
        STUDY / "raw/expansion",
        ("nopf", "stride", "multi", "spp"),
    )
