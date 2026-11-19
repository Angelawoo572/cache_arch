"""Recreate six frozen tables and corresponding Stride-shadow choices from the archived sources."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = [
    ("gap_bfs3_late", "gap_bfs3_late", "bfs3-late", "page"),
    ("row_period2_paced", "row_period2_paced", "row2", "page"),
]
TABLES.extend(
    ("lbm-r1", "lbm-r2", "lbm-" + rep, rep)
    for rep in ("page", "exact", "unrelated_older", "last_global_pc")
)


def prepare(job):
    output, (src, test, name, rep) = job
    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "predictability/export.py"),
            src,
            test,
            str(output / (name + ".txt")),
            "--mode",
            rep,
            "--expected",
            str(output / (name + "-expected.csv")),
        ],
        check=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE.parent / "raw/predictability")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(prepare, [(args.output, t) for t in TABLES]))
