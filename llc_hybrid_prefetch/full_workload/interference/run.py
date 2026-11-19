"""Bounded matching solo and active-core runs. Selection is recorded in README.md."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import subprocess
import sys

FULL = Path(__file__).resolve().parents[1]
MCF, BFS = "mcf17_discovery", "bfs_discovery"
PLACEMENTS = {
    "pair": ((MCF, BFS), (0, 1)),
    "mcf": ((MCF,), (0,)),
    "bfs": ((BFS,), (1,)),
}
ARMS = ("nopf", "stride", "multi", "spp")


def run(job):
    policy, name = job
    cases, slots = PLACEMENTS[name]
    traces = [str(FULL / "raw/traces" / (case + ".champsimtrace")) for case in cases]
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(FULL / "native/run.py"),
            "--case",
            "interference_" + name,
            "--policy",
            policy,
            "--slots",
            *map(str, slots),
            "--traces",
            *traces,
            "--warmup",
            "5000000",
            "--roi",
            "20000000",
            "--sample",
            "10000",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, product(ARMS, PLACEMENTS)))
