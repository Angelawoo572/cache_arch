"""Run exactly the registered independent continuous Stride replays on BFS-3 and Soplex."""

from importlib import import_module
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
sys.path.insert(0, str(FULL / "native"))
native = import_module("run")

RAW = FULL / "raw/continuous"
NATIVE = FULL / "raw/native/runs"
CASES = ("bfs_discovery", "soplex06_heldout")


def run(case):
    return native.run(
        case,
        "stride",
        [FULL / "raw/traces" / (case + ".champsimtrace")],
        [0],
        50_000_000,
        200_000_000,
    )


if __name__ == "__main__":
    native.BIN = RAW / "build/pythia/bin/champsim"
    native.RAW = RAW
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, CASES))
