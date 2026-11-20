"""Replay two named graph inputs with identical instruction boundaries."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "expansion"))
from run import native_module  # expansion entry point uses its own study paths


def main():
    native = native_module()
    cases = ("snap_stanford", "snap_google")
    jobs = []
    with ThreadPoolExecutor(4) as pool:
        for case in cases:
            trace = HERE.parent / "raw/graph_inputs" / (case + ".champsimtrace")
            for policy in ("stride", "nopf", "multi", "spp"):
                jobs.append(
                    pool.submit(
                        native.run,
                        case,
                        policy,
                        [trace],
                        [0],
                        250_000,
                        2_000_000,
                        sample=10_000,
                    )
                )
        for job in jobs:
            print(job.result().name, flush=True)


if __name__ == "__main__":
    main()
