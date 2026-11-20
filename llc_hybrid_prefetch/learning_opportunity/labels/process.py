"""Build and measure the designated policy for each case with a complete alignment."""

import argparse
import time

from build import build, sources
from deadlines import measure_deadlines
from measure import HERE, measure, save
from sequence import Sequence
from vocabulary import summarize


def process(case, policy):
    seq = Sequence(build(case, policy))
    folder = HERE / "results"
    folder.mkdir(exist_ok=True)
    stem = f"{case}__{policy}_"
    deadline = folder / (stem + "deadlines.csv")
    if not deadline.is_file():
        save(deadline, measure_deadlines(seq))
    novelty = folder / (stem + "novelty.csv")
    if not novelty.is_file():
        save(novelty, summarize(seq))
    horizons, windows = [folder / (stem + s + ".csv") for s in ("horizons", "windows")]
    if not (horizons.is_file() and windows.is_file()):
        first, second = measure(seq)
        save(horizons, first)
        save(windows, second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--policy", default="stride")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for in-progress native captures"
    )
    args = parser.parse_args()
    pending = list(dict.fromkeys(args.cases))
    while pending:
        for case in pending[:]:
            try:
                sources(case, args.policy)
            except FileNotFoundError:
                if not args.wait:
                    raise
                continue
            print("Processing", case, flush=True)
            process(case, args.policy)
            pending.remove(case)
        if pending:
            time.sleep(10)
