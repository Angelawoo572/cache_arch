"""Update optional analyses after the registered native and indexed policy runs finish."""

import argparse
import csv
import subprocess
import sys
import time

from build import ARMS
from measure import HERE, save
from sequence import ROOT, Sequence
from visibility import measure as visibility
from vocabulary import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    pending = list(dict.fromkeys(args.cases))
    while pending:
        for case in pending[:]:
            name = case + "__stride"
            if not (ROOT / name / "index.json").is_file():
                continue
            path = HERE / "results" / (name + "_novelty.csv")
            current_novelty = False
            if path.is_file():
                with path.open() as f:
                    current_novelty = (
                        "eligible_distinct_line_first_future_demand"
                        in csv.DictReader(f).fieldnames
                    )
            if not current_novelty:
                save(path, summarize(Sequence(ROOT / name)))
            if not all((ROOT / f"{case}__{arm}/index.json").is_file() for arm in ARMS):
                continue
            path = HERE / "results" / (case + "_visibility.csv")
            if not path.exists():
                save(path, visibility(case))
            expected = ("horizons", "windows", "deadlines", "novelty")
            if not all(
                (HERE / "results" / f"{name}_{kind}.csv").is_file() for kind in expected
            ):
                continue
            print("Four-policy labels ready", case, flush=True)
            pending.remove(case)
        if pending:
            if not args.wait:
                raise FileNotFoundError(
                    "Missing indexed policy or Stride summaries for "
                    + ", ".join(pending)
                )
            time.sleep(15)
    for script in ("audit.py", "examples.py"):
        subprocess.run([sys.executable, "-B", str(HERE / script)], check=True)
    cases = [c for c in args.cases if c + "_slot1" in args.cases]
    if cases:
        subprocess.run(
            [sys.executable, "-B", str(HERE / "placement.py"), *cases], check=True
        )
    subprocess.run([sys.executable, "-B", str(HERE / "summarize.py")], check=True)


if __name__ == "__main__":
    main()
