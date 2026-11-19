"""Project complete LLC demand ingress and offline paired labels from native logs."""

from concurrent.futures import ProcessPoolExecutor
import argparse
from pathlib import Path
import sys

import numpy as np

from run import DEFAULT_OUTPUT, POLICIES, PROJECT, select

sys.path.append(str(PROJECT / "opportunity_study/history"))
from extract import parse_arm, FIELDS


def make(job):
    root, case = job
    folders = {p: root / "runs" / case / p for p in POLICIES}
    assert all((f / "complete.json").is_file() for f in folders.values())
    baseline = parse_arm(folders["nopf"])
    dest = root / "history" / case
    dest.mkdir(parents=True, exist_ok=True)
    for policy, folder in folders.items():
        events, _ = parse_arm(folder, baseline)
        events = events[0]
        fields = {
            k: np.fromiter(
                (r[i] for r in events),
                dtype=("uint64" if k in ("pc", "va") else "int64"),
                count=len(events),
            )
            for i, k in enumerate(FIELDS)
        }
        assert np.all(np.diff(fields["t"]) >= 0)
        tmp = dest / (policy + ".tmp.npz")
        np.savez_compressed(tmp, trace=case, **fields)
        tmp.replace(dest / (policy + ".npz"))
    return case


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    a = parser.parse_args()
    with ProcessPoolExecutor(a.jobs) as pool:
        for case in pool.map(
            make, [(a.output.resolve(), r["case"]) for r in select(a.case)]
        ):
            print(case, flush=True)
