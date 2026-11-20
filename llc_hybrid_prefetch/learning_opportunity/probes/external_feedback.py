"""Transfer frozen BFS probes with and without resolved cache and Stride feedback."""

import argparse
import csv
import time

import numpy as np

from extract import HERE, RAW, STUDY
from feedback import prepare
from fit import counters, load, predict, splits, train, write

SCOPES = ("with_time", "with_feedback", "with_stride_feedback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    with (STUDY / "expansion/sources.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    d, n = load("bfs_discovery")
    data = d.loc[splits(d, n)["train"]]
    ref = np.unique(data.pc)
    models = {s: train(data, s, ref, 8, "delta") for s in SCOPES}
    rows = []
    while cases:
        pending = []
        for case in cases:
            verified = (
                STUDY
                / "raw/capture/runs"
                / f"{case}__stride"
                / "capture_validation.json"
            )
            if not verified.exists() or not (RAW / f"{case}__stride.npz").exists():
                pending.append(case)
                continue
            prepare(case)
            d, n = load(case)
            test = d.loc[splits(d, n)["test"]]
            for s, model in models.items():
                pred = predict(model, test, s, ref, 8, "delta")
                rows.append(
                    dict(
                        source="bfs_discovery",
                        destination=case,
                        horizon=8,
                        encoding="delta",
                        scope=s,
                        **counters(pred, np.ones(len(test), bool)),
                    )
                )
            write(HERE / "external_feedback.csv", rows)
            print(case, len(test), flush=True)
        cases = pending
        if not args.wait:
            break
        if cases:
            time.sleep(20)


if __name__ == "__main__":
    main()
