"""Apply the existing BFS fit to the separately registered service and AI traces."""

import argparse
import csv
import time

import numpy as np

from extract import HERE, RAW, STUDY, extract, locate
from fit import HORIZONS, counters, load, masks_for, predict, splits, train, write

SCOPES = ("same_pc", "local_region", "global32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    wait = parser.parse_args().wait
    with (STUDY / "domains/sources.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    d, n = load("bfs_discovery")
    training = d.loc[splits(d, n)["train"]]
    ref = np.unique(training.pc)
    models = {s: train(training, s, ref, 8, "delta") for s in SCOPES}
    rows = []
    outcomes = []
    last_ready = {}
    while cases:
        pending = []
        for case in cases:
            ready = set()
            for arm in ("nopf", "stride", "multi", "spp"):
                try:
                    locate(case, arm)
                    ready.add(arm)
                except FileNotFoundError:
                    pass
            if len(ready) < 4:
                pending.append(case)
            if "stride" not in ready or last_ready.get(case) == ready:
                continue
            last_ready[case] = ready
            rows = [r for r in rows if r["destination"] != case]
            outcomes = [r for r in outcomes if r["destination"] != case]
            if not (RAW / f"{case}__stride.npz").is_file():
                extract(case, "stride")
            frame, count = load(case)
            test = frame.loc[splits(frame, count)["test"]]
            masks = masks_for(case, test)
            k = HORIZONS.index(8)
            for scope, model in models.items():
                pred = predict(model, test, scope, ref, 8, "delta")
                info = dict(
                    source="bfs_discovery",
                    destination=case,
                    scope=scope,
                    horizon=8,
                    encoding="delta",
                )
                for cohort, mask in (
                    ("all", np.ones(len(test), bool)),
                    ("unseen_donor_pc", ~np.isin(test.pc, ref)),
                    (
                        "new_line_known_page",
                        (test.y8_seen == 0) & (test.y8_seenpage > 0),
                    ),
                ):
                    rows.append({**info, "cohort": cohort, **counters(pred, mask)})
                for arm in ("stride", "multi", "spp"):
                    if "nopf" not in masks or arm not in masks:
                        continue
                    for status, code in (
                        ("no_tag", 0),
                        ("ordinary", 1),
                        ("persistent", 2),
                        ("late", 6),
                        ("timely", 9),
                    ):
                        selected = (masks["nopf"][:, k] == 2) & (
                            masks[arm][:, k] == code
                        )
                        outcomes.append(
                            {
                                **info,
                                "arm": arm,
                                "original_miss_outcome": status,
                                **counters(pred, selected),
                            }
                        )
            write(HERE / "domain_transfer.csv", rows)
            write(HERE / "domain_outcomes.csv", outcomes)
            print(case, len(test), flush=True)
        cases = pending
        if not wait:
            break
        if cases:
            time.sleep(23)


if __name__ == "__main__":
    main()
