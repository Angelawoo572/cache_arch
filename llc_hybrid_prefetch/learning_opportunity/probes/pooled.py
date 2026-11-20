"""Test frozen, balanced multi-program training on the registered independent recordings."""

import argparse
import csv
import time

import numpy as np
import pandas as pd

from extract import HERE, RAW, STUDY, locate
from fit import counters, load, predict, splits, train, write

DONORS = ("bfs_discovery", "mcf17_discovery", "soplex06_heldout")
SCOPES = ("same_pc", "local_region", "global32", "no_pc_ids")
ROWS_PER_DONOR = 12000


def training_data():
    parts = []
    for name in DONORS:
        data, n = load(name)
        train_slice = data.loc[splits(data, n)["train"]]
        if len(train_slice) < ROWS_PER_DONOR:
            raise ValueError("Insufficient training samples in donor")
        # Equal-frequency deterministic spread over the donor training segment.
        indices = (
            np.arange(ROWS_PER_DONOR) * len(train_slice) + len(train_slice) // 2
        ) // ROWS_PER_DONOR
        parts.append(train_slice.iloc[indices])
    return pd.concat(parts, axis=0, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    with (STUDY / "expansion/sources.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    data = training_data()
    ref = np.unique(data.pc)
    models = {scope: train(data, scope, ref, 8, "delta") for scope in SCOPES}
    path = HERE / "pooled_transfer.csv"
    rows = []
    while cases:
        pending = []
        for case in cases:
            try:
                locate(case, "stride")
                if not (RAW / f"{case}__stride.npz").exists():
                    raise FileNotFoundError
            except FileNotFoundError:
                pending.append(case)
                continue
            data, n = load(case)
            test = data.loc[splits(data, n)["test"]]
            novel = ~np.isin(test.pc, ref)
            for scope, model in models.items():
                pr = predict(model, test, scope, ref, 8, "delta")
                for cohort, mask in (
                    ("all", np.ones(len(test), bool)),
                    ("unseen_donor_pc", novel),
                    (
                        "new_line_known_page",
                        (test.y8_seen == 0) & (test.y8_seenpage > 0),
                    ),
                ):
                    rows.append(
                        dict(
                            destination=case,
                            scope=scope,
                            cohort=cohort,
                            **counters(pr, mask),
                        )
                    )
            print(case, len(test), flush=True)
            write(path, rows)
        cases = pending
        if not args.wait:
            break
        if cases:
            time.sleep(17)


if __name__ == "__main__":
    main()
