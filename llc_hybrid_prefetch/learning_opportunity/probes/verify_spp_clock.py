"""Check sampled SPP labels against native retirement masks and the original probe scorer."""

import numpy as np
import pandas as pd

from extract import HERE, RAW
from fit import HORIZONS, load, masks_for, predict, splits, train
from spp_clock import CASES, SCOPES


def main():
    d, n = load("bfs_discovery")
    donor = d.loc[splits(d, n)["train"]]
    ref = np.unique(donor.pc)
    models = {m: train(donor, m, ref, 8, "delta") for m in SCOPES}
    ledger = pd.read_csv(HERE / "spp_clock_claims.csv")
    for case in CASES:
        with np.load(RAW / f"{case}__spp.npz") as f:
            n = int(f["total_arrivals"][0])
            a = pd.DataFrame({k: f[k] for k in f.files if k != "total_arrivals"})
        test = a.loc[splits(a, n)["test"]]
        native = masks_for(case, test)
        raw = pd.read_csv(RAW / "spp_clock" / f"{case}.csv.gz")
        for scope, model in models.items():
            records = raw.loc[raw.scope == scope].reset_index(drop=True)
            assert np.array_equal(records.key, test.key)
            for arm in ("nopf", "spp"):
                assert np.array_equal(records[arm], native[arm][:, HORIZONS.index(8)])
            original = predict(model, test, scope, ref, 8, "delta")
            assert np.array_equal(records.legal & records.exact, original["forced_hit"])
            assert np.array_equal(~records.other_wins, original["scored"])
            assert np.array_equal(
                records.legal & records.exact & ~records.other_wins, original["hit"]
            )
            for kind, mask in (("late", 6), ("timely", 9)):
                selected = records[
                    records.legal
                    & records.exact
                    & (records.nopf == 2)
                    & (records.spp == mask)
                ]
                row = ledger[
                    (ledger.case == case)
                    & (ledger.scope == scope)
                    & (ledger.claim == kind)
                    & (ledger.policy == "one_proposal")
                ].iloc[0]
                assert len(selected) == row.correct_legal
                timed = selected.dropna(subset=["fill", "proposed"])
                assert not (timed.proposed > timed.claim_cycle).any()
                real_advance = timed.proposed - timed.cycle
                first_margin = (
                    timed.first_future_cycle
                    - timed.cycle
                    - (timed["fill"] - timed.proposed)
                )
                assert (
                    int(((real_advance > 0) & (first_margin > 0)).sum())
                    == row.before_enough_first_future
                )
    print(
        "Verified three SPP streams, six frozen probe fits, native NoPF/SPP nibbles, and service margins to first demand."
    )


if __name__ == "__main__":
    main()
