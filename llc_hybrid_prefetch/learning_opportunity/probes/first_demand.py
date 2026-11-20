"""Measure the earliest actual future demand for each frozen probe address."""

import argparse
import sys

import numpy as np

from extract import HERE, STUDY
from fit import load, predict, splits, train, write

sys.path.insert(0, str(STUDY / "labels"))
from sequence import ROOT, Sequence

WINDOWS = (512, 2048, 8192)
SCOPES = ("same_pc", "local_region", "global32")


def run(source, destinations):
    frame, count = load(source)
    training = frame.loc[splits(frame, count)["train"]]
    reference = np.unique(training.pc)
    models = {s: train(training, s, reference, 8, "delta") for s in SCOPES}
    rows = []
    for dest in destinations:
        frame, count = load(dest)
        test = frame.loc[splits(frame, count)["test"]]
        seq = Sequence(ROOT / f"{dest}__stride")
        start = 4 * count // 5
        ranks = np.arange(start, count)
        observations = seq.observations(ranks)
        line, cycles = observations[:, 2], observations[:, 3]
        positions = test["index"].to_numpy() - start
        if not np.array_equal(test.key.to_numpy(), observations[positions, 0]):
            raise ValueError("Cue order mismatch")
        for scope in SCOPES:
            scored = predict(models[scope], test, scope, reference, 8, "delta")
            # Compute first future use only after generating all concrete predictions.
            issued = scored["legal"]
            first = np.full(len(test), -1, np.int64)
            for i in np.flatnonzero(issued):
                a = positions[i]
                end = cycles.searchsorted(cycles[a] + WINDOWS[-1], side="right")
                match = np.flatnonzero(line[a + 1 : end] == scored["pred"][i])
                if len(match):
                    first[i] = a + 1 + int(match[0])
            observed_lead = np.full(len(test), -1, np.int64)
            used = first >= 0
            observed_lead[used] = (
                cycles[first[used]] - cycles[positions[used]]
            ).astype(np.int64)
            statuses = np.full((len(test), 4), -1, np.int64)
            statuses[used] = seq.outcomes(first[used] + start)
            for window in WINDOWS:
                eligible = cycles[positions] + window <= cycles[-1]
                occurs = eligible & used & (observed_lead <= window)
                qualified = occurs & (observed_lead >= 130)
                missed = qualified & (statuses[:, 0] == 2)
                rows.append(
                    dict(
                        source=source,
                        destination=dest,
                        horizon=8,
                        scope=scope,
                        window_cycles=window,
                        eligible_cues=int(eligible.sum()),
                        legal_offers=int((eligible & issued).sum()),
                        any_first_demand=int(occurs.sum()),
                        first_demand_before_130=int((occurs & ~qualified).sum()),
                        qualified_first_demand=int(qualified.sum()),
                        qualified_outside_h8=int(
                            (qualified & (first != positions + 8)).sum()
                        ),
                        h8_qualified_but_first_too_early=int(
                            (
                                eligible
                                & scored["deadline"]
                                & (observed_lead >= 0)
                                & (observed_lead < 130)
                            ).sum()
                        ),
                        first_nopf_miss=int(missed.sum()),
                        first_original_spp_late=int(
                            (missed & (statuses[:, 3] == 6)).sum()
                        ),
                        first_original_spp_persistent=int(
                            (missed & (statuses[:, 3] == 2)).sum()
                        ),
                        first_original_spp_timely=int(
                            (missed & (statuses[:, 3] == 9)).sum()
                        ),
                    )
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", nargs="+", required=True)
    args = parser.parse_args()
    write(HERE / f"first_demand_{args.source}.csv", run(args.source, args.dest))
