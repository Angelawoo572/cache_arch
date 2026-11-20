"""Bound the earliest legal trigger in the preceding 64 LLC arrivals."""

import argparse

import numpy as np

from measure import HERE, full_arrays, quantile, sample, save
from sequence import ROOT, Sequence


def measure_deadlines(seq):
    events, labels = full_arrays(seq)
    targets = sample(seq)
    physical = events[:, 2]
    cycles = events[:, 3]
    page = physical[targets] >> 6
    # A trigger can retain 64 prior/current page records. Its earliest page witness
    # can be another 63 events before the first of the 64 putative trigger points.
    lookback = targets[:, None] - np.arange(127, 0, -1)
    witness = ((physical[np.maximum(0, lookback)] >> 6) == page[:, None]) & (
        lookback >= 0
    )
    start = np.maximum(0, targets - 64)
    loc = np.arange(len(targets))
    latest_window = witness[:, -64:]
    earliest = {
        "same_page": np.where(
            latest_window.any(axis=1),
            lookback[loc, latest_window.argmax(axis=1) + 63],
            -1,
        ),
        "page_in_last64": np.where(
            witness.any(axis=1),
            np.maximum(start, lookback[loc, witness.argmax(axis=1)]),
            -1,
        ),
    }
    first_seen = seq.first["page"][seq.keys["page"].searchsorted(page)].astype(np.int64)
    earliest["any_prior_observed_page"] = np.where(
        first_seen < targets, np.maximum(start, first_seen), -1
    )
    masks = (labels[targets, None] >> np.array([0, 4, 8, 12], dtype="<u2")) & 15
    original = masks[:, 0] == 2
    cohorts = {
        "all_admissions": np.ones(len(targets), dtype=bool),
        "original_miss": original,
        "original_all_three_miss_or_late": original
        & np.isin(masks[:, 1:], [2, 6]).all(axis=1),
        "original_spp_late": original & (masks[:, 3] == 6),
        "original_spp_timely": original & (masks[:, 3] == 9),
    }
    results = []
    for vocabulary, triggers in earliest.items():
        if np.any(triggers[triggers >= 0] >= targets[triggers >= 0]):
            raise ValueError("Future information entered the legal trigger")
        for cohort, include in cohorts.items():
            good = include & (triggers >= 0)
            leads = cycles[targets[good]] - cycles[triggers[good]]
            ranks = targets[good] - triggers[good]
            results.append(
                dict(
                    case=seq.meta["case"],
                    policy=seq.meta["policy"],
                    cohort=cohort,
                    vocabulary=vocabulary,
                    sampled_targets=int(include.sum()),
                    incomplete_history=int((include & (targets < 127)).sum()),
                    has_legal_anchor=int(good.sum()),
                    rank_gap_p50=quantile(ranks, 0.5),
                    lead_p10_cycles=quantile(leads, 0.1),
                    lead_p50_cycles=quantile(leads, 0.5),
                    lead_p90_cycles=quantile(leads, 0.9),
                    lead_ge64=int((leads >= 64).sum()),
                    lead_ge130=int((leads >= 130).sum()),
                    lead_ge134=int((leads >= 134).sum()),
                    lead_ge146=int((leads >= 146).sum()),
                    lead_ge194=int((leads >= 194).sum()),
                    lead_ge256=int((leads >= 256).sum()),
                )
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--policy", default="stride")
    args = parser.parse_args()
    for case in args.cases:
        name = f"{case}__{args.policy}"
        save(
            HERE / "results" / f"{name}_deadlines.csv",
            measure_deadlines(Sequence(ROOT / name)),
        )
