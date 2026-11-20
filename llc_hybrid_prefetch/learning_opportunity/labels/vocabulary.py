"""Count retrospective novel and familiar targets and the alternatives at each anchor."""

import argparse
import numpy as np

from measure import HERE, full_arrays, sample, save, quantile
from sequence import HORIZONS, ROOT, Sequence


def summarize(seq):
    events, masks = full_arrays(seq)
    a = sample(seq)
    history = events[np.maximum(0, a[:, None] - np.arange(64)), 2] >> 6
    distinct_pages = 1 + np.count_nonzero(
        np.diff(np.sort(history, axis=1), axis=1), axis=1
    )
    rows = []
    for k in HORIZONS:
        observed = a + k < seq.size
        anchor = a[observed]
        dest = anchor + k
        lines = events[dest, 2]
        seen = seq.observed_before("line", lines, anchor)
        page_seen = seq.observed_before("page", lines >> 6, anchor)
        region_seen = seq.observed_before("region", lines >> 12, anchor)
        lead = events[dest, 3] - events[anchor, 3]
        retained = (history[observed] == (lines >> 6)[:, None]).any(axis=1)
        repeats_anchor = events[anchor, 2] == lines
        # Compare every observed arrival strictly between the trigger and the labeled target.
        intervening = events[anchor[:, None] + np.arange(1, k), 2]
        already_demanded = (intervening == lines[:, None]).any(axis=1)
        codes = (masks[dest, None] >> np.array([0, 4, 8, 12], dtype="<u2")) & 15
        miss = codes[:, 0] == 2
        gap = miss & np.isin(codes[:, 1:], [2, 6]).all(axis=1)
        structural = (events[anchor, 1] != events[dest, 1]) & ~seq.observed_before(
            "pc", events[dest, 1], anchor
        )
        bins = (
            ("seen_line", seen),
            ("new_line_known_page", ~seen & page_seen),
            ("new_page_known_region", ~page_seen & region_seen),
            ("new_region", ~region_seen),
        )
        assert np.all(np.stack([mask for _, mask in bins]).sum(axis=0) == 1)
        for name, b in bins:
            legal = b & retained
            eligible = legal & (lead >= 130)
            alternatives = 64 * distinct_pages[observed][legal]
            rows.append(
                dict(
                    case=seq.meta["case"],
                    policy=seq.meta["policy"],
                    horizon=k,
                    novelty=name,
                    observed_targets=int(b.sum()),
                    new_target_pc=int((b & structural).sum()),
                    retained_page=int(legal.sum()),
                    legal_lines_p50_when_reachable=quantile(alternatives, 0.5),
                    legal_lines_p90_when_reachable=quantile(alternatives, 0.9),
                    retained_page_lead_ge130=int(eligible.sum()),
                    original_misses=int((b & miss).sum()),
                    original_all_gap=int((b & gap).sum()),
                    original_all_gap_retained_page_lead_ge130=int(
                        (eligible & gap).sum()
                    ),
                    eligible_equals_anchor_line=int(
                        (eligible & gap & repeats_anchor).sum()
                    ),
                    eligible_intervening_demand_same_line=int(
                        (eligible & gap & already_demanded).sum()
                    ),
                    eligible_distinct_line_first_future_demand=int(
                        (eligible & gap & ~repeats_anchor & ~already_demanded).sum()
                    ),
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    for case in args.cases:
        rows = summarize(Sequence(ROOT / (case + "__stride")))
        save(HERE / "results" / (case + "__stride_novelty.csv"), rows)
