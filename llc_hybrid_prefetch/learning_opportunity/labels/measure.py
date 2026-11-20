"""Measure legal-address ceilings and lead times on a fixed random sample of all arrivals."""

import argparse
import csv
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np

from sequence import FIELDS, HORIZONS, ROOT, SCOPES, Sequence

HERE = Path(__file__).resolve().parent
SAMPLES = 32768
SEED = 448
SERVICE_CYCLES = 130


def grouped_offsets(seq, events, anchors, key):
    """Batch equivalent of relative() for the same and complementary group."""
    steps = np.array(HORIZONS, dtype=np.int64)
    shape = (len(anchors), len(steps))
    same = np.full(shape, -1, dtype=np.int64)
    other = np.full(shape, -1, dtype=np.int64)
    values = (
        events[anchors, 1]
        if key == "pc"
        else events[anchors, 2] >> (6 if key == "page" else 12)
    )
    groups = seq.keys[key].searchsorted(values)
    ordered = np.argsort(groups, kind="stable")
    runs = np.split(ordered, np.flatnonzero(np.diff(groups[ordered])) + 1)
    for rows in runs:
        g = groups[rows[0]]
        low = int(seq.ends[key][g - 1]) if g else 0
        high = int(seq.ends[key][g])
        members = seq.ranks[key][low:high]
        a = anchors[rows]
        pos = members.searchsorted(a.astype("<u4"))
        np.testing.assert_array_equal(members[pos], a)
        target = pos[:, None] + steps
        same[rows] = np.where(
            target < len(members),
            members[np.minimum(target, len(members) - 1)].astype(np.int64),
            -1,
        )
        outside_pos = (a - pos)[:, None] + steps - 1
        excluded = members.astype(np.int64) - np.arange(len(members))
        rank = outside_pos + excluded.searchsorted(outside_pos, side="right")
        other[rows] = np.where(rank < seq.size, rank, -1)
    return same, other


def sample(seq):
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(seq.size, size=min(SAMPLES, seq.size), replace=False))


def quantile(data, q):
    return int(np.quantile(data, q, method="inverted_cdf")) if len(data) else ""


def full_arrays(seq):
    events = np.empty((seq.size, len(FIELDS)), dtype="<u8")
    labels = np.empty(seq.size, dtype="<u2")
    for b, (lo, hi, _, _) in enumerate(seq.meta["chunks"]):
        events[lo:hi] = seq._events(b)
        labels[lo:hi] = seq._labels(b)
    return events, labels


def measure(seq):
    events, labels = full_arrays(seq)
    anchors = sample(seq)
    gathered = {}
    global_ranks = anchors[:, None] + np.array(HORIZONS)
    gathered["global"] = np.where(global_ranks < seq.size, global_ranks, -1)
    for key in ("pc", "page", "region"):
        gathered["same_" + key], gathered["other_" + key] = grouped_offsets(
            seq, events, anchors, key
        )
    for scope in SCOPES:
        for i in (0, len(anchors) // 2, len(anchors) - 1):
            np.testing.assert_array_equal(
                gathered[scope][i], seq.relative(int(anchors[i]), HORIZONS, scope)
            )
    # This vocabulary retains the pages of the latest 64 arrivals including the current event.
    history = events[np.maximum(0, anchors[:, None] - np.arange(64)), 2] >> 6
    results = []
    for scope, future in gathered.items():
        for col, step in enumerate(HORIZONS):
            observed = future[:, col] >= 0
            a, dest = anchors[observed], future[observed, col]
            line, page, region = (
                events[dest, 2],
                events[dest, 2] >> 6,
                events[dest, 2] >> 12,
            )
            samepage = events[a, 2] >> 6 == page
            in_recent = (history[observed] == page[:, None]).any(axis=1)
            seen = seq.observed_before("line", line, a)
            pg_seen = seq.observed_before("page", page, a)
            rg_seen = seq.observed_before("region", region, a)
            diff_pc = events[a, 1] != events[dest, 1]
            lead = (events[dest, 3] - events[a, 3]).astype(np.int64)
            distance = dest - a
            outcomes = (labels[dest, None] >> np.array([0, 4, 8, 12], dtype="<u2")) & 15
            original = outcomes[:, 0] == 2
            late_or_miss = np.isin(outcomes[:, 1:], [2, 6]).all(axis=1)
            original_gap = original & late_or_miss
            potential = in_recent & (lead >= SERVICE_CYCLES)
            assert not np.any(
                (in_recent & ~pg_seen) | (seen & ~pg_seen) | (pg_seen & ~rg_seen)
            )
            results.append(
                dict(
                    case=seq.meta["case"],
                    policy=seq.meta["policy"],
                    scope=scope,
                    horizon=step,
                    total_arrivals=seq.size,
                    anchors=len(anchors),
                    observed=len(a),
                    censored=len(anchors) - len(a),
                    same_page=int(samepage.sum()),
                    previously_seen_exact_line=int(seen.sum()),
                    new_line_known_page=int((~seen & pg_seen).sum()),
                    new_page_known_region=int((~pg_seen & rg_seen).sum()),
                    new_region=int((~rg_seen).sum()),
                    page_in_last64=int(in_recent.sum()),
                    different_pc=int(diff_pc.sum()),
                    target_pc_previously_unseen=int(
                        (~seq.observed_before("pc", events[dest, 1], a)).sum()
                    ),
                    median_intervening_arrivals=quantile(distance - 1, 0.5),
                    lead_p10_cycles=quantile(lead, 0.1),
                    lead_p50_cycles=quantile(lead, 0.5),
                    lead_p90_cycles=quantile(lead, 0.9),
                    legal_lead_ge64=int((in_recent & (lead >= 64)).sum()),
                    legal_lead_ge130=int(potential.sum()),
                    legal_lead_ge256=int((in_recent & (lead >= 256)).sum()),
                    original_miss=int(original.sum()),
                    original_and_all_three_miss_or_late=int(original_gap.sum()),
                    original_all_gap_legal_lead_ge130=int(
                        (original_gap & potential).sum()
                    ),
                    original_stride_timely=int(
                        (original & (outcomes[:, 1] == 9)).sum()
                    ),
                    original_stride_late=int((original & (outcomes[:, 1] == 6)).sum()),
                    original_multi_timely=int((original & (outcomes[:, 2] == 9)).sum()),
                    original_multi_late=int((original & (outcomes[:, 2] == 6)).sum()),
                    original_spp_timely=int((original & (outcomes[:, 3] == 9)).sum()),
                    original_spp_late=int((original & (outcomes[:, 3] == 6)).sum()),
                )
            )
    footprints = []
    for width in (32, 128, 512):
        deadline = events[anchors, 3] + width
        complete = deadline < events[-1, 3]
        a = anchors[complete]
        end = events[:, 3].searchsorted(deadline[complete], side="right")
        lines = np.array([len(np.unique(events[i + 1 : j, 2])) for i, j in zip(a, end)])
        assert np.all(lines <= end - a - 1)
        footprints.append(
            dict(
                case=seq.meta["case"],
                policy=seq.meta["policy"],
                window_cycles=width,
                anchors=len(anchors),
                complete=len(a),
                censored=len(anchors) - len(a),
                empty=int((lines == 0).sum()),
                total_future_events=int((end - a - 1).sum()),
                sum_distinct_lines=int(lines.sum()),
                distinct_lines_p10=quantile(lines, 0.1),
                distinct_lines_p50=quantile(lines, 0.5),
                distinct_lines_p90=quantile(lines, 0.9),
                distinct_lines_p99=quantile(lines, 0.99),
            )
        )
    return results, footprints


def save(path, rows):
    with NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", delete=False, newline=""
    ) as f:
        tmp = Path(f.name)
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--policy", default="stride")
    args = parser.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    for case in args.cases:
        name = f"{case}__{args.policy}"
        opportunities, windows = measure(Sequence(ROOT / name))
        save(out / f"{name}_horizons.csv", opportunities)
        save(out / f"{name}_windows.csv", windows)
        print(case, flush=True)
