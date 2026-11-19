"""Full intersections of displaced and retained successes on original NoPF demand misses."""

import argparse
import csv
from itertools import combinations
from pathlib import Path
import numpy as np

from run import DEFAULT_OUTPUT, HERE, POLICIES, select

CLASSES = ("direct_timely", "late", "persistent", "ordinary_hit", "no_tag")


def load(root, case, policy):
    with np.load(root / "history" / case / (policy + ".npz")) as f:
        return {k: f[k] for k in f.files}


def med(values):
    v = values[np.isfinite(values)]
    return float(np.median(v)) if len(v) else ""


def frame(root, case):
    base = load(root, case, "nopf")
    miss = base["hit"] == 0
    original = base["instr"][miss] * 256 + base["op"][miss]
    ordered = np.argsort(original)
    original = original[ordered]
    data = {}
    for policy in POLICIES[1:]:
        q = load(root, case, policy)
        rows = np.flatnonzero(q["baseline"] == 0)
        keys = q["instr"][rows] * 256 + q["op"][rows]
        pos = np.searchsorted(original, keys)
        assert np.array_equal(original[pos], keys)
        cls = np.full(len(original), 4)
        codes = np.select(
            [
                (q["hit"] == 1) & (q["pf_kind"] == 1),
                (q["hit"] == 0) & (q["pf_kind"] == 2),
                q["hit"] == 0,
                q["hit"] == 1,
            ],
            [0, 1, 2, 3],
            default=4,
        )
        cls[pos] = codes[rows]
        pre = (q["pf_gen"] >= 0) & (q["pf_fill"] >= 0)
        lead = np.full(len(original), np.nan)
        lat = lead.copy()
        w = lead.copy()
        usable = pre[rows]
        lead[pos[usable]] = q["t"][rows[usable]] - q["pf_gen"][rows[usable]]
        lat[pos[usable]] = q["pf_fill"][rows[usable]] - q["pf_gen"][rows[usable]]
        w[pos] = q["head"][rows]
        data[policy] = cls, lead, lat, w
    return data


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a.add_argument("--summaries", type=Path, default=HERE)
    a.add_argument("--case", nargs="+")
    args = a.parse_args()
    if args.output.resolve() != DEFAULT_OUTPUT and args.summaries.resolve() == HERE:
        a.error("Use a separate summary directory for an alternate window.")
    output = []
    for row in select(args.case):
        case = row["case"]
        data = frame(args.output, case)
        for a, b in combinations(POLICIES[1:], 2):
            ca, la, ta, wa = data[a]
            cb, lb, tb, wb = data[b]
            for ac, aname in enumerate(CLASSES):
                for bc, bname in enumerate(CLASSES):
                    ids = (ca == ac) & (cb == bc)
                    output.append(
                        dict(
                            case=case,
                            policy_a=a,
                            policy_b=b,
                            a=aname,
                            b=bname,
                            demands=int(ids.sum()),
                            a_p50_generation_to_ingress=med(la[ids]),
                            b_p50_generation_to_ingress=med(lb[ids]),
                            a_p50_generation_to_fill=med(ta[ids]),
                            b_p50_generation_to_fill=med(tb[ids]),
                            a_p50_operand_head=med(wa[ids]),
                            b_p50_operand_head=med(wb[ids]),
                        )
                    )
    args.summaries.mkdir(exist_ok=True, parents=True)
    with (args.summaries / "paired_outcomes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(output)


if __name__ == "__main__":
    main()
