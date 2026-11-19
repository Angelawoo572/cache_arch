"""Audit the support of absolute physical-line associations in the fixed NoPF observation windows."""

import argparse
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from run import FULL, RAW, HERE, rows
from summarize import FAR, write
from timing import gzrows

U64 = (1 << 64) - 1


def index(line):
    # The repo's temporal key is the three-word tuple (physical line, 0, 0).
    z = 0
    for word in (line, 0, 0):
        z = (z * 131 + word) & U64
        z ^= z >> 19
        z ^= z >> 9
    return z & 255


def train(table, key, tag, target):
    prior = table.get(key)
    if prior is None or prior[0] != tag:
        table[key] = (tag, target, 1)
    elif prior[1] == target:
        table[key] = (tag, target, min(3, prior[2] + 1))
    elif prior[2] > 1:
        table[key] = (tag, prior[1], prior[2] - 1)
    else:
        table[key] = (tag, target, 1)


def audit(events):
    counts = Counter()
    prior_pos = {}
    prior_context = set()
    bounded = {}
    unbounded = {}
    previous = None
    for i, (pc, line) in enumerate(events):
        counts["llc_ingresses"] += 1
        context = (pc, line % 64)
        if context in prior_context:
            counts["seen_pc_offset"] += 1
        prior_context.add(context)
        if line in prior_pos:
            counts["seen_exact_line"] += 1
            if i - prior_pos[line] <= 256:
                counts["seen_within_256_events"] += 1
        prior_pos[line] = i
        if previous is not None:
            train(bounded, index(previous), previous, line)
            train(unbounded, previous, previous, line)
        for name, table, key in (
            ("bounded", bounded, index(line)),
            ("unbounded", unbounded, line),
        ):
            entry = table.get(key)
            if entry is None or entry[0] != line:
                continue
            counts[name + "_resident_key"] += 1
            if entry[2] < 2:
                continue
            counts[name + "_confident_candidate"] += 1
            if i + 1 < len(events):
                counts[name + "_evaluable"] += 1
                if events[i + 1][1] == entry[1]:
                    counts[name + "_immediate_successor_hit"] += 1
        previous = line
    return counts


def main(out, cases):
    roster = cases or [r["case"] for r in rows(FULL / "intake/selection.csv")] + list(
        FAR
    )
    fields = [
        "llc_ingresses",
        "seen_exact_line",
        "seen_within_256_events",
        "seen_pc_offset",
    ]
    fields.extend(
        name + "_" + k
        for name in ("bounded", "unbounded")
        for k in (
            "resident_key",
            "confident_candidate",
            "evaluable",
            "immediate_successor_hit",
        )
    )
    result = []
    for case in roster:
        d = RAW / "runs" / (case + "__nopf")
        if not (d / "complete.json").exists():
            raise ValueError("No completed baseline: " + case)
        starts = [int(r["first_ordinal"]) for r in rows(d / "phases.csv")]
        islands = [[] for q in starts]
        for r in gzrows(d / "loads.csv.gz"):
            if r["level"] == "LLC" and r["stage"] == "ingress":
                q = bisect_right(starts, int(r["instr_id"])) - 1
                if q < 0 or int(r["instr_id"]) - starts[q] >= 250000:
                    raise ValueError("Event outside the registered sample")
                islands[q].append((int(r["pc"]), int(r["physical_line"])))
        total = Counter()
        for q, island in enumerate(islands):
            counts = audit(island)
            result.append(
                dict(case=case, quarter=q, **{key: counts[key] for key in fields})
            )
            total.update(counts)
        result.append(
            dict(case=case, quarter="all", **{key: total[key] for key in fields})
        )
    write(out / "temporal_support.csv", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    main(args.out, args.cases)
