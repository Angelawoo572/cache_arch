"""Separate target reach from earliest frozen address identification, reference lead and actual Stride generation."""

import argparse
from collections import Counter, defaultdict
import numpy as np
from characterize import publish, SERVICES
from pattern import load, legal, snapshots, MODES
from streams import keys, events, location, parts, label
from replay import CASES


def summary(case, q, z, nohead, scope, fit):
    obs, _ = snapshots(z)
    measured = keys(z)
    fields = defaultdict(Counter)
    values = defaultdict(lambda: defaultdict(list))
    for j in range(16, len(obs)):
        if z["baseline"][j] != 0:
            continue
        target = int(z["pa"][j])
        t = int(z["t"][j])
        service = SERVICES["external"]
        source = int(z["gen"][j])
        native_lead = t - source if source >= 0 else None
        best = {}
        reach = -1
        for i in range(j - 1, j - 17, -1):
            base = int(z["pa"][i])
            if base == target:
                break
            age = t - int(z["t"][i])
            if not 0 < age <= 4096:
                continue
            ks, pages = obs[i]
            if not legal(base, target, pages):
                continue
            reach = max(reach, age)
            if i % 8 != 3:
                continue
            for mode, k in zip(MODES, ks):
                action = fit[mode].get(k)
                if action is None:
                    continue
                proposal = action[0] if mode == "temporal" else base + action[0]
                if proposal == target:
                    best[mode] = age
        for mode in MODES:
            c = fields[mode]
            x = values[mode]
            c["original_miss_targets"] += 1
            c["reachable"] += reach >= 0
            c["reach_passes_service"] += reach >= service
            if mode not in best:
                continue
            lead = best[mode]
            c["correct_on_sparse_grant"] += 1
            c["correct_but_short_lead"] += lead < service
            c["correct_adequate_lead"] += lead >= service
            x["earliest_correct"].append(lead)
            x["unrealized_from_earliest_reach"].append(reach - lead)
            if lead < service:
                continue
            c["adequate_with_positive_nopf_head"] += nohead[measured[j]] > 0
            if z["kind"][j] == 2:
                c["stride_late_target"] += 1
                if native_lead is not None:
                    c["stride_late_named_earlier"] += lead > native_lead
                    x["lead_difference_vs_stride_late"].append(lead - native_lead)
            if z["kind"][j] == 0 and z["hit"][j] == 0:
                c["persistent_stride_miss"] += 1
    rows = []
    for mode in MODES:
        row = dict(
            case=case,
            scope=scope,
            quarter=q,
            mode=mode,
            service_reference=SERVICES["external"],
            **fields[mode],
        )
        for name, seq in values[mode].items():
            for p, v in zip(("p10", "p50", "p90"), np.percentile(seq, [10, 50, 90])):
                row[name + "_" + p] = float(v)
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=("full", "diagnostic"))
    args = parser.parse_args()
    scope = args.scope
    fit = load()
    out = []
    for c in (
        *CASES,
        *(
            ("bfs_discovery_far", "mcf17_discovery_far", "pr_heldout_entire")
            if scope == "full"
            else ()
        ),
    ):
        if any(
            not (location(c, p, scope) / "complete.json").is_file()
            for p in ("nopf", "stride")
        ):
            continue
        b = events(c, "nopf", scope)
        z = label(events(c, "stride", scope), b)
        nohead = dict(zip(keys(b), b["head"]))
        for q, zq in parts(z, scope, c):
            out.extend(summary(c, q, zq, nohead, scope, fit))
    publish(scope + "_opportunity_leads", out)
