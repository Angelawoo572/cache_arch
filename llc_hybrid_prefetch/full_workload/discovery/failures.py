"""Explain sampled original misses and each immediate post-timely outcome with native producer evidence."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from streams import events, label, location, lines, parts
from characterize import publish
from replay import CASES


def proposals(folder):
    res = {}
    by = defaultdict(list)
    data = lines(folder, "candidates")
    h = next(data)
    for r in data:
        id_ = int(r[h["id"]])
        stage = r[h["stage"]]
        if stage == "proposed":
            rec = dict(
                target=int(r[h["target"]]),
                start=int(r[h["cycle"]]),
                status="proposed",
                resolved=-1,
                reason="",
                later=None,
            )
            res[id_] = rec
            by[rec["target"]].append(rec)
        elif id_ in res:
            a = res[id_]
            if stage in ("accepted", "rejected"):
                a.update(
                    status=stage, resolved=int(r[h["cycle"]]), reason=r[h["reason"]]
                )
            if stage in ("redundant", "evicted", "timely", "late"):
                a["later"] = (int(r[h["cycle"]]), stage)
    return {line: ([a["start"] for a in values], values) for line, values in by.items()}


def explain(qrows, mapped):
    a = qrows
    c = Counter()
    previous = {}
    prev_line = {}
    for j in range(len(a["instr"])):
        line = int(a["pa"][j])
        same = prev_line.get(line, -1)
        prev_line[line] = j
        if j < 16 or int(a["baseline"][j]) != 0:
            continue
        pc = int(a["pc"][j])
        hit = int(a["hit"][j])
        kind = int(a["kind"][j])
        now = int(a["t"][j])
        if kind == 1:
            state = "attributed_timely"
        elif kind == 2:
            state = "attributed_late"
        elif hit == 1:
            state = "ordinary_hit"
        elif hit < 0:
            state = "no_independent_tag"
        else:
            earliest = int(a["t"][max(j - 16, same)])
            times, events_ = mapped.get(line, ([], []))
            ix = bisect_left(times, now) - 1
            if ix < 0 or times[ix] < earliest:
                state = "miss_no_prior_offer_in_span"
            else:
                item = events_[ix]
                if item["resolved"] < 0 or item["resolved"] >= now:
                    state = "miss_prior_offer_not_admitted_before_demand"
                elif item["status"] == "rejected":
                    state = "miss_prior_offer_rejected_" + item["reason"]
                elif item["later"] and item["later"][0] < now:
                    state = "miss_prior_admitted_" + item["later"][1]
                else:
                    state = "miss_prior_admitted_without_matching_first_use"
        for view, k in (("global", 0), ("pc", pc), ("physical_page", line // 64)):
            last = previous.get((view, k))
            previous[(view, k)] = state
            c[(view, "all", state)] += 1
            if last == "attributed_timely":
                c[(view, "immediately_after_timely", state)] += 1
    return c


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scope", choices=("diagnostic", "full"), required=True)
    cli.add_argument("--cases", nargs="+")
    a = cli.parse_args()
    out = []
    cases = a.cases or (
        *CASES,
        *(
            ("bfs_discovery_far", "mcf17_discovery_far", "pr_heldout_entire")
            if a.scope == "full"
            else ()
        ),
    )
    for case in cases:
        if not all(
            (location(case, p, a.scope) / "complete.json").exists()
            for p in ("stride", "nopf")
        ):
            continue
        b = events(case, "nopf", a.scope)
        z = label(events(case, "stride", a.scope), b)
        cand = proposals(location(case, "stride", a.scope))
        for quarter, r in parts(z, a.scope, case):
            for (view, previous, kind), count in sorted(explain(r, cand).items()):
                out.append(
                    dict(
                        case=case,
                        scope=a.scope,
                        quarter=quarter,
                        view=view,
                        cohort=previous,
                        outcome=kind,
                        occurrences=count,
                    )
                )
    publish(a.scope + "_transition_causes", out)
