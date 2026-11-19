"""Within- and cross-recording tests on fixed islands, including an explicit candidate-reach ceiling."""

from collections import Counter, defaultdict
import argparse
import numpy as np
from pattern import snapshots, legal
from streams import location, label, events, parts, keys
from characterize import publish
from replay import CASES

H = (1, 4, 16)
MODES = ("local", "ordered", "shape", "bag", "anonymous_local")


def phase_keys(ks):
    return (*ks[:4], ks[5])


def readcase(case):
    if not all(
        (location(case, p, "full") / "complete.json").is_file()
        for p in ("nopf", "stride", "multi", "spp")
    ):
        return None
    b = events(case, "nopf", "full")
    return {
        p: dict(
            parts(label(b if p == "nopf" else events(case, p, "full"), b), "full", case)
        )
        for p in ("nopf", "stride", "multi", "spp")
    }


def future(z, i, h, service, pages):
    j = i + h
    target = int(z["pa"][j])
    base = int(z["pa"][i])
    t = int(z["t"][j] - z["t"][i])
    if (
        z["baseline"][j] != 0
        or target in z["pa"][i + 1 : j]
        or not service <= t <= 4096
        or not legal(base, target, pages)
    ):
        return None
    return target - base


def fit(z, service):
    counts = defaultdict(lambda: defaultdict(Counter))
    hits = defaultdict(Counter)
    obs, _ = snapshots(z)
    for i in range(len(obs) - 16):
        ks, pages = obs[i]
        for h in H:
            action = future(z, i, h, service, pages)
            for mode, key in zip(MODES, phase_keys(ks)):
                at = (mode, h)
                hits[at][key] += 1
                if action is not None:
                    counts[at][key][action] += 1
    tables = {}
    for at, hits_ in hits.items():
        eligible = []
        for k, n in hits_.items():
            choices = counts[at][k]
            if n < 4 or not choices:
                continue
            d, v = min(choices.items(), key=lambda x: (-x[1], abs(x[0]), x[0]))
            if v >= 2 and 5 * v >= n:
                eligible.append((k, n, d, v))
        tables[at] = {
            k: (d, n, v)
            for k, n, d, v in sorted(eligible, key=lambda x: (-x[1], x[0]))[:256]
        }
    return tables


def percentiles(seq):
    return [float(x) for x in np.percentile(seq, [10, 50, 90])] if seq else ["", "", ""]


def reaches(z, service):
    obs, first = snapshots(z)
    c = Counter()
    max_leads = []
    max_events = []
    for j in range(16, len(obs)):
        if z["baseline"][j] != 0:
            continue
        target = int(z["pa"][j])
        now = int(z["t"][j])
        c["original_miss_targets"] += 1
        got = set()
        leads = []
        ages = []
        for i in range(j - 1, j - 17, -1):
            source = int(z["pa"][i])
            gap = now - int(z["t"][i])
            page = source // 64
            if source == target:
                break
            if gap <= 0 or gap > 4096:
                continue
            d = target - source
            if abs(d) > 512:
                continue
            got.add("within_displacement")
            if target // 64 == page:
                got.add("same_page")
            if target // 64 not in obs[i][1]:
                continue
            got.add("expressible")
            if gap < service:
                continue
            got.add("enough_lead")
            if target // 64 != page:
                got.add("cross_page_with_lead")
            if first[target] > i:
                got.add("fresh_line_with_lead")
            leads.append(gap)
            ages.append(j - i)
        for k in got:
            c[k] += 1
        if leads:
            max_leads.append(max(leads))
            max_events.append(max(ages))
    c.update(
        dict(
            zip(
                ("early_lead_p10", "early_lead_p50", "early_lead_p90"),
                percentiles(max_leads),
            )
        )
    )
    c.update(
        dict(
            zip(
                ("early_distance_p10", "early_distance_p50", "early_distance_p90"),
                percentiles(max_events),
            )
        )
    )
    return c


def evaluate(z, arms, service, tables, case, donor, q):
    source = dict(case=case, donor=donor, quarter=q, service=service)
    scores = defaultdict(Counter)
    branches = defaultdict(dict)
    obs, first = snapshots(z)
    ids = keys(z)
    # Build vetoes by operand on all three independently timed streams; actual PA must align.
    gaps = set()
    for p, a in arms.items():
        fail = {
            k
            for j, k in enumerate(keys(a))
            if a["kind"][j] != 1 and (a["hit"][j] == 0 or a["kind"][j] == 2)
        }
        gaps = fail if p == "stride" else gaps & fail
    for i in range(len(obs) - 16):
        ks, pages = obs[i]
        base = int(z["pa"][i])
        for h in H:
            j = i + h
            target = int(z["pa"][j])
            correct = future(z, i, h, service, pages)
            gap = ids[j] in gaps and correct is not None
            choices = [
                tables.get((m, h), {}).get(k) for m, k in zip(MODES, phase_keys(ks))
            ]
            local = bool(choices[0] and correct == choices[0][0])
            for mode, choice in zip(MODES, choices):
                c = scores[(h, mode)]
                c["anchors"] += 1
                c["legal_original_with_lead"] += correct is not None
                c["gap_with_lead"] += gap
                win = bool(choice and correct == choice[0])
                c["supported"] += choice is not None
                c["issued_legally"] += bool(
                    choice and legal(base, base + choice[0], pages)
                )
                c["matches"] += win
                c["gap_matches"] += win and gap
                c["gained_gap_vs_local"] += win and not local and gap
                c["lost_gap_vs_local"] += local and not win and gap
                c["matches_to_new_line"] += win and first[target] > i
                if mode == "ordered" and win:
                    key = (h, ks[0])
                    records = branches[key]
                    if correct not in records or (
                        gap and not records[correct]["unserved"]
                    ):
                        records[correct] = dict(
                            source,
                            horizon=h,
                            local=" ".join(map(str, ks[0])),
                            previous_pcs=" ".join(map(str, ks[1][-3:])),
                            cue_instr=ids[i][0],
                            cue_operand=ids[i][1],
                            cue_pc=int(z["pc"][i]),
                            cue_line=base,
                            cue_cycle=int(z["t"][i]),
                            action=correct,
                            target_instr=ids[j][0],
                            target_operand=ids[j][1],
                            target_line=target,
                            target_cycle=int(z["t"][j]),
                            train_support=choice[1],
                            votes=choice[2],
                            unserved=int(gap),
                            head=int(z["head"][j]),
                        )
    rows = [
        dict(source, mode=m, horizon=h, fit_entries=len(tables.get((m, h), {})), **c)
        for (h, m), c in scores.items()
    ]
    examples = []
    contexts = 0
    for targets in branches.values():
        if len(targets) < 2:
            continue
        contexts += 1
        if len(examples) < 12:
            examples.extend(
                sorted(
                    targets.values(), key=lambda r: (-r["unserved"], r["target_instr"])
                )[:2]
            )
    return (
        rows,
        examples,
        dict(source, conditional_context_horizons=contexts, **reaches(z, service)),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "cases",
        nargs="*",
        default=(
            *CASES,
            "bfs_discovery_far",
            "mcf17_discovery_far",
            "pr_heldout_entire",
        ),
    )
    a = cli.parse_args()
    frozen = {}
    allrows = [[], [], []]
    for donor in ("bfs_discovery", "mcf17_discovery"):
        inp = readcase(donor)
        if inp is None:
            continue
        z = inp["nopf"][1]
        mask = (
            (z["dispatch"] >= 0)
            & (z["fill"] >= z["dispatch"])
            & (z["fill"] <= z["t"][-1])
        )
        if not mask.any():
            continue
        service = int(np.median((z["fill"] - z["dispatch"])[mask]))
        frozen[donor] = (service, fit(inp["stride"][1], service))
    for case in a.cases:
        donor = (
            "bfs_discovery" if case.startswith(("bfs", "pr_")) else "mcf17_discovery"
        )
        inp = readcase(case)
        if inp is None or donor not in frozen:
            continue
        for q, z in inp["stride"].items():
            if case == donor and q == 1:
                continue
            arms = {p: inp[p][q] for p in ("stride", "multi", "spp")}
            x = evaluate(z, arms, *frozen[donor], case, donor, q)
            for store, rows in zip(allrows, x):
                store.extend(rows if isinstance(rows, list) else [rows])
        print(case, flush=True)
    for name, rows in zip(
        ("phase_transfer", "phase_conditional_examples", "phase_reach"), allrows
    ):
        publish(name, rows)
