"""Quantify signed-address generation, conditional futures, transitions and multi-stream structure."""

import argparse
from collections import Counter, defaultdict
import json
from pattern import (
    CASES as DONORS,
    HERE,
    MODES,
    eligible,
    hist,
    legal,
    load,
    save,
    snapshots,
)
from streams import events, keys, label, parts, location
from replay import CASES

SERVICES = json.loads((HERE / "service.json").read_text())
NATIVE = ("stride", "multi", "spp")


def target_kind(z, j):
    if j is None:
        return "absent_ingress"
    return (
        "timely"
        if z["kind"][j] == 1
        else "late"
        if z["kind"][j] == 2
        else "miss"
        if z["hit"][j] == 0
        else "resident_hit"
        if z["hit"][j] == 1
        else "no_tag"
    )


def forests():
    """Separate conditional branch test at identical event distances, same split and capacity."""
    result = {}
    counts = defaultdict(Counter)
    targets = defaultdict(lambda: defaultdict(Counter))
    for case in DONORS:
        z = hist(case, "stride")
        obs, _ = snapshots(z)
        end = int(0.6 * len(obs))
        for i in range(end - 16):
            keyset, pages = obs[i]
            valid = dict(
                (j, d) for d, j in eligible(z, i, end, SERVICES["donors"][case], pages)
            )
            for h in (1, 4, 16):
                for mode, key in zip(MODES[:3], keyset[:3]):
                    k = (mode, h)
                    counts[k][key] += 1
                    if i + h in valid:
                        targets[k][key][valid[i + h]] += 1
    for k, count in counts.items():
        top = {}
        for key, n in count.items():
            pos = targets[k][key]
            if n < 4 or not pos:
                continue
            a, p = min(pos.items(), key=lambda v: (-v[1], abs(v[0]), v[0]))
            if p >= 2 and 5 * p >= n:
                top[key] = (a, p, n)
        result[k] = dict(sorted(top.items(), key=lambda x: (-x[1][2], x[0]))[:256])
    return result


def score(
    case, scope, quarter, z, arms, start, end, tables, fixed, service, donor_pcs, split
):
    obs, first = snapshots(z)
    ids = keys(z)
    idx = {p: {key: i for i, key in enumerate(keys(e))} for p, e in arms.items()}
    visits = {m: Counter() for m in (*MODES, "sparse1")}
    metrics = Counter()
    # Count baseline-attributed miss/hit runs independently by PC, page and ingress order.
    last = {}
    timed = defaultdict(set)
    orig = Counter()
    for i in range(start, end):
        if int(z["baseline"][i]) != 0:
            continue
        outcome = target_kind(z, i)
        pc = int(z["pc"][i])
        page = int(z["pa"][i]) // 64
        for view, group in (("global", 0), ("pc", pc), ("page", page)):
            k = (view, group)
            orig[k] += 1
            if outcome == "timely":
                timed[view].add(group)
            if last.get(k) == "timely":
                metrics[view + "_after_timely_" + outcome] += 1
            last[k] = outcome
    for (view, group), n in orig.items():
        metrics[view + "_streams"] += 1
        if group not in timed[view]:
            metrics[view + "_silent_streams"] += 1
            metrics[view + "_misses_in_silent_streams"] += n
    older = {}
    branches = defaultdict(dict)
    examples = {}
    train_pcs = donor_pcs
    legal_targets = set()
    strong_targets = set()
    for i in range(start, max(start, end - 16)):
        ks, pages = obs[i]
        pc = int(z["pc"][i])
        line = int(z["pa"][i])
        at = int(z["t"][i])
        prev = older.get(pc)
        if (
            prev is not None
            and i - prev > 1
            and ks[0][2] == ks[0][3]
            and 0 < abs(ks[0][2] - 64) < 64
        ):
            if line - int(z["pa"][i - 1]) != ks[0][2] - 64:
                metrics["interleaved_local_repeat"] += 1
        older[pc] = i
        metrics["anchors"] += 1
        metrics["pc_absent_in_training"] += pc not in train_pcs
        future = dict(eligible(z, i, end, service, pages))
        for j in future.values():
            legal_targets.add(ids[j])
        if future:
            metrics["anchors_with_legal_original_miss_and_lead"] += 1
        if i % 8 != 3:
            continue
        for mode, key in zip((*MODES, "sparse1"), (*ks, ())):
            c = visits[mode]
            c["grants"] += 1
            entry = (1, 0, 0) if mode == "sparse1" else tables[mode].get(key)
            if entry is None:
                continue
            c["context_supported"] += 1
            d = entry[0] - line if mode == "temporal" else entry[0]
            target = line + d
            if not legal(line, target, pages) or (
                mode == "sparse1" and line // 64 != target // 64
            ):
                continue
            c["legal_offers"] += 1
            if d not in future:
                continue
            j = future[d]
            target_key = ids[j]
            c["original_matches_with_lead"] += 1
            c["new_line_since_sample_start"] += int(first[target] > i)
            c["cross_page_match"] += int(line // 64 != target // 64)
            statuses = (
                [target_kind(arms[p], idx[p].get(target_key)) for p in NATIVE]
                if arms
                else []
            )
            gap = bool(statuses) and all(s in ("miss", "late") for s in statuses)
            c["persistent_or_late_all_three"] += gap
            c["unseen_pc_match"] += pc not in train_pcs
            if gap:
                strong_targets.add(target_key)
                cat = (
                    mode,
                    "cross_page" if line // 64 != target // 64 else "same_page",
                )
                if cat not in examples:
                    examples[cat] = dict(
                        mode=mode,
                        cue_instr=ids[i][0],
                        cue_op=ids[i][1],
                        pc=pc,
                        cue_line=line,
                        cue_key=" ".join(map(str, key)),
                        target_instr=target_key[0],
                        target_op=target_key[1],
                        target_line=target,
                        decision_cycle=at,
                        demand_cycle=int(z["t"][j]),
                        lead=int(z["t"][j]) - at,
                        events=j - i,
                        observed_service_reference=service,
                        head=int(z["head"][j]),
                        new_since_window=first[target] > i,
                        stride=statuses[0],
                        multi=statuses[1],
                        spp=statuses[2],
                        training_support=entry[2],
                        action_support=entry[1],
                    )
        for h in (1, 4, 16):
            j = i + h
            d = int(z["pa"][j]) - line
            if future.get(d) != j:
                continue
            pred = [fixed[(m, h)].get(k) for m, k in zip(MODES[:3], ks[:3])]
            for m, r in zip(MODES[:3], pred):
                if r is not None and r[0] == d:
                    metrics[f"h{h}_{m}_correct"] += 1
            if pred[1] is None or pred[1][0] != d:
                continue
            metrics["ordered_only_fixed_horizon"] += pred[0] is None or pred[0][0] != d
            k = (ks[0], h)
            event = ids[j]
            if d not in branches[k]:
                branches[k][d] = dict(
                    horizon=h,
                    local=" ".join(map(str, ks[0])),
                    older=" ".join(map(str, ks[1][-3:])),
                    delta=d,
                    pc=pc,
                    cue_instr=ids[i][0],
                    cue_op=ids[i][1],
                    target_instr=event[0],
                    target_op=event[1],
                    cue_line=line,
                    target_line=line + d,
                    cycles=int(z["t"][j]) - at,
                    training_count=pred[1][2],
                    votes=pred[1][1],
                )
    metrics["distinct_legal_original_targets"] = len(legal_targets)
    metrics["distinct_matched_targets_missed_or_late_all_three"] = len(strong_targets)
    amb = []
    source = dict(case=case, scope=scope, quarter=quarter, split=split)
    for group in branches.values():
        if len(group) >= 2:
            metrics["conditional_local_contexts"] += 1
            if len(amb) < 8:
                amb.extend(dict(source, **e) for e in list(group.values())[:2])
    metrics["stride_ingresses"] = end - start
    metrics["sampled_original_miss_ingresses"] = sum(
        orig[k] for k in orig if k[0] == "global"
    )
    return (
        [dict(source, mode=m, **c) for m, c in visits.items()],
        dict(source, **metrics),
        [dict(source, **e) for e in examples.values()],
        amb,
    )


def publish(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    if not fields:
        fields = ["case", "scope", "quarter", "split"]
    save(HERE / (name + ".csv"), rows, fields)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scope", choices=("diagnostic", "full", "source"), required=True)
    cli.add_argument("--cases", nargs="+", default=CASES)
    a = cli.parse_args()
    if a.scope == "full" and a.cases == CASES:
        a.cases = (
            *CASES,
            "bfs_discovery_far",
            "mcf17_discovery_far",
            "pr_heldout_entire",
        )
    tables = load()
    fixed = forests()
    allrows = [[], [], [], []]
    donor_pcs = set()
    for case in DONORS:
        z = hist(case, "stride")
        donor_pcs.update(int(x) for x in z["pc"][: int(0.6 * len(z["t"]))])
    for case in DONORS if a.scope == "source" else a.cases:
        work = []
        if a.scope == "source":
            z = hist(case, "stride")
            z["kind"] = z["pf_kind"]
            n = len(z["t"])
            service = SERVICES["donors"][case]
            for lo, hi, split in ((0.6, 0.8, "development"), (0.8, 1, "source_test")):
                work.append((1, z, {}, int(lo * n), int(hi * n), service, split))
        else:
            if not all(
                (location(case, p, a.scope) / "complete.json").exists()
                for p in ("nopf", *NATIVE)
            ):
                continue
            b = events(case, "nopf", a.scope)
            arms = {p: label(events(case, p, a.scope), b) for p in NATIVE}
            divided = {p: dict(parts(v, a.scope, case)) for p, v in arms.items()}
            for q, z in divided["stride"].items():
                work.append(
                    (
                        q,
                        z,
                        {p: divided[p][q] for p in NATIVE},
                        0,
                        len(z["t"]),
                        SERVICES["external"],
                        "external",
                    )
                )
        for q, z, arms, lo, hi, service, split in work:
            results = score(
                case,
                a.scope,
                q,
                z,
                arms,
                lo,
                hi,
                tables,
                fixed,
                service,
                donor_pcs,
                split,
            )
            for group, records in zip(allrows, results):
                group.extend([records] if isinstance(records, dict) else records)
        print(case, len(work), "parts", flush=True)
    for suffix, rows in zip(
        ("scores", "populations", "examples", "conditional"), allrows
    ):
        publish(a.scope + "_" + suffix, rows)


if __name__ == "__main__":
    main()
