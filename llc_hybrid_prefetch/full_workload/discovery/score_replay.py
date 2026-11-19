"""Check native proposals against a prefix-only independent reader and audit paired first uses."""

import argparse
from collections import Counter
import gzip
import numpy as np
from pattern import table, snapshots, load, legal
from streams import events, keys, location, label, lines
from replay import CASES, POLICIES
from characterize import target_kind, publish


def accepted(folder):
    rec = lines(folder, "candidates")
    h = next(rec)
    items = {}
    hits = {}
    for row in rec:
        stage = row[h["stage"]]
        ident = row[h["id"]]
        if stage == "proposed":
            items[ident] = dict(
                source=row[h["reason"]],
                cue=int(row[h["instr_id"]]),
                op=int(row[h["operand"]]),
                line=int(row[h["base"]]),
                target=int(row[h["target"]]),
                gen=int(row[h["cycle"]]),
                dispatch=-1,
                fill=-1,
            )
        elif ident in items:
            a = items[ident]
            if stage in ("dispatch", "fill"):
                a[stage] = int(row[h["cycle"]])
            if stage in ("timely", "late"):
                key = (int(row[h["demand_instr_id"]]), int(row[h["demand_operand"]]))
                hits[key] = a
    return hits


def validate_offers(folder, z, mode, fit):
    if mode not in ("local", "ordered", "shape", "sparse1", "sparse4", "shadow"):
        return {}
    obs, _ = snapshots(z)
    it = lines(folder, "choices")
    h = next(it)
    choices = Counter()
    decisions = {}
    events_by_cue = lines(folder, "candidates")
    header = next(events_by_cue)
    for event in events_by_cue:
        stage = event[header["stage"]]
        if stage in ("accepted", "rejected"):
            key = (int(event[header["instr_id"]]), int(event[header["operand"]]))
            decisions[key] = (stage, event[header["reason"]])
    for i, (record, (ks, pages)) in enumerate(zip(it, obs, strict=True)):
        instr, op = int(z["instr"][i]), int(z["op"][i])
        assert (int(record[h["instr_id"]]), int(record[h["operand"]])) == (instr, op)
        line = int(z["pa"][i])
        stride = int(record[h["stride"]])
        choice = int(record[h["chosen"]])
        expected = stride
        if i % 8 == 3 and mode != "shadow":
            choices["grants"] += 1
            if mode in ("sparse1", "sparse4"):
                entry = (1 if mode == "sparse1" else 4, 0, 0)
            else:
                entry = fit[mode].get(ks[("local", "ordered", "shape").index(mode)])
            if entry:
                target = line + entry[0]
                if legal(line, target, pages) and (
                    mode not in ("sparse1", "sparse4") or target // 64 == line // 64
                ):
                    choices["independent_legal"] += 1
                    if stride == target:
                        choices["agrees"] += 1
                    elif stride:
                        choices["different_native_proposal_displaced"] += 1
                        stage, reason = decisions[(instr, op)]
                        choices["displaced_then_" + stage] += 1
                        if stage == "rejected":
                            choices["displaced_rejected_" + reason] += 1
                    else:
                        choices["native_was_silent"] += 1
                    expected = target
        if choice != expected:
            raise ValueError(
                f"Python/native proposal mismatch {mode} at {instr}/{op}: {expected} vs {choice}"
            )
    return choices


def compressed_join(folders, ref):
    # The two bytes per instruction encode four independent operand nibbles. Demand lookup hit/miss
    # and first-use provenance are independent bits. Iterate each operand in bounded chunks.
    from contextlib import ExitStack

    totals = Counter()
    with ExitStack() as stack:
        streams = [
            stack.enter_context(gzip.open(folders[m] / "outcomes_core0.u16le.gz", "rb"))
            for m in ("nopf", ref, "shape")
        ]
        while True:
            chunks = [f.read(2 * 1024 * 1024) for f in streams]
            if len(set(map(len, chunks))) != 1:
                raise ValueError("unequal outcome stream length")
            if not chunks[0]:
                break
            arrays = [np.frombuffer(b, dtype="<u2") for b in chunks]
            totals["instructions_joined"] += len(arrays[0])
            for shift in (0, 4, 8, 12):
                base, r, a = [(v >> shift) & 15 for v in arrays]
                orig = (base & 2) != 0
                ht = (a & 8) != 0
                rt = (r & 8) != 0
                totals["original_misses"] += int(np.sum(orig))
                totals["original_timely_kept"] += int(np.sum(orig & ht & rt))
                totals["original_timely_gained"] += int(np.sum(orig & ht & ~rt))
                totals["original_timely_lost"] += int(np.sum(orig & ~ht & rt))
                totals["original_late_to_timely"] += int(
                    np.sum(orig & ht & ((r & 4) != 0))
                )
    return totals


def analyze(case, scope, tables):
    opts = (
        list(POLICIES)
        if scope == "diagnostic"
        else ["nopf", "stride", "multi", "spp", "shape"]
    )
    paths = {p: location(case, p, scope) for p in opts}
    if any(not (f / "complete.json").exists() for f in paths.values()):
        return [], [], [], []
    b = events(case, "nopf", scope)
    prepared = {p: label(events(case, p, scope), b) for p in opts}
    paired = {p: dict(zip(keys(z), range(len(z["t"])))) for p, z in prepared.items()}
    bmap = paired["nopf"]
    eligible = {k for k, j in bmap.items() if b["hit"][j] == 0}
    coredata = {p: table(f / "cores.csv")[0] for p, f in paths.items()}
    sample = "all_diagnostic_loads" if scope == "diagnostic" else "four_250K_islands"
    perf = []
    groups = []
    witnesses = []
    complete = []
    for p, z in prepared.items():
        n = int(coredata[p]["instructions"])
        cycles = int(coredata[p]["cycles"])
        counts = {r["key"]: int(r["value"]) for r in table(paths[p] / "stats.csv")}
        check = validate_offers(paths[p], z, p, tables) if scope == "diagnostic" else {}
        perf.append(
            dict(
                case=case,
                scope=scope,
                arm=p,
                instructions=n,
                cycles=cycles,
                ipc=n / cycles,
                savings_vs_stride_pct=100
                * (1 - cycles / int(coredata["stride"]["cycles"])),
                head_unique_cycles=counts["head_unique_cycles_core_0"],
                dram_demand_bytes=counts["dram_demand_bytes"],
                dram_prefetch_bytes=counts["dram_prefetch_bytes"],
                accepted=counts["prefetch_accepted"],
                independent_different_proposals=counts.get("study_chosen_frozen", 0),
                timely=counts.get("study_pf_timely", 0),
                late=counts.get("study_pf_late", 0),
                **check,
            )
        )
        if p not in ("shape", "local", "ordered", "sparse1", "sparse4"):
            continue
        hits = accepted(paths[p])
        rmap = paired[p]
        shape_timely = {k for k, j in rmap.items() if z["kind"][j] == 1} & eligible
        for ref in (
            (
                "stride",
                "multi",
                "spp",
                "look2",
                "look4",
                "temporal",
                "sparse1",
                "sparse4",
            )
            if scope == "diagnostic"
            else ("stride", "multi", "spp")
        ):
            if ref == p:
                continue
            rz = prepared[ref]
            ri = paired[ref]
            other = {k for k, j in ri.items() if rz["kind"][j] == 1} & eligible
            gained = shape_timely - other
            lost = other - shape_timely
            counts = Counter()
            chosen_examples = {}
            for key in gained:
                a = hits.get(key)
                prior = target_kind(rz, ri.get(key))
                counts["gained_from_" + prior] += 1
                if not a or a["source"] != "frozen":
                    continue
                counts["gained_direct_frozen"] += 1
                j = rmap[key]
                prev = ri.get(key)
                counts["gained_direct_positive_nopf_head"] += b["head"][bmap[key]] > 0
                ahead = int(z["t"][j]) - a["gen"]
                reflead = (
                    (int(rz["t"][prev]) - int(rz["gen"][prev]))
                    if prev is not None and rz["gen"][prev] >= 0
                    else None
                )
                if prior == "late" and reflead is not None:
                    counts["gained_direct_late_previously"] += 1
                    counts["gained_direct_offered_earlier_than_ref"] += ahead > reflead
                if scope == "diagnostic" and a["fill"] < 0:
                    raise ValueError("missing timely complete lifecycle")
                cat = (p, ref, prior)
                if (
                    cat in chosen_examples
                    and chosen_examples[cat]["target_instr"] <= key[0]
                ):
                    continue
                cue_index = rmap.get((a["cue"], a["op"]))
                sequence = (
                    slice(max(0, cue_index - 3), cue_index + 1)
                    if cue_index is not None
                    else slice(0, 0)
                )
                chosen_examples[cat] = dict(
                    case=case,
                    scope=scope,
                    actual_arm=p,
                    prior_arm=ref,
                    prior_status=prior,
                    pc=int(z["pc"][j]),
                    target_instr=key[0],
                    target_op=key[1],
                    target_line=int(z["pa"][j]),
                    cue_instr=a["cue"],
                    cue_op=a["op"],
                    cue_line=a["line"],
                    observed_pcs=" ".join(str(int(x)) for x in z["pc"][sequence]),
                    observed_lines=" ".join(str(int(x)) for x in z["pa"][sequence]),
                    generation=a["gen"],
                    dispatch=a["dispatch"],
                    fill=a["fill"],
                    demand_ingress=int(z["t"][j]),
                    lookup=int(z["tag"][j]),
                    lead=ahead,
                    prior_lead=reflead,
                    nopf_head=int(b["head"][bmap[key]]),
                )
            groups.append(
                dict(
                    case=case,
                    scope=scope,
                    load_coverage=sample,
                    actual_arm=p,
                    reference=ref,
                    ref_timely=len(other),
                    actual_timely=len(shape_timely),
                    kept=len(shape_timely & other),
                    gained=len(gained),
                    lost=len(lost),
                    **counts,
                )
            )
            witnesses.extend(chosen_examples.values())
            if scope == "full":
                row = compressed_join(paths, ref)
                if row["instructions_joined"] != n:
                    raise ValueError("outcome count mismatch")
                complete.append(dict(case=case, actual_arm=p, reference=ref, **row))
    if scope == "diagnostic":
        assert all(coredata["shadow"][k] == v for k, v in coredata["stride"].items())
        assert np.array_equal(prepared["stride"]["kind"], prepared["shadow"]["kind"])
    return perf, groups, witnesses, complete


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--scope", required=True, choices=("diagnostic", "full"))
    a.add_argument("--cases", nargs="+", default=CASES)
    args = a.parse_args()
    if args.scope == "full" and args.cases == CASES:
        args.cases = (*CASES, "pr_heldout_entire")
    result = [[], [], [], []]
    fit = load()
    for case in args.cases:
        rr = analyze(case, args.scope, fit)
        for store, rows in zip(result, rr):
            store.extend(rows)
        print(case, len(rr[0]), flush=True)
    for name, rows in zip(
        ("performance", "first_uses", "native_examples", "complete_load_alignment"),
        result,
    ):
        if name != "complete_load_alignment" or args.scope == "full":
            publish(args.scope + "_" + name, rows)
