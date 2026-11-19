"""Re-run the two stronger conventional controls and audit four fixed BFS-3 loads."""

import bisect
import csv
import gzip
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from analyze import save
from run import RAW, STUDY, comparison_cases, csvs, run, totals

CASE = "gap_bfs3_late"
IDS = (51681598, 51766261, 51839379, 51967364)


def data(folder, stem):
    with gzip.open(folder / (stem + ".csv.gz"), "rt") as source:
        yield from csv.DictReader(source)


def audit(policy):
    folder = run(comparison_cases()[CASE], policy, "four_targets", detail="1")
    baseline = RAW / "runs" / (CASE + "__" + policy + "__open")
    measured = csvs(folder / "cores.csv")
    assert measured == csvs(baseline / "cores.csv") and totals(folder) == totals(baseline)
    stats = totals(folder)
    specification = {int(r["target_instr"]): r for r in csvs(STUDY / "history/page_exclusive_loads.csv")}
    assert set(IDS) == specification.keys()
    lines = {int(t["target_line"]) for t in specification.values()}
    seen = {}
    strides = defaultdict(list)
    head = {}
    for r in data(folder, "loads"):
        if r["level"] != "LLC" or r["cpu"] != "0" or int(r["physical_line"]) not in lines:
            continue
        i = int(r["instr_id"])
        line = int(r["physical_line"])
        clock = int(r["cycle"])
        if r["stage"] == "ingress":
            strides[line].append(clock)
        if i not in specification:
            continue
        ref = specification[i]
        assert (int(r["operand"]), int(r["pc"]), int(r["virtual_line"]), line) == tuple(
            int(ref[k]) for k in ("target_operand", "target_pc", "target_virtual_line", "target_line")
        )
        if r["stage"] in ("lookup", "ingress"):
            row = seen.setdefault(i, {})
            assert r["stage"] not in row
            row[r["stage"]] = clock
            if r["stage"] == "lookup":
                row["hit"] = int(r["hit"])
    for r in data(folder, "head"):
        idx = int(r["instr_id"])
        if (
            r["cpu"] == "0"
            and idx in specification
            and r["operand"] == specification[idx]["target_operand"]
        ):
            head[idx] = int(r["cycles"])
    transactions = defaultdict(dict)
    direct = {}
    for r in data(folder, "candidates"):
        if int(r["target"]) not in lines:
            continue
        i = int(r["id"])
        stage = r["stage"]
        t = int(r["cycle"])
        tx = transactions[i]
        tx[stage] = t
        if stage == "proposed":
            tx.update(
                i=i,
                anchor=int(r["instr_id"]),
                operand=int(r["operand"]),
                anchor_pc=int(r["pc"]),
                base=int(r["base"]),
                target=int(r["target"]),
                source=r["reason"],
            )
        if stage == "rejected":
            tx["rejection"] = r["reason"]
        if stage in ("timely", "late") and r["demand_cpu"] == "0":
            idx = int(r["demand_instr_id"])
            if idx in specification and r["demand_operand"] == specification[idx]["target_operand"]:
                assert idx not in direct and int(r["target"]) == int(specification[idx]["target_line"])
                assert int(r["id"]) in transactions
                direct[idx] = (tx, stage)
    results = []
    for idx in IDS:
        ref = specification[idx]
        d = seen[idx]
        line = int(ref["target_line"])
        seq = sorted(strides[line])
        previous = bisect.bisect_left(seq, d["ingress"]) - 1
        lower = seq[previous] if previous >= 0 else -1
        recent = [
            x for x in transactions.values() if x["target"] == line and lower < x["proposed"] < d["ingress"]
        ]
        source, stage = direct.get(idx, ({}, "none"))
        if source:
            assert source["accepted"] <= source[stage] and (stage != "timely" or d["hit"] == 1)
        selected = sorted(recent, key=lambda x: (x["proposed"], x["i"]))[-1] if recent else {}
        results.append(
            dict(
                case=CASE,
                policy=policy,
                cpu=0,
                target_instr=idx,
                target_operand=ref["target_operand"],
                target_pc=ref["target_pc"],
                target_virtual_line=ref["target_virtual_line"],
                target_physical_line=line,
                frozen_cue_instr=ref["cue_instr"],
                demand_ingress=d["ingress"],
                demand_lookup=d["lookup"],
                llc_outcome="hit" if d["hit"] else "miss",
                head_wait=head.get(idx, 0),
                first_use=stage,
                source=source.get("source", ""),
                trigger_instr=source.get("anchor", ""),
                trigger_operand=source.get("operand", ""),
                trigger_pc=source.get("anchor_pc", ""),
                trigger_line=source.get("base", ""),
                proposal=source.get("proposed", ""),
                accepted=source.get("accepted", ""),
                dispatch=source.get("dispatch", ""),
                fill=source.get("fill", ""),
                first_use_cycle=source.get(stage, ""),
                same_as_frozen_cue=int(source["anchor"] == int(ref["cue_instr"])) if source else "",
                earlier_proposals_since_last_llc_ingress=len(recent),
                last_prior_anchor=selected.get("anchor", ""),
                last_prior_rejection=selected.get("rejection", ""),
                open_arm_cycles=measured[0]["cycles"],
                open_and_replay_dram_bytes=sum(
                    stats["dram_" + k + "_bytes"] for k in ("demand", "prefetch", "write")
                ),
                all_core_stats_equal=1,
            )
        )
    return results


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(audit, ("spp", "multi")))
    save("bfs_strong_targets.csv", [r for group in groups for r in group])
