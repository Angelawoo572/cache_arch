"""Attribute live requests to individual loads and compare same-work replays."""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import median

from run import HERE, RAW, MODES, pairs, read_case, source


def rows(folder, name):
    with gzip.open(folder / (name + ".csv.gz"), "rt") as stream:
        yield from csv.DictReader(stream)


def counters(folder):
    with (folder / "stats.csv").open() as stream:
        return {s["key"]: int(s["value"]) for s in csv.DictReader(stream)}


def cycles(folder):
    with (folder / "cores.csv").open() as stream:
        return [int(s["cycles"]) for s in csv.DictReader(stream)]


def write(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summarize():
    summaries = []
    examples = []
    details = []
    for train, test in pairs():
        name = test if train == test else train + "_to_" + test
        z, meta, cuts, gate, keys = read_case(test)
        index = {
            (int(a), int(b)): i for i, (a, b) in enumerate(zip(z["instr"], z["op"]))
        }
        baseline = source(test)
        for mode in ("noop", *MODES):
            folder = RAW / "cue_runs" / name / mode
            if not (folder / "complete.json").is_file():
                raise ValueError("unfinished " + str(folder))
            c, old = cycles(folder), cycles(baseline)
            stats = counters(folder)
            if mode == "noop" and (stats != counters(baseline) or old != c):
                raise ValueError("no-issue replay mismatch " + name)
            with (folder / "cue.csv").open() as stream:
                decisions = list(csv.DictReader(stream))
            if mode in MODES:
                # No decision can alter the machine before the first override.
                for r in decisions:
                    i = index[(int(r["instr"]), int(r["operand"]))]
                    if int(r["key"]) != keys[i][MODES.index(mode)] or int(
                        r["cycle"]
                    ) != int(z["t"][i]):
                        raise ValueError("native/offline prefix mismatch")
                    if r["action"] == "override":
                        break
            acts = Counter(r["action"] for r in decisions)
            taken = {
                (int(r["instr"]), int(r["operand"])): r
                for r in decisions
                if r["action"] == "override"
            }
            ids = {}
            stages = Counter()
            before = []
            credited = 0
            originally_late = 0
            confirmed = 0
            for r in rows(folder, "candidates"):
                pid = int(r["id"])
                stage = r["stage"]
                if stage == "proposed" and r["reason"] == "intervention":
                    a = (int(r["instr_id"]), int(r["operand"]))
                    ids[pid] = dict(
                        a=a,
                        gen=int(r["cycle"]),
                        target=int(r["target"]),
                        dispatch=-1,
                        fill=-1,
                    )
                if pid not in ids:
                    continue
                item = ids[pid]
                stages[stage] += 1
                if stage == "rejected":
                    stages["reject_" + r["reason"]] += 1
                if stage in ("dispatch", "fill"):
                    item[stage] = int(r["cycle"])
                if stage in ("timely", "late"):
                    item.update(
                        claim=stage,
                        demand=int(r["cycle"]),
                        dcpu=int(r["demand_cpu"]),
                        d=(int(r["demand_instr_id"]), int(r["demand_operand"])),
                    )
            if len(ids) != acts["override"]:
                raise ValueError("override accounting")
            for item in ids.values():
                if (
                    item.get("claim") != "timely"
                    or item["dcpu"] != 0
                    or item["d"] not in index
                ):
                    continue
                j = index[item["d"]]
                a = item["a"]
                i = index.get(a)
                if z["pa"][j] != item["target"] or not (
                    z["baseline"][j] == 0 and z["hit"][j] == 0
                ):
                    continue
                credited += 1
                if z["pf_kind"][j] != 2:
                    continue
                originally_late += 1
                if (
                    i is None
                    or int(taken[a]["physical_base"]) != z["pa"][i]
                    or z["pf_gen"][j] < 0
                ):
                    continue
                earlier = int(z["pf_gen"][j] - z["t"][i])
                if earlier <= 0:
                    continue
                confirmed += 1
                before.append(earlier)
                if (
                    len([e for e in examples if (e["case"], e["mode"]) == (name, mode)])
                    >= 3
                ):
                    continue
                examples.append(
                    dict(
                        case=name,
                        mode=mode,
                        anchor_instr=a[0],
                        anchor_operand=a[1],
                        anchor_pc=int(z["pc"][i]),
                        anchor_pa=int(z["pa"][i]),
                        target_instr=item["d"][0],
                        target_operand=item["d"][1],
                        target_pc=int(z["pc"][j]),
                        target_pa=int(z["pa"][j]),
                        anchor_on_stride_clock=int(z["t"][i]),
                        native_stride_proposal=int(z["pf_gen"][j]),
                        native_stride_ingress=int(z["t"][j]),
                        native_stride_fill=int(z["pf_fill"][j]),
                        earlier_by_cycles=earlier,
                        ingress_event_distance=j - i,
                        cue_age_events=int(taken[a]["cue_age"]),
                        replay_proposal=item["gen"],
                        replay_dispatch=item["dispatch"],
                        replay_fill=item["fill"],
                        replay_lookup=item["demand"],
                    )
                )
            dram = sum(
                stats[k]
                for k in (
                    "dram_demand_bytes",
                    "dram_prefetch_bytes",
                    "dram_write_bytes",
                )
            )
            summaries.append(
                dict(
                    case=name,
                    memory=mode,
                    table_matches=len(decisions),
                    agreements=acts["agree"],
                    overrides=len(ids),
                    accepted=stages["accepted"],
                    timely=stages["timely"],
                    late=stages["late"],
                    evicted_unused=stages["evicted"],
                    direct_timely_on_paired_misses=credited,
                    original_late_now_timely=originally_late,
                    confirmed_earlier_anchor=confirmed,
                    median_cycles_before_native=(median(before) if before else ""),
                    core0_cycles=c[0],
                    peer_cycles=(c[1] if len(c) > 1 else ""),
                    stride_makespan=max(old),
                    makespan=max(c),
                    cycles_saved=max(old) - max(c),
                    dram_bytes=dram,
                    head_union_core0=stats["head_unique_cycles_core_0"],
                )
            )
            for k, v in sorted(stages.items()):
                details.append(dict(case=name, memory=mode, stage=k, count=v))
    write("replay_results.csv", summaries)
    write("event_examples.csv", examples)
    write("lifecycle.csv", details)


if __name__ == "__main__":
    summarize()
