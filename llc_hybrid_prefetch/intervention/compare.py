"""Summarize what earlier issue changed on the intended loads and either core."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
from earlier_issue import event_file, read_llc


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def totals(folder):
    cores = {int(r["cpu"]): int(r["cycles"]) for r in table(folder / "cores.csv")}
    counters = {r["key"]: int(r["value"]) for r in table(folder / "stats.csv")}
    traffic = sum(counters["dram_" + kind + "_bytes"] for kind in ("demand", "prefetch", "write"))
    return cores, counters, traffic


def compare(baseline, replay, diagnostic):
    targets = table(diagnostic / "targets.csv")
    expected = {}
    for r in targets:
        c = int(r["cpu"])
        key = (c, int(r["anchor_instr"]), int(r["anchor_operand"]))
        expected[key] = (c, int(r["target_instr"]), int(r["target_operand"]))
    selected = set(expected.values())
    _, hits, _ = read_llc(replay)
    intervention_ids, accepted, states, intended = {}, set(), Counter(), Counter()
    with event_file(replay, "candidates") as f:
        for row in csv.DictReader(f):
            stage = row["stage"]
            i = int(row["id"])
            if stage == "proposed" and row["reason"] == "intervention":
                key = tuple(int(row[field]) for field in ("cpu", "instr_id", "operand"))
                intervention_ids[i] = expected[key]
            if i not in intervention_ids:
                continue
            if stage == "accepted":
                accepted.add(i)
            if stage in ("timely", "late"):
                d = tuple(int(row[field]) for field in ("demand_cpu", "demand_instr_id", "demand_operand"))
                if d == intervention_ids[i]:
                    intended[stage] += 1
            if stage in ("rejected", "late", "timely", "redundant", "evicted", "unresolved"):
                states[stage] += 1
    cb, _, tb = totals(baseline)
    cr, sr, tr = totals(replay)
    if set(cb) != set(cr):
        raise ValueError("different active cores")
    sel = json.loads((diagnostic / "selection.json").read_text())
    result = dict(planned=len(selected),matched_anchors=sr["plan_matched"],accepted_overrides=len(accepted),
                  all_timely_overrides=states["timely"],all_late_overrides=states["late"],
                  timely_on_intended_demand=intended["timely"],late_on_intended_demand=intended["late"],
                  targets_llc_hits=sum(hits.get(k) == 1 for k in selected),
                  targets_llc_misses=sum(hits.get(k) == 0 for k in selected),
                  targets_no_completed_lookup=sum(k not in hits for k in selected),
                  stride_cycles=max(cb.values()),replay_cycles=max(cr.values()),cycles_saved=max(cb.values())-max(cr.values()),
                  stride_core0=cb[0],replay_core0=cr[0],stride_core1=cb.get(1),replay_core1=cr.get(1),
                  stride_dram_bytes=tb,replay_dram_bytes=tr,median_service=sel["median_service"],
                  stride_candidates_in_lead_window=sel["eligible_targets"])
    if len(intervention_ids) != sr["plan_matched"] or len(intervention_ids) > len(expected):
        raise ValueError("intervention triggers do not reconcile")
    if sum(states.values()) != len(intervention_ids):
        raise ValueError("intervention lifecycle does not reconcile")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("baseline", "replay", "diagnostic"):
        p.add_argument(a, type=Path)
    args = p.parse_args()
    print(json.dumps(compare(args.baseline, args.replay, args.diagnostic), indent=2))
