"""Collect aggregate replay outcomes and a traceable example for each intervention."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from compare import compare, table, event_file


def write(path, rows):
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def explain(root, case):
    baseline = root / "runs" / (case + "-stride")
    replay = root / "intervention" / (case + "-stride")
    diag = root / "diagnostics" / case
    totals = compare(baseline, replay, diag)
    targets = {}
    for r in table(diag / "targets.csv"):
        key = tuple(int(r[x]) for x in ("cpu", "target_instr", "target_operand"))
        targets[key] = r
    was_late = set()
    with event_file(baseline, "candidates") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] == "late":
                was_late.add(tuple(int(row[x]) for x in ("demand_cpu","demand_instr_id","demand_operand")))
    totals["original_stride_late_on_target"] = len(was_late & set(targets))
    counts = Counter(key[0] for key in targets)
    totals.update(targets_core0=counts[0], targets_core1=counts[1])
    ids, chosen = {}, {}
    with event_file(replay, "candidates") as stream:
        for row in csv.DictReader(stream):
            num = row["id"]
            if row["reason"] == "intervention" and row["stage"] == "proposed":
                ids[num] = {}
            if num in ids:
                ids[num][row["stage"]] = int(row["cycle"])
                if row["stage"] in ("timely","late"):
                    key = tuple(int(row[x]) for x in ("demand_cpu","demand_instr_id","demand_operand"))
                    if key in targets:
                        chosen.setdefault(row["stage"], (key,num))
    examples = []
    for stage, (key,num) in chosen.items():
        r = targets[key]
        examples.append(dict(case=case, outcome=stage, cpu=r["cpu"], anchor_instr=r["anchor_instr"],
              target_instr=r["target_instr"], target_operand=r["target_operand"], virtual_line=r["virtual_line"],
              delta=r["delta"], ordinary_stride_target_was_late=int(key in was_late), original_head_wait=r["head_wait_cycles"],
              baseline_trigger_cycle=r["stride_trigger_cycle"],baseline_target_ingress_cycle=r["stride_target_cycle"],
              replay_accepted_cycle=ids[num]["accepted"],replay_dispatch_cycle=ids[num].get("dispatch"),
              replay_fill_cycle=ids[num].get("fill"),replay_demand_lookup_cycle=ids[num][stage]))
    return dict(case=case, **totals), examples


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("cases", nargs="+")
    args=parser.parse_args()
    summaries, examples=[], []
    for name in args.cases:
        result, cases = explain(args.raw, name)
        summaries.append(result)
        examples.extend(cases)
    write(args.results / "earlier_issue.csv", summaries)
    if examples:
        write(args.results / "earlier_issue_examples.csv", examples)
