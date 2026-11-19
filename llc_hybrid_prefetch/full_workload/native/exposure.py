"""Compare retirement-head wait distributions for matched original misses within the fixed detail windows."""

import argparse
from pathlib import Path
from collections import defaultdict
from run import RAW, HERE, FULL, rows
from timing import gzrows, sorted_quantile
from summarize import FAR, POLICIES, write


def key(r):
    return int(r["instr_id"]), int(r["operand"])


def head_waits(folder):
    return {key(r): int(r["cycles"]) for r in gzrows(folder / "head.csv.gz")}


def original_misses(folder):
    return {
        key(r): int(r["physical_line"])
        for r in gzrows(folder / "loads.csv.gz")
        if r["stage"] == "lookup" and r["level"] == "LLC" and r["hit"] == "0"
    }


def classify(folder, targets):
    states = {}
    for r in gzrows(folder / "loads.csv.gz"):
        if r["level"] != "LLC" or r["stage"] != "lookup":
            continue
        k = key(r)
        if k in targets:
            if int(r["physical_line"]) != targets[k]:
                raise ValueError("Mismatch in physical identity of aligned operand")
            states[k] = "ordinary_hit" if r["hit"] == "1" else "persistent_miss"
    for r in gzrows(folder / "candidates.csv.gz"):
        if r["stage"] not in ("timely", "late"):
            continue
        k = (int(r["demand_instr_id"]), int(r["demand_operand"]))
        if k in targets:
            if int(r["target"]) != targets[k]:
                raise ValueError("Mismatch in physical identity of first-use claim")
            states[k] = r["stage"]
    return states


def main(out, allow_partial, cases):
    data = []
    roster = cases or [r["case"] for r in rows(FULL / "intake/selection.csv")] + list(
        FAR
    )
    for case in roster:
        base = RAW / "runs" / (case + "__nopf")
        if not (base / "complete.json").exists() and allow_partial:
            continue
        targets = original_misses(base)
        baseline = head_waits(base)
        for policy in POLICIES[1:]:
            arm = RAW / "runs" / (case + "__" + policy)
            if not (arm / "complete.json").exists() and allow_partial:
                continue
            after = head_waits(arm)
            states = classify(arm, targets)
            per_class = defaultdict(list)
            for k in targets:
                group = states.get(k, "no_arm_tag")
                per_class[group].append((baseline.get(k, 0), after.get(k, 0)))
            for group, pairs in sorted(per_class.items()):
                before, after_times = zip(*pairs)
                data.append(
                    dict(
                        case=case,
                        policy=policy,
                        cohort=group,
                        original_sampled_misses=len(pairs),
                        nopf_positive_head_wait=sum(v > 0 for v in before),
                        arm_positive_head_wait=sum(v > 0 for v in after_times),
                        nopf_head_wait_p50=sorted_quantile(before, 0.5),
                        nopf_head_wait_p90=sorted_quantile(before, 0.9),
                        arm_head_wait_p50=sorted_quantile(after_times, 0.5),
                        arm_head_wait_p90=sorted_quantile(after_times, 0.9),
                    )
                )
    write(out / "sampled_exposure.csv", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    main(args.out, args.completed_only, args.cases)
