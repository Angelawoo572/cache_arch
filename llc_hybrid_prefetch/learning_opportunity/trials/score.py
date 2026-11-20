"""Join diagnostic targets, lifecycle timing and the displaced outcomes across each complete replay."""

from itertools import product
from score_helpers import turnover
from select_targets import CASES, HERE, RAW, RUNS, dump, key, read


def events(path, anchors, targets):
    stages = {}
    for r in read(path / "candidates.csv.gz"):
        ident = key(r)
        if ident in anchors:
            stages.setdefault(ident, {})[r["stage"]] = r
    demands = {
        key(r): r
        for r in read(path / "loads.csv.gz")
        if r["level"] == "LLC" and r["stage"] == "ingress" and key(r) in targets
    }
    waits = {}
    for r in read(path / "head.csv.gz"):
        k = key(r)
        if k in targets:
            waits[k] = waits.get(k, 0) + int(r["cycles"])
    return stages, demands, waits


def analyze(case, mode):
    rows = [r for r in read(HERE / "targets.csv") if r["case"] == case]
    path = RAW / (case + "__" + mode)
    ref = RUNS / (case + "__spp")
    targets = {(int(r["target_instr"]), int(r["target_operand"])) for r in rows}
    anchors = {(int(r[mode + "_instr"]), int(r[mode + "_operand"])) for r in rows}
    gains, losses, status = turnover(case, path, targets, "spp")
    stages, demands, waits = events(path, anchors, targets)
    original, _, prior_waits = events(ref, anchors, targets)
    _, _, no_waits = events(RUNS / (case + "__nopf"), set(), targets)
    result = []
    for row in rows:
        target = int(row["target_instr"]), int(row["target_operand"])
        anchor = int(row[mode + "_instr"]), int(row[mode + "_operand"])
        targetline = int(row["target_line"])
        oldmask = status[target][1]
        assert (
            status[target][0] == 2
            and oldmask == {"plain": 2, "late": 6}[row["original_spp"]]
        )
        actual = stages.get(anchor, {})
        accepted = actual.get("accepted", {})
        proposed = actual.get("proposed", {})
        consumer = actual.get("timely") or actual.get("late") or {}
        demand = demands.get(target, {})
        physical_match = bool(proposed and int(proposed["target"]) == targetline)
        if demand and int(demand["physical_line"]) != targetline:
            raise ValueError("Matched demand changed physical line")
        end = actual.get("fill", {}).get("cycle", "")
        cycle = demand.get("cycle", "")
        margin = int(cycle) - int(end) if cycle and end and physical_match else ""
        original_offer = original.get(anchor, {}).get("proposed", {}).get("target", "")
        result.append(
            dict(
                case=case,
                timing=mode,
                quarter=row["quarter"],
                target_instr=target[0],
                operand=target[1],
                target_line=targetline,
                baseline_spp=row["original_spp"],
                baseline_head_wait=prior_waits.get(target, 0),
                nopf_head_wait=no_waits.get(target, 0),
                new_head_wait=waits.get(target, 0),
                native_target=original_offer,
                candidate_matches_target=int(physical_match),
                accepted=int(bool(accepted) and physical_match),
                reject_reason=actual.get("rejected", {}).get("reason", ""),
                generated=proposed.get("cycle", ""),
                issued=accepted.get("cycle", ""),
                dispatch=actual.get("dispatch", {}).get("cycle", ""),
                fill=end,
                demand=cycle,
                fill_margin=margin,
                attributed_here=int(
                    physical_match
                    and consumer.get("demand_instr_id") == str(target[0])
                    and consumer.get("demand_operand") == str(target[1])
                    and "timely" in actual
                ),
                mask=status[target][2],
            )
        )

    def stats(d):
        return {r["key"]: int(r["value"]) for r in read(d / "stats.csv")}

    a, b = stats(ref), stats(path)

    def cycles(d):
        return int(next(read(d / "cores.csv"))["cycles"])

    total = sum(int(r["mask"]) == 9 for r in result)
    return dict(
        case=case,
        timing=mode,
        planned=len(rows),
        matched=b.get("plan_matched", 0),
        accepted_exact=sum(r["accepted"] for r in result),
        target_timely=total,
        timely_from_planned_offer=sum(r["attributed_here"] for r in result),
        original_timely_gained=gains,
        original_timely_lost=losses,
        cycle_savings=cycles(ref) - cycles(path),
        baseline_cycles=cycles(ref),
        treated_cycles=cycles(path),
        change_head_union_cycles=b["head_unique_cycles_core_0"]
        - a["head_unique_cycles_core_0"],
        dram_byte_change=sum(
            b[k] - a[k]
            for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes")
        ),
    ), result


if __name__ == "__main__":
    all_results = [analyze(*job) for job in product(CASES, ("early", "near"))]
    dump(HERE / "effects.csv", [s for s, _ in all_results])
    dump(HERE / "request_timing.csv", [r for _, group in all_results for r in group])
