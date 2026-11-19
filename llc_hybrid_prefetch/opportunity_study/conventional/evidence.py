"""Print complete lifecycle examples in which a conventional alternative covers paired baseline misses."""

import csv
from concurrent.futures import ThreadPoolExecutor

from analyze import save
from critical import rows
from run import BASE, comparison_cases, run


def lookups(d, ids):
    result = {}
    with rows(d, "loads") as f:
        for line in f:
            if ",LLC,lookup," not in line:
                continue
            record = line.strip().split(",")
            key = tuple(map(int, record[:3]))
            if key in ids:
                result[key] = tuple(map(int, (record[3], record[4], record[5], record[8], record[9])))
    return result


def collect(job):
    case, policy, source = job
    native = BASE / "raw/qualified" if case == "mcf-cactus-r1" else BASE / "opportunity_study/raw/runs"
    replay = run(comparison_cases()[case], policy, "evidence", detail="1")
    transactions = {}
    complete = []
    with rows(replay, "candidates") as f:
        for r in csv.DictReader(f):
            stage = r["stage"]
            i = r["id"]
            if stage == "ready" and r["reason"] == source:
                transactions[i] = dict(r)
            if i not in transactions:
                continue
            tx = transactions[i]
            if stage in ("ready", "accepted", "dispatch", "fill", "timely"):
                tx[stage] = int(r["cycle"])
            if stage == "timely" and all(k in tx for k in ("ready", "accepted", "dispatch", "fill")):
                tx["consumer"] = tuple(
                    int(r[k]) for k in ("demand_cpu", "demand_instr_id", "demand_operand")
                )
                complete.append(tx)
    ids = {tx["consumer"] for tx in complete}
    off = lookups(native / (case + "-nopf"), ids)
    stride = lookups(native / (case + "-stride"), ids)
    on = lookups(replay, ids)
    evidence = []
    for tx in complete:
        k = tx["consumer"]
        if not all(k in d for d in (off, stride, on)):
            continue
        a, b, c = off[k], stride[k], on[k]
        if a[4] or b[4] or c[4] != 1:
            continue
        assert a[:3] == b[:3] == c[:3] and int(tx["target"]) == c[2] and c[3] == tx["timely"]
        evidence.append(
            dict(
                case=case,
                policy=policy,
                source=source,
                trigger_cpu=tx["cpu"],
                trigger_id=tx["instr_id"],
                trigger_operand=tx["operand"],
                trigger_pc=tx["pc"],
                trigger_line=tx["base"],
                target_cpu=k[0],
                target_id=k[1],
                target_operand=k[2],
                target_pc=c[0],
                target_virtual_line=c[1],
                target_physical_line=c[2],
                ready=tx["ready"],
                accepted=tx["accepted"],
                dispatch=tx["dispatch"],
                fill=tx["fill"],
                demand_hit=tx["timely"],
                fill_slack=tx["timely"] - tx["fill"],
                nopf_lookup=a[4],
                stride_lookup=b[4],
                policy_lookup=c[4],
            )
        )
    # Save examples across the whole measurement interval in addition to its frequency.
    sample = [evidence[i] for i in sorted({0, len(evidence) // 2, len(evidence) - 1})] if evidence else []
    return dict(
        case=case,
        policy=policy,
        source=source,
        timely_dispatch_fill_demand=len(complete),
        matched_nopf_miss_stride_miss_alternative_hit=len(evidence),
    ), sample


if __name__ == "__main__":
    jobs = [
        ("mcf-cactus-r1", "multi", "previous_delta"),
        ("streams_onepc_paced", "spp", "spp"),
        ("row_period2_paced", "multi", "previous_delta"),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(collect, jobs))
    save("conventional_examples.csv", [e for _, sample in result for e in sample])
    save("conventional_examples_counts.csv", [r for r, _ in result])
