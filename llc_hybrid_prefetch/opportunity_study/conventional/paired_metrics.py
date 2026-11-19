"""Summarize fixed-work demand tag lookups, prefetch lifecycles and matched coverage for new workloads."""

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
TERMINAL = {"timely", "late", "evicted", "redundant", "unresolved"}


def records(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def frac(numerator, denominator):
    return round(numerator / denominator, 8) if denominator else ""


def consume(folder):
    cores = records(folder / "cores.csv")
    events = [Counter() for _ in cores]
    tags = [{} for _ in cores]
    used = [dict(timely=set(), late=set()) for _ in cores]
    accepted, finished = {}, {}
    with gzip.open(folder / "loads.csv.gz", "rb") as source:
        for line in source:
            if b",lookup," not in line and b",LLC,ingress," not in line:
                continue
            cpu, iid, op, pc, virtual, physical, level, stage, _, hit = line.split(b",", 10)[:10]
            if level not in (b"LLC", b"L2"):
                continue
            cpu = int(cpu)
            if stage == b"ingress":
                events[cpu]["ingress"] += 1
                continue
            outcome = int(hit)
            assert outcome in (0, 1)
            events[cpu][level.decode(), outcome] += 1
            if level == b"LLC":
                key = (int(iid), int(op))
                assert key not in tags[cpu]
                tags[cpu][key] = (outcome, int(pc), int(virtual), int(physical))
    with gzip.open(folder / "candidates.csv.gz", "rt") as source:
        for r in csv.DictReader(source):
            stage, cpu, cid = r["stage"], int(r["cpu"]), int(r["id"])
            events[cpu][stage] += 1
            if stage == "accepted":
                assert cid not in accepted
                accepted[cid] = cpu
            if stage in TERMINAL:
                assert cid in accepted and accepted[cid] == cpu and cid not in finished
                finished[cid] = stage
            if stage in ("timely", "late"):
                target = used[int(r["demand_cpu"])][stage]
                key = int(r["demand_instr_id"]), int(r["demand_operand"])
                assert key not in target
                target.add(key)
    assert accepted.keys() == finished.keys(), folder
    counts = {r["key"]: int(r["value"]) for r in records(folder / "stats.csv")}
    assert len(accepted) == counts["prefetch_accepted"]
    return cores, events, tags, used


def summarize(spec):
    case = spec["case"]
    arms = {p: consume(STUDY / "raw/runs" / (case + "-" + p)) for p in ("nopf", "stride")}
    output = []
    for policy, (cores, all_events, all_tags, all_used) in arms.items():
        for c, (core, counts, tags, used) in enumerate(zip(cores, all_events, all_tags, all_used)):
            base = arms["nopf"][2][c]
            assert all(v[1:] == base[k][1:] for k, v in tags.items() if k in base)
            base_miss = {k for k, v in base.items() if v[0] == 0}
            hits = {k for k, v in tags.items() if v[0] == 1}
            assert not (used["late"] & used["timely"])
            timely_cover = len(base_miss & used["timely"] & hits)
            late_cover = len(base_miss & used["late"])
            paired_hits = len(base_miss & hits)
            accepted, timely, late = (counts[k] for k in ("accepted", "timely", "late"))
            instructions, cycles = int(core["instructions"]), int(core["cycles"])
            row = dict(
                case=case,
                core=c,
                policy=policy,
                instructions=instructions,
                cycles=cycles,
                ipc=frac(instructions, cycles),
            )
            for level in ("L2", "LLC"):
                nhit, miss = counts[level, 1], counts[level, 0]
                lookups = nhit + miss
                prefix = level.lower() + "_demand_"
                row.update(
                    {
                        prefix + name: value
                        for name, value in (
                            ("lookups", lookups),
                            ("hits", nhit),
                            ("misses", miss),
                            ("hit_rate", frac(nhit, lookups)),
                            ("miss_rate", frac(miss, lookups)),
                            ("mpki", frac(1000 * miss, instructions)),
                        )
                    }
                )
            row.update(
                llc_ingress=counts["ingress"],
                pf_proposed=counts["proposed"],
                pf_accepted=accepted,
                pf_dispatched=counts["dispatch"],
                pf_filled=counts["fill"],
                pf_first_timely=timely,
                pf_first_late=late,
                pf_claimed=timely + late,
                pf_evicted_unused=counts["evicted"],
                pf_redundant=counts["redundant"],
                pf_unresolved=counts["unresolved"],
                accepted_per_ingress=frac(accepted, counts["ingress"]),
                acceptance_rate=frac(accepted, counts["proposed"]),
                accepted_accuracy=frac(timely + late, accepted),
                accepted_timely_accuracy=frac(timely, accepted),
                timely_fraction_of_claimed=frac(timely, timely + late),
                nopf_demand_llc_misses=len(base_miss),
                matched_miss_to_hit=paired_hits,
                attributable_timely_on_nopf_miss=timely_cover,
                attributable_late_on_nopf_miss=late_cover,
                paired_coverage=frac(paired_hits, len(base_miss)),
                direct_timely_coverage=frac(timely_cover, len(base_miss)),
                direct_inclusive_coverage=frac(timely_cover + late_cover, len(base_miss)),
            )
            output.append(row)
    return output


def claimed_bfs_timing(case):
    """Match transactions that have the same original producer, first consumer and physical line."""
    folder = STUDY / "raw/runs" / (case + "-stride")
    transactions = {}
    claims = {}
    with gzip.open(folder / "candidates.csv.gz", "rt") as source:
        for r in csv.DictReader(source):
            if r["cpu"] != "0":
                continue
            cid, stage, t = int(r["id"]), r["stage"], int(r["cycle"])
            if stage == "proposed":
                transactions[cid] = dict(
                    anchor=(int(r["instr_id"]), int(r["operand"])), line=int(r["target"]), proposed=t
                )
            if cid not in transactions:
                continue
            v = transactions[cid]
            if stage in ("dispatch", "fill", "timely", "late"):
                v[stage] = t
            if stage in ("timely", "late") and r["demand_cpu"] == "0":
                demand = (int(r["demand_instr_id"]), int(r["demand_operand"]))
                identity = v["anchor"], demand, v["line"]
                assert identity not in claims
                claims[identity] = v
    arrivals = {}
    with gzip.open(folder / "loads.csv.gz", "rb") as source:
        for line in source:
            if line.startswith(b"0,") and b",LLC,ingress," in line:
                r = line.split(b",")
                arrivals[int(r[1]), int(r[2])] = int(r[8])
    for (_, key, _), events in claims.items():
        events["demand"] = arrivals.get(key)
    return claims


def service_report():
    solo, mixed = map(claimed_bfs_timing, ("gap_bfs8", "gap_bfs8_lbm"))
    common = solo.keys() & mixed.keys()
    results = []
    for label, group in (
        ("all_matched_claims", common),
        ("timely_to_late", [k for k in common if "timely" in solo[k] and "late" in mixed[k]]),
    ):
        complete = [
            k
            for k in group
            if all(
                arm[k].get(event) is not None
                for arm in (solo, mixed)
                for event in ("proposed", "dispatch", "fill", "demand")
            )
        ]
        for name, arm in (("solo", solo), ("with_lbm", mixed)):
            row = dict(
                workload="gap_bfs8",
                case=name,
                cohort=label,
                matched_requests=len(group),
                complete_lifecycles=len(complete),
            )
            for earlier, later in (
                ("proposed", "demand"),
                ("proposed", "dispatch"),
                ("dispatch", "fill"),
                ("fill", "demand"),
            ):
                row[f"median_{earlier}_to_{later}"] = statistics.median(
                    arm[k][later] - arm[k][earlier] for k in complete
                )
            results.append(row)
    with (HERE / "bfs_service.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    samples = json.loads((STUDY / "workloads/cases.json").read_text())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        output = [r for block in pool.map(summarize, samples) for r in block]
    reference = {
        (r["case"], int(r["core"])): r
        for r in records(STUDY / "workloads/outcomes.csv")
        if r["slice"] == "full" and r["label"] == "all"
    }
    for row in output:
        r = reference[row["case"], row["core"]]
        p = "base" if row["policy"] == "nopf" else "stride"
        assert row["llc_ingress"] == int(r[p + "_ingress"]) and row["llc_demand_misses"] == int(
            r[p + "_llc_miss"]
        )
        if p == "stride":
            assert row["attributable_timely_on_nopf_miss"] == int(r["miss_hit_firstuse"]) and row[
                "matched_miss_to_hit"
            ] == int(r["miss_hit"])
    with (HERE / "paired_metrics.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    service_report()
    print(f"Validated {len(samples)} paired workloads and {len(output)} fixed-core policy measurements.")
