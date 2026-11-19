"""Check cross-report denominators, measured spans, and gain/loss accounting."""

import csv
from pathlib import Path

FULL = Path(__file__).resolve().parents[1]
ARMS = {"nopf", "stride", "multi", "spp", "temporal"}


def read(path):
    with (FULL / path).open() as f:
        return list(csv.DictReader(f))


def main():
    selected = {r["case"] for r in read("intake/selection.csv")}
    if len(selected) != 7:
        raise ValueError("Expected seven distinct complete recordings")
    views = selected | {"bfs_discovery_far", "mcf17_discovery_far"}
    perf = {}
    for path, cases, roi in (
        ("native/performance.csv", views, 200_000_000),
        ("native/entire/performance.csv", {"pr_heldout_entire"}, 450_000_001),
    ):
        rows = read(path)
        if len(rows) != len(cases) * len(ARMS) or {
            (r["case"], r["policy"]) for r in rows
        } != {(case, arm) for case in cases for arm in ARMS}:
            raise ValueError("Missing or duplicated measured arm in " + path)
        for row in rows:
            if (
                int(row["warmup"]) != 50_000_000
                or int(row["measured_instructions"]) != roi
            ):
                raise ValueError("Unexpected retirement boundaries")
            if sum(
                int(row[k])
                for k in (
                    "orig_to_timely",
                    "orig_to_late",
                    "orig_to_ordinary_hit",
                    "orig_to_persistent_miss",
                    "orig_without_arm_tag",
                )
            ) != int(row["nopf_original_misses"]):
                raise ValueError("Non-partitioning dynamic outcomes")
            if sum(
                int(row[k])
                for k in (
                    "timely_first_use",
                    "late_merge",
                    "redundant",
                    "unresolved",
                    "unused_evicted",
                )
            ) != int(row["accepted"]):
                raise ValueError("Accepted request outcomes do not partition")
            perf[row["case"], row["policy"]] = row
    stream_rows = read("streams/event_counts.csv")
    for case in views:
        for arm in ("stride", "multi", "spp"):
            source = perf[case, arm]
            for view in ("all", "pc", "physical_256KiB_region"):
                rows = [
                    r
                    for r in stream_rows
                    if (r["case"], r["policy"], r["view"]) == (case, arm, view)
                ]
                if len(rows) != 15 or sum(int(r["events"]) for r in rows) != int(
                    source["nopf_original_misses"]
                ):
                    raise ValueError("Wrong full-program-order episode denominator")
                if sum(
                    int(r["events"]) for r in rows if r["current"] == "timely_first_use"
                ) != int(source["orig_to_timely"]):
                    raise ValueError("Episode attribution differs from simulator")
    joins = [
        r
        for r in read("discovery/full_complete_load_alignment.csv")
        if r["actual_arm"] == "shape"
    ]
    if not {
        (c, p)
        for c in selected | {"pr_heldout_entire"}
        for p in ("stride", "spp", "multi")
    } <= {(r["case"], r["reference"]) for r in joins}:
        raise ValueError("Missing full independent-address replay")
    for row in joins:
        src = perf.get((row["case"], row["reference"]))
        if src and (
            int(row["original_misses"]) != int(src["nopf_original_misses"])
            or int(row["original_timely_kept"]) + int(row["original_timely_lost"])
            != int(src["orig_to_timely"])
        ):
            raise ValueError("Full replay disagrees with conventional first-use count")
    paired = read("interference/per_core.csv")
    if len(paired) != 8 or any(
        int(r["solo_timely_pair_late"]) > int(r["solo_only_direct"]) for r in paired
    ):
        raise ValueError("Invalid interference lifecycle counts")
    checked = read("streams/arithmetic_replay.csv")
    if len(checked) != 36:
        raise ValueError("Unfinished warm arithmetic controls")
    for r in checked:
        ref = next(
            p for p in checked if p["case"] == r["case"] and p["policy"] == "stride"
        )
        if int(r["direct_timely"]) - int(ref["direct_timely"]) != int(
            r["gained_vs_stride"]
        ) - int(r["lost_vs_stride"]):
            raise ValueError("Arithmetic replay displaced-load accounting mismatch")
    continuous = read("continuous/validation.csv")
    if {r["case"] for r in continuous} != {"bfs_discovery", "soplex06_heldout"}:
        raise ValueError("Missing continuous-ingress validation")
    for row in continuous:
        ref = perf[row["case"], "stride"]
        if (
            int(row["total_stride_ingress"]) != int(ref["llc_demand_ingress"])
            or row["identical_counters_and_per_operand_masks"] != "True"
            or row["identical_ordered_detail_records"] != "True"
        ):
            raise ValueError("Continuous instrumentation changes baseline measurements")
    attribution = read("streams/mcf_transfer/target_outcomes.csv")
    details = read("streams/mcf_transfer/target_details.csv")
    mcf_cases = {"mcf17_discovery", "mcf17_discovery_far", "mcf17_transfer"}
    if len(attribution) != 6 or {(r["case"], r["arm"]) for r in attribution} != {
        (c, a) for c in mcf_cases for a in ("local", "shared_increment")
    }:
        raise ValueError("Missing full-scale MCF region or arithmetic arm")
    for row in attribution:
        members = [
            d for d in details if (d["case"], d["arm"]) == (row["case"], row["arm"])
        ]
        ids = {(d["target_instr"], d["target_op"]) for d in members}
        if len(ids) != len(members) or len(ids) != int(row["original_sampled_targets"]):
            raise ValueError("MCF sampled target identity or denominator differs")
        if sum(d["actual_use"] == "timely" for d in members) != int(row["timely"]):
            raise ValueError("MCF sampled timely count differs")
    print(
        "Verified 45 primary arms, five EOF arms, whole-window operand attribution, "
        "paired-core controls, continuous arrival order on two real traces, "
        "36 short arithmetic controls and six full-scale MCF interventions."
    )


if __name__ == "__main__":
    main()
