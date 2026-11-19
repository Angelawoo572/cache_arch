"""Audit final per-view coverage partition, source bounds, physical page restrictions and candidate accounting."""

import argparse
from pathlib import Path
from run import FULL, HERE, rows
from summarize import POLICIES, write


def validate(directory):
    performance = rows(directory / "performance.csv")
    quarters = rows(directory / "quarters.csv")
    joint = rows(directory / "joint_coverage.csv")
    result = []
    for case in dict.fromkeys(r["case"] for r in performance):
        cert = rows(FULL / "intake" / (case + ".validation.csv"))[0]
        arms = [r for r in performance if r["case"] == case]
        if {r["policy"] for r in arms} != set(POLICIES) or len(arms) != len(POLICIES):
            raise ValueError("Missing or duplicate arm")
        baseline = next(r for r in arms if r["policy"] == "nopf")
        nopf = int(baseline["llc_misses"])
        roi = int(baseline["measured_instructions"])
        warm = int(baseline["warmup"])
        start = int(cert["materialized_start"]) + warm + 1
        last = start + roi - 1
        if last > int(cert["source_records"]):
            raise ValueError("Measured beyond complete source")
        for r in arms:
            key = (case, r["policy"])
            if int(r["nopf_original_misses"]) != nopf:
                raise ValueError(("Wrong matched denominator", key))
            parts = (
                "orig_to_timely",
                "orig_to_late",
                "orig_to_persistent_miss",
                "orig_to_ordinary_hit",
                "orig_without_arm_tag",
            )
            if sum(int(r[k]) for k in parts) != nopf:
                raise ValueError(("Original misses do not partition", key))
            if int(r["accepted"]) + int(r["rejected"]) != int(r["proposed"]):
                raise ValueError(("Candidate partition", key))
            if int(r["proposed"]) > int(r["llc_demand_ingress"]):
                raise ValueError(("Degree-one proposal budget", key))
            if r["policy"] != "temporal" and int(r["accepted_crosspage"]) != 0:
                raise ValueError(("Same-page policy crossed page", key))
            if r["policy"] == "nopf" and any(
                int(r[k]) for k in ("accepted", "proposed", "dram_prefetch_bytes")
            ):
                raise ValueError(("NoPF issued a request", key))
            q = [a for a in quarters if (a["case"], a["policy"]) == key]
            if (
                len(q) != 4
                or int(q[0]["first_ordinal"]) != start
                or int(q[-1]["last_ordinal"]) != last
            ):
                raise ValueError(("Source bounds", key))
            for i in range(4):
                lo = int(q[i]["first_ordinal"])
                hi = int(q[i]["last_ordinal"])
                if (
                    lo != start + (roi * i + 3) // 4
                    or hi != start + (roi * (i + 1) + 3) // 4 - 1
                ):
                    raise ValueError(("Quarter bounds", key, i))
            if sum(int(a["cycles"]) for a in q) != int(r["cycles"]):
                raise ValueError(("Quarter execution time", key))
        j = next(r for r in joint if r["case"] == case and r["quarter"] == "all")
        if int(j["any_timely"]) + int(j["none_timely"]) != nopf:
            raise ValueError("Joint census")
        result.append(
            dict(
                case=case,
                validated_policies=len(arms),
                source_records=cert["source_records"],
                first_measured_ordinal=start,
                last_measured_ordinal=last,
                total_retired_across_policies=roi * len(arms),
                nopf_original_misses=nopf,
                exact_retirement_and_quarters=True,
                matched_partitions=True,
                degree_one_and_physical_scope=True,
            )
        )
    write(directory / "validation_summary.csv", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=HERE)
    validate(parser.parse_args().dir)
