"""Measure lead and service on the fixed in-window trigger cohort using native request identities."""

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

from run import FULL, HERE, RAW, rows
from summarize import FAR, POLICIES, write


def within(identifier, quarters, width):
    return any(
        start <= identifier < min(start + width, end + 1) for start, end in quarters
    )


def gzrows(path):
    with gzip.open(path, "rt") as f:
        yield from csv.DictReader(f)


def sorted_quantile(values, p):
    # Inverse empirical CDF, including the first and last observation.
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else ""


def measure(case, policy, folder):
    q = [
        (int(r["first_ordinal"]), int(r["last_ordinal"]))
        for r in rows(folder / "phases.csv")
    ]
    width = 250000
    ingress = {}
    for r in gzrows(folder / "loads.csv.gz"):
        if r["level"] == "LLC" and r["stage"] == "ingress":
            ingress[(int(r["instr_id"]), int(r["operand"]))] = int(r["cycle"])
    cohorts = {}
    for r in gzrows(folder / "candidates.csv.gz"):
        rid, stage = int(r["id"]), r["stage"]
        if stage == "accepted":
            if not within(int(r["instr_id"]), q, width):
                raise ValueError("Unregistered issued trigger")
            cohorts[rid] = dict(issue=int(r["cycle"]), target=int(r["target"]))
        if rid not in cohorts:
            continue
        req = cohorts[rid]
        if stage in (
            "dispatch",
            "fill",
            "late",
            "timely",
            "evicted",
            "unresolved",
            "redundant",
        ):
            req[stage] = int(r["cycle"])
        if stage in ("timely", "late"):
            req["demand"] = (int(r["demand_instr_id"]), int(r["demand_operand"]))
    out = []
    for stage in ("timely", "late", "evicted", "unresolved", "redundant"):
        claims = [req for req in cohorts.values() if stage in req]
        metrics = defaultdict(list)
        for req in claims:
            for key, start, end in (
                ("request_to_dispatch", "issue", "dispatch"),
                ("dispatch_to_fill", "dispatch", "fill"),
                ("request_to_fill", "issue", "fill"),
            ):
                if start in req and end in req:
                    metrics[key].append(req[end] - req[start])
            if "demand" in req:
                arrival = ingress.get(req["demand"])
                metrics["request_to_first_claim"].append(req[stage] - req["issue"])
                if arrival is not None:
                    metrics["request_to_demand_ingress"].append(arrival - req["issue"])
                    if "fill" in req:
                        metrics["fill_margin_at_demand_ingress"].append(
                            arrival - req["fill"]
                        )
                if "fill" in req:
                    metrics["fill_margin_at_tag_or_merge"].append(
                        req[stage] - req["fill"]
                    )
        for key in (
            "request_to_dispatch",
            "dispatch_to_fill",
            "request_to_fill",
            "request_to_first_claim",
            "request_to_demand_ingress",
            "fill_margin_at_demand_ingress",
            "fill_margin_at_tag_or_merge",
        ):
            values = metrics[key]
            out.append(
                dict(
                    case=case,
                    policy=policy,
                    claim=stage,
                    accepted_sample=len(cohorts),
                    cohort_size=len(claims),
                    interval_cycles=key,
                    observed=len(values),
                    p10=sorted_quantile(values, 0.1),
                    p50=sorted_quantile(values, 0.5),
                    p90=sorted_quantile(values, 0.9),
                )
            )
    if sum(
        r["cohort_size"] for r in out if r["interval_cycles"] == "request_to_fill"
    ) != len(cohorts):
        raise ValueError("Sampled accepted lifecycle fails to partition")
    return out


def main(out, only_complete, cases=None):
    results = []
    roster = cases or [r["case"] for r in rows(FULL / "intake/selection.csv")] + list(
        FAR
    )
    for case in roster:
        for policy in POLICIES:
            if policy == "nopf":
                continue
            folder = RAW / "runs" / (case + "__" + policy)
            if not (folder / "complete.json").exists():
                if only_complete:
                    continue
                raise ValueError("Missing full run: " + str(folder))
            results.extend(measure(case, policy, folder))
    write(out / "sampled_timeliness.csv", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", help="explicit view IDs for a separate secondary panel"
    )
    args = parser.parse_args()
    main(args.out, args.completed_only, args.cases)
