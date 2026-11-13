#!/usr/bin/env python3
"""Join each policy's directly used LLC prefetches to the matched NoPF misses."""

import argparse
import csv
import json
from pathlib import Path
import sys


EVENT_FIELDS = {"cpu", "instr_id", "operand", "pc", "virtual_line", "hit", "prefetch_use"}


def read_events(path, result):
    events = {}
    timely, late, misses = 0, 0, 0
    with path.open() as source:
        reader = csv.DictReader(source)
        if not EVENT_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("{} lacks identity or prefetch-use fields; rerun with --events".format(path))
        for row in reader:
            identity = tuple(int(row[field]) for field in ("cpu", "instr_id", "operand"))
            if identity in events:
                raise ValueError("{} has duplicate demand identity {}".format(path, identity))
            address = int(row["pc"]), int(row["virtual_line"])
            hit = row["hit"]
            use = row["prefetch_use"]
            if hit not in ("0", "1") or use not in ("none", "timely", "late"):
                raise ValueError("{} has an invalid outcome for {}".format(path, identity))
            if (use == "timely" and hit != "1") or (use == "late" and hit != "0"):
                raise ValueError("{} has inconsistent hit and prefetch use for {}".format(path, identity))
            events[identity] = address, hit == "1", use
            timely += use == "timely"
            late += use == "late"
            misses += hit == "0"

    expected = tuple(result[field] for field in (
        "llc_demand_accesses", "llc_demand_misses", "prefetch_timely", "prefetch_late",
    ))
    if (len(events), misses, timely, late) != expected:
        raise ValueError("{} does not reconcile with the simulator summary".format(path))
    return events


def rate(count, total):
    return count / total if total else None


def coverage_rows(run):
    with (run / "summary.json").open() as source:
        results = json.load(source)
    baseline_result = next((result for result in results if result["policy"] == "nopf"), None)
    if baseline_result is None:
        raise ValueError("the run must include NoPF")
    baseline = read_events((run / baseline_result["log"]).with_name("demand.csv"), baseline_result)
    baseline_misses = {identity for identity, (_, hit, _) in baseline.items() if not hit}
    total = len(baseline_misses)

    for result in results:
        policy = result["policy"]
        events = baseline if policy == "nopf" else read_events((run / result["log"]).with_name("demand.csv"), result)
        for identity in baseline.keys() & events.keys():
            if baseline[identity][0] != events[identity][0]:
                raise ValueError("{} changed PC or virtual line for {}".format(policy, identity))

        matched = baseline_misses & events.keys()
        timely = sum(events[identity][2] == "timely" for identity in matched)
        late = sum(events[identity][2] == "late" for identity in matched)
        yield {
            "policy": policy,
            "nopf_misses": total,
            "matched_nopf_misses": len(matched),
            "match_rate": rate(len(matched), total),
            "matched_timely_uses": timely,
            "matched_late_uses": late,
            "direct_coverage": rate(timely + late, total),
            "direct_timely_coverage": rate(timely, total),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="matched run directory created with run.py --events")
    args = parser.parse_args()
    try:
        rows = list(coverage_rows(args.run))
    except (OSError, ValueError, KeyError) as error:
        print("Coverage report failed: {}".format(error), file=sys.stderr)
        return 1
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
