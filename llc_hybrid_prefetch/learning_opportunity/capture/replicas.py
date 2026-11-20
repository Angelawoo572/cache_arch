"""Summarize independent Mac and Sacramento integrity checks without exposing machine-specific paths."""

import argparse
import csv
import json

from build import HERE, RAW
from run import CASES, POLICIES, external_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    rows = []
    certificates = RAW / (
        "external_verified_hosts" if args.external else "verified_hosts"
    )
    target = HERE / ("external_replicas.csv" if args.external else "replicas.csv")
    cases, arms = (
        (external_cases(), ("stride",)) if args.external else (CASES, POLICIES)
    )
    for case in sorted(cases):
        for arm in sorted(arms):
            name = case + "__" + arm + ".json"
            paths = [certificates / node / name for node in ("mac", "sacramento")]
            if args.partial and any(not path.is_file() for path in paths):
                continue
            left, right = (json.loads(path.read_text()) for path in paths)
            if left.get("schema") != 2 or right.get("schema") != 2:
                if args.partial:
                    continue
                raise ValueError(f"Refresh the replica certificates: {name}")
            if left.keys() != right.keys():
                raise ValueError(f"Different certificate schemas: {name}")
            for field in left:
                if left[field] != right[field]:
                    raise ValueError(f"{name}: cross-host difference in {field}")
            rows.append(
                {
                    "case": case,
                    "policy": arm,
                    "payload_origin": "Mac with verified copy"
                    if case
                    in (
                        "bfs_discovery",
                        "bfs_transfer",
                        "soplex06_heldout",
                        "pr_heldout",
                    )
                    else "independent simulation on both hosts",
                    "matched_observer_rows": left["logical_rows"],
                    "strictly_ordered_rows": left["logical_rows"]
                    - left["postrun_unresolved"]["rows"],
                    "postrun_inventory_records": left["postrun_unresolved"]["rows"],
                    "shutdown_cycle": left["postrun_unresolved"]["cycle"],
                    "matched_uncompressed_bytes": left["logical_bytes"],
                    "matched_outcome_words": left["config"]["roi"],
                    "matched_stream_blocks": sum(
                        len(blocks) for blocks in left["members"].values()
                    ),
                    "matched_fixed_work_cycles": left["cores"][0]["cycles"],
                }
            )
    if rows:
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(target)
    print("Verified cross-host replicas:", len(rows))


if __name__ == "__main__":
    main()
