"""Summarize independent Mac and Sacramento integrity checks without exposing machine-specific paths."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

from build import BUILD, HERE, RAW
from run import CASES, POLICIES, external_cases


def record_origins(certificates, cases, arms):
    """Run on the receiving host. Each completed run retains its original command."""
    output = certificates / "sacramento/origins.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in sorted(cases):
        for arm in sorted(arms):
            name = case + "__" + arm
            completion = RAW / "runs" / name / "complete.json"
            if not completion.is_file():
                continue
            command = shlex.split(json.loads(completion.read_text())["command"])
            executable = Path(command[0])
            if executable == BUILD / "bin/champsim":
                origin = "independent simulation on both hosts"
            elif executable.parts[:2] == ("/", "Users") and str(executable).endswith(
                "/learning_opportunity/raw/capture/build/pythia/bin/champsim"
            ):
                origin = "Mac with verified copy"
            else:
                raise ValueError(
                    f"Unrecognized capture executable for {name}: {executable}"
                )
            rows.append((name, origin))
    temp = output.with_suffix(".tmp")
    with temp.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("run", "origin"))
        writer.writerows(rows)
    temp.replace(output)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument(
        "--record-origins",
        action="store_true",
        help="Read actual run commands on Sacramento",
    )
    args = parser.parse_args()
    rows = []
    certificates = RAW / (
        "external_verified_hosts" if args.external else "verified_hosts"
    )
    target = HERE / ("external_replicas.csv" if args.external else "replicas.csv")
    cases, arms = (
        (external_cases(), ("stride",)) if args.external else (CASES, POLICIES)
    )
    if args.record_origins:
        if not sys.platform.startswith("linux"):
            parser.error("Record origins on the receiving Linux host")
        print("Recorded payload origins:", record_origins(certificates, cases, arms))
        return
    origin_path = certificates / "sacramento/origins.csv"
    if origin_path.is_file():
        with origin_path.open(newline="") as file:
            origins = {r["run"]: r["origin"] for r in csv.DictReader(file)}
    elif args.partial:
        return
    else:
        raise ValueError(
            "Record run origins on Sacramento and copy its certificate directory"
        )
    for case in sorted(cases):
        for arm in sorted(arms):
            stem = case + "__" + arm
            name = stem + ".json"
            paths = [certificates / node / name for node in ("mac", "sacramento")]
            if args.partial and (
                stem not in origins or any(not path.is_file() for path in paths)
            ):
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
                    "payload_origin": origins[stem],
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
