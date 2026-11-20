"""Build a compact registry for the registered long captures after parity validation."""

import argparse
import csv
import json

from build import FULL, HERE, RAW
from run import CASES, POLICIES, external_cases


def collect(cases, policies, partial):
    rows = []
    for case in sorted(cases):
        for policy in sorted(policies):
            run = RAW / "runs" / (case + "__" + policy)
            marker = run / "capture_validation.json"
            if not marker.exists() and partial:
                continue
            row = json.loads(marker.read_text())
            meta = json.loads((run / "complete.json").read_text())["config"]
            row["raw_directory"] = str(run.relative_to(FULL.parents[1]))
            row["input_view"] = meta["traces"][0]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Publish the separately registered external Stride runs",
    )
    args = parser.parse_args()
    panels = (
        ("captures.csv", CASES, POLICIES),
        ("fixture.csv", ["bfs_discovery_fixture"], ["nopf", "spp"]),
    )
    if args.external:
        panels = (("external_captures.csv", external_cases(), ["stride"]),)
    for filename, cases, policies in panels:
        rows = collect(cases, policies, args.partial)
        if not rows:
            continue
        tmp = (HERE / filename).with_suffix(".tmp")
        with tmp.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(HERE / filename)
        print(filename, len(rows))


if __name__ == "__main__":
    main()
