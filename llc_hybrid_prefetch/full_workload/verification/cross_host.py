"""Compare independently completed PageRank runs, including all measured operand masks."""

import argparse
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("nopf", "stride", "multi", "spp")
ARTIFACTS = (
    "cores.csv",
    "phases.csv",
    "stats.csv",
    "pages.csv.gz",
    "outcomes_core0.u16le.gz",
)


def compare(left, right):
    path_open = gzip.open if left.suffix == ".gz" else open
    if left.name == "pages.csv.gz":
        with gzip.open(left, "rt") as a, gzip.open(right, "rt") as b:
            arows, brows = csv.reader(a), csv.reader(b)
            if next(arows) != next(brows):
                raise ValueError("Page-map schemas differ")
            mapping = sorted(arows)
            if mapping != sorted(brows):
                raise ValueError("Cross-host mapping differs")
            return len(mapping), "identical mapping rows"
    size = 0
    with path_open(left, "rb") as a, path_open(right, "rb") as b:
        while True:
            block = a.read(1 << 20)
            if block != b.read(1 << 20):
                raise ValueError(f"Cross-host disagreement in {left.name}")
            size += len(block)
            if not block:
                return size, "identical uncompressed bytes"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "remote_copy",
        type=Path,
        help="Directory containing the four independent run folders",
    )
    args = parser.parse_args()
    out = []
    for arm in ARMS:
        folder = "pr_heldout__" + arm
        local, independent = (
            ROOT / "raw/native/runs" / folder,
            args.remote_copy / folder,
        )
        config = json.loads((local / "complete.json").read_text())["config"]
        if config != json.loads((independent / "complete.json").read_text())["config"]:
            raise ValueError("Independent run differed in inputs or setup")
        for name in ARTIFACTS:
            size, unit = compare(local / name, independent / name)
            if name == "outcomes_core0.u16le.gz" and size != 2 * config["roi"]:
                raise ValueError("Partial ROI")
            out.append((arm, name, size, unit))
    with (ROOT / "verification/linux_repeat.csv").open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("policy", "matched_artifact", "compared_count", "agreement"))
        w.writerows(out)


if __name__ == "__main__":
    main()
