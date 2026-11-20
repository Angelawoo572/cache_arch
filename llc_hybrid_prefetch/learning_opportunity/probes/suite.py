"""Run the registered transfer matrix as native data becomes available."""

import argparse
import csv
import time

from extract import RAW, STUDY, extract, locate
from fit import run

ORIGINAL = (
    "bfs_discovery",
    "bfs_discovery_far",
    "bfs_transfer",
    "mcf17_discovery",
    "mcf17_discovery_far",
    "mcf17_transfer",
    "pr_heldout",
    "mcf06_discovery",
    "soplex06_heldout",
)
MODES = (
    "prior",
    "current",
    "same_pc",
    "local_region",
    "global32",
    "hide_other_pc",
    "with_time",
)
ENCODINGS = ("delta", "page_offset")


def pairs():
    with (STUDY / "expansion/sources.csv").open() as f:
        external = [r["case"] for r in csv.DictReader(f)]
    tests = [(c, c) for c in ORIGINAL if c not in ("bfs_discovery", "soplex06_heldout")]
    tests += [("bfs_discovery", c) for c in external]
    tests += [
        ("bfs_discovery", "bfs_transfer"),
        ("bfs_discovery", "bfs_discovery_far"),
        ("mcf17_discovery", "mcf17_transfer"),
        ("mcf17_discovery", "mcf17_discovery_far"),
        ("soplex06_heldout", "soplex247"),
        ("pr_heldout", "pr14"),
        ("mcf06_discovery", "mcf51"),
        ("mcf17_layout_original", "mcf17_layout_original"),
        ("mcf17_layout_original", "mcf17_layout_reversed"),
        ("gcc734", "gcc734"),
        ("gcc734", "gcc734_slot1"),
        ("cc5", "cc5"),
        ("cc5", "cc5_slot1"),
    ]
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    plan = pairs()
    while plan:
        remain = []
        for source, dest in plan:
            out = RAW / "fits" / f"matrix_{source}_to_{dest}"
            if (out / "scores.csv").is_file() and (out / "blocks.csv").is_file():
                continue
            try:
                for case in {source, dest}:
                    locate(case, "stride")
            except FileNotFoundError:
                remain.append((source, dest))
                continue
            for case in {source, dest}:
                if not (RAW / f"{case}__stride.npz").is_file():
                    extract(case, "stride")
            run(source, [dest], MODES, ENCODINGS, out)
        plan = remain
        if not args.wait:
            break
        if plan:
            time.sleep(20)


if __name__ == "__main__":
    main()
