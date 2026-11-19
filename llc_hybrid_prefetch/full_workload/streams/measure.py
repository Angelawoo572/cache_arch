"""Group all paired original LLC tag misses by their program-order stream."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
ROOT = FULL.parents[1]
RAW = FULL / "raw/native"
ARMS = ("nopf", "stride", "multi", "spp")
FIELDS = {
    "event": "case policy view preceding current events changed_pc changed_page intervening_other_miss",
    "group": "case policy view minimum_original_misses streams never_direct_streams original_misses misses_in_never_direct_streams",
}


def run(case, exe):
    dirs = [RAW / "runs" / (case + "__" + arm) for arm in ARMS]
    configs = [json.loads((p / "complete.json").read_text())["config"] for p in dirs]
    config = configs[0]
    for key in ("case", "traces", "slots", "warmup", "roi", "skip"):
        if any(c[key] != config[key] for c in configs[1:]):
            raise ValueError(f"Unmatched {case} {key}")
    trace = ROOT / config["traces"][0]
    # Native slices are already extracted before the simulator's own skip/warmup.
    start = config["warmup"] + config["skip"]
    result = subprocess.run(
        [str(exe), str(trace), str(start), str(config["roi"]), *map(str, dirs), case],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.reader(result.stdout.splitlines()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+")
    args = p.parse_args()
    with (FULL / "intake/selection.csv").open() as f:
        cases = args.cases or (
            [r["case"] for r in csv.DictReader(f)]
            + ["bfs_discovery_far", "mcf17_discovery_far"]
        )
    exe = RAW / "episodes"
    RAW.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        ["c++", "-O3", "-std=c++14", str(HERE / "episodes.cc"), "-lz", "-o", str(exe)],
        check=True,
    )
    by_kind = {k: [] for k in FIELDS}
    for case in cases:
        for kind, *fields in run(case, exe):
            by_kind[kind].append(fields)
    for kind, header in FIELDS.items():
        dest = HERE / (kind + "_counts.csv")
        with dest.with_suffix(".tmp").open("w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header.split())
            writer.writerows(by_kind[kind])
        dest.with_suffix(".tmp").replace(dest)


if __name__ == "__main__":
    main()
