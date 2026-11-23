#!/usr/bin/env python3
"""Measure every captured window against every prefetching arm and collect
the results as tables.

    python analysis/np_report.py [--arms stride,fast,bop,ampm,imp] [--only PREFIX]

Reads raw/runs/, writes one JSON per window and arm under raw/measure/ and the
tables under measurement/tables/ (tracked in git).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_measure import CHEAPEST_NAMES, Window, measure  # noqa: E402

STUDY = Path(os.environ.get("GEM5_STUDY", HERE.parent))
RUNS = STUDY / "raw" / "runs"
MEASURE = STUDY / "raw" / "measure"
TABLES = STUDY / "measurement" / "tables"
SLICER = STUDY / "raw" / "tools" / "np_slice"


def windows():
    """(workload, offset) pairs that have a finished no-prefetch capture."""
    found = []
    for counts in sorted(RUNS.glob("*__nopf/capture/capture_counts.json")):
        name = counts.parent.parent.name
        workload, offset, _ = name.rsplit("__", 2)
        found.append((workload, offset))
    return found


def edges_for(nopf_dir):
    edges = nopf_dir / "capture" / "edges.u64le"
    if not edges.exists():
        subprocess.check_call([str(SLICER), str(nopf_dir / "capture"), str(edges)])
    return edges


def flatten(prefix, value, row):
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, inner, row)
    elif not isinstance(value, list):
        row[prefix] = value


def write_table(name, rows):
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="stride,fast,bop,ampm,imp")
    parser.add_argument("--only", default="", help="workload name prefix")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    arms = args.arms.split(",")
    MEASURE.mkdir(parents=True, exist_ok=True)

    for workload, offset in windows():
        if not workload.startswith(args.only):
            continue
        nopf_dir = RUNS / f"{workload}__{offset}__nopf"
        window = None
        for arm in arms:
            fast_dir = RUNS / f"{workload}__{offset}__{arm}"
            if not (fast_dir / "capture" / "capture_counts.json").exists():
                continue
            out = MEASURE / f"{workload}__{offset}__{arm}.json"
            if out.exists() and not args.force:
                continue
            if window is None:
                window = Window(nopf_dir, edges_for(nopf_dir))
            result = measure(window, fast_dir)
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"measured {workload} {offset} {arm}", flush=True)
    write_tables()


def write_tables():
    """Rebuild every table from all results on disk, so that several
    invocations over different workloads can run side by side."""
    results = []
    for path in sorted(MEASURE.glob("*__o*__*.json")):
        if path.name.endswith(".walker.json"):
            continue
        workload, offset, arm = path.stem.rsplit("__", 2)
        results.append((workload, offset, arm, json.loads(path.read_text())))
    sections = {key: [] for key in ("window", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")}
    relations = []
    for workload, offset, arm, result in results:
        ident = {"workload": workload, "offset": offset[1:], "arm": arm}
        for key in sections:
            row = dict(ident)
            flatten("", result[key], row)
            sections[key].append(row)
        for rank, rel in enumerate(result["M5"].get("top", [])):
            relations.append({**ident, "rank": rank + 1, **rel})
    names = {
        "window": "windows.csv",
        "M1": "m1_residual.csv",
        "M2": "m2_mechanism.csv",
        "M3": "m3_lead.csv",
        "M4": "m4_entropy.csv",
        "M5": "m5_relation_reuse.csv",
        "M6": "m6_trigger_rate.csv",
        "M7": "m7_operand_age.csv",
        "M8": "m8_translation.csv",
    }
    for key, name in names.items():
        write_table(name, sections[key])
    write_table("m5_top_relations.csv", relations)
    print(f"{len(results)} window-arm results; tables in {TABLES}")


if __name__ == "__main__":
    main()
