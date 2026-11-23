#!/usr/bin/env python3
"""Run the validation checks and write them as tables.

    python analysis/np_validation_table.py            # validation programs, V1 to V5
    python analysis/np_validation_table.py --windows  # every captured window, V1, V4, V5
    python analysis/np_validation_table.py --bound    # bounded signature-path class

Writes measurement/tables/validation.csv and validation_windows.csv. V2 needs
a run without the capture and V3 a program of known structure, so both exist
only for the validation programs. V6, the causal-prefix test, applies to
predictor features and is not part of the capture.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("GEM5_STUDY", HERE.parent))
RUNS = STUDY / "raw" / "runs"
TABLES = STUDY / "measurement" / "tables"
TOOL = [sys.executable, str(HERE / "np_validate.py")]
PROGRAMS = ("micro_stride", "micro_list", "micro_tree", "micro_csr", "micro_hash")
ARMS = ("nopf", "stride", "fast")
ALL_ARMS = ("nopf", "stride", "fast", "bop", "ampm", "imp")


def check(*args):
    done = subprocess.run(TOOL + [str(a) for a in args], capture_output=True, text=True)
    return done.stdout + done.stderr


def verdict(text, tag):
    return "pass" if f"{tag} PASS" in text else "FAIL"


def worst_difference(texts):
    """Largest count difference among the rows V4 judges, in percent."""
    worst = 0.0
    for text in texts:
        for line in text.splitlines():
            if "includes upgrade requests" in line:
                continue
            for value in re.findall(r"([0-9.]+)%", line):
                worst = max(worst, float(value))
    return worst


def write(name, rows):
    if not rows:
        print(f"nothing to write for {name}")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {TABLES / name}")


def programs():
    rows = []
    for program in PROGRAMS:
        run = {arm: RUNS / f"{program}__o0__{arm}" for arm in ARMS}
        if not all((d / "capture" / "capture_counts.json").exists() for d in run.values()):
            continue
        same = check("same", *run.values())
        intrusion = check("intrusion", run["nopf"], RUNS / f"{program}__o0__nopf__nocapture")
        storage = [check("storage", d) for d in run.values()]
        reconcile = [check("reconcile", d) for d in run.values()]
        structure = check("structure", run["nopf"], "--expect", program)

        first = same.splitlines()[0].split()  # loads N uops N addresses same value hash H name
        compared = re.search(r"statistics compared: (\d+); differing: (\d+)", intrusion)
        explained = re.search(r"explained by an affine relation ([0-9.]+)", structure)
        found = re.search(r"V3 \S+: expected (.*); found (.*)", structure)
        rows.append(
            {
                "program": program,
                "committed_loads": first[1],
                "committed_uops": first[3],
                "load_value_hash": first[8],
                "load_values_equal_in_3_arms": "yes" if "load values equal" in same else "no",
                "V1_same_loads_in_3_arms": verdict(same, "V1"),
                "V2_statistics_compared": compared.group(1) if compared else "",
                "V2_statistics_differing": compared.group(2) if compared else "",
                "V2_non_intrusion": verdict(intrusion, "V2"),
                "V3_expected_shift_base": found.group(1) if found else "",
                "V3_found_shift_base": found.group(2) if found else "",
                "V3_loads_explained": explained.group(1) if explained else "",
                "V3_known_structure": verdict(structure, "V3"),
                "V4_worst_count_difference_percent": f"{worst_difference(reconcile):.3f}",
                "V4_reconciliation_3_arms": "pass" if all("V4 PASS" in t for t in reconcile) else "FAIL",
                "V5_storage_3_arms": "pass" if all("V5 PASS" in t for t in storage) else "FAIL",
            }
        )
    write("validation.csv", rows)
    for row in rows:
        print(row["program"], *(row[k] for k in row if row[k] in ("pass", "FAIL")))
    return rows


def one_window(item):
    workload, offset = item
    run = {
        arm: RUNS / f"{workload}__{offset}__{arm}"
        for arm in ALL_ARMS
        if (RUNS / f"{workload}__{offset}__{arm}" / "capture" / "capture_counts.json").exists()
    }
    same = check("same", *run.values())
    storage = [check("storage", d) for d in run.values()]
    reconcile = [check("reconcile", d) for d in run.values()]
    first = same.splitlines()[0].split()
    row = {
        "workload": workload,
        "offset": offset[1:],
        "arms": " ".join(run),
        "committed_loads": first[1],
        "committed_uops": first[3],
        "load_values_equal_in_all_arms": "yes" if "load values equal" in same else "no",
        "V1_same_loads_in_all_arms": verdict(same, "V1"),
        "V4_worst_count_difference_percent": f"{worst_difference(reconcile):.3f}",
        "V4_reconciliation_all_arms": "pass" if all("V4 PASS" in t for t in reconcile) else "FAIL",
        "V5_storage_all_arms": "pass" if all("V5 PASS" in t for t in storage) else "FAIL",
    }
    print(*row.values(), flush=True)
    return row


def windows(prefix, jobs):
    items = []
    for counts in sorted(RUNS.glob(f"{prefix}*__nopf/capture/capture_counts.json")):
        workload, offset, _ = counts.parent.parent.name.rsplit("__", 2)
        items.append((workload, offset))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one_window, items))
    write("validation_windows.csv", rows)
    return rows


def bound_table(prefix):
    """Every window that has both a "fast" and a "fastcheck" capture."""
    rows = []
    for counts in sorted(RUNS.glob(f"{prefix}*__fastcheck/capture/capture_counts.json")):
        workload, offset, _ = counts.parent.parent.name.rsplit("__", 2)
        stock = RUNS / f"{workload}__{offset}__fast"
        if not (stock / "capture" / "capture_counts.json").exists():
            continue
        text = check("bound", stock, counts.parent.parent)
        classes = re.search(r"classes: (\S+) and (\S+)", text)
        compared = re.search(r"statistics in both runs: (\d+); differing: (\d+)", text)
        stopped = re.search(r"stopped at the limit: (\S+)", text)
        rows.append(
            {
                "workload": workload,
                "offset": offset[1:],
                "class_in_fast": classes.group(1) if classes else "",
                "class_in_fastcheck": classes.group(2) if classes else "",
                "statistics_compared": compared.group(1) if compared else "",
                "statistics_differing": compared.group(2) if compared else "",
                "capture_window_equal": "yes" if "loads equal: True" in text else "no",
                "lookaheads_stopped_in_fastcheck": stopped.group(1) if stopped else "",
                "same": "pass" if "BOUND SAME" in text else "FAIL",
            }
        )
        print(*rows[-1].values(), flush=True)
    write("validation_spp_bound.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", action="store_true", help="check every captured window")
    parser.add_argument("--bound", action="store_true", help="compare fast with fastcheck")
    parser.add_argument("--only", default="gap_", help="workload name prefix for --windows")
    parser.add_argument("-j", type=int, default=4)
    args = parser.parse_args()
    if args.bound:
        rows = bound_table(args.only)
    else:
        rows = windows(args.only, args.j) if args.windows else programs()
    return 0 if all("FAIL" not in row.values() for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
