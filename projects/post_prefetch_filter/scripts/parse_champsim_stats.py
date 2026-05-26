#!/usr/bin/env python3
"""Parse ChampSim stdout/stderr log into one CSV row.

This parser is deliberately tolerant because ChampSim output formatting changes
across versions. It tries to extract:

- cumulative IPC / instruction count / cycles
- L1D, L2C, LLC access/hit/miss/hit-rate/miss-rate/MPKI
- prefetch requested/issued/useful/useless/accuracy when present

Usage:
  python3 projects/post_prefetch_filter/scripts/parse_champsim_stats.py \
    --trace 602.gcc_s-734B \
    --log projects/post_prefetch_filter/results/spp_baseline/logs/602.gcc_s-734B.spp_baseline.log

Append to an existing CSV:
  python3 .../parse_champsim_stats.py --trace ... --log ... --append-csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional

FIELDS = [
    "trace",
    "ipc",
    "instructions",
    "cycles",
    "L1D_access",
    "L1D_hit",
    "L1D_miss",
    "L1D_hit_rate",
    "L1D_miss_rate",
    "L1D_MPKI",
    "L2C_access",
    "L2C_hit",
    "L2C_miss",
    "L2C_hit_rate",
    "L2C_miss_rate",
    "L2C_MPKI",
    "LLC_access",
    "LLC_hit",
    "LLC_miss",
    "LLC_hit_rate",
    "LLC_miss_rate",
    "LLC_MPKI",
    "prefetch_requested",
    "prefetch_issued",
    "prefetch_useful",
    "prefetch_useless",
    "prefetch_accuracy",
]

CACHE_LEVELS = ["L1D", "L2C", "LLC"]


def first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def first_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def find_cache_line(level: str, text: str) -> Optional[str]:
    # Most useful lines look like "cpu0->L1D TOTAL ... HIT: ... MISS: ...".
    candidates = []
    for line in text.splitlines():
        if level in line and "TOTAL" in line and ("HIT" in line or "MISS" in line):
            candidates.append(line)
    return candidates[-1] if candidates else None


def parse_cache_level(level: str, text: str, instructions: Optional[int]) -> Dict[str, str]:
    row: Dict[str, str] = {}
    line = find_cache_line(level, text)

    hit = miss = access = None

    if line:
        hit = first_int(r"HIT:\s*([0-9,]+)", line)
        miss = first_int(r"MISS:\s*([0-9,]+)", line)
        access = first_int(r"ACCESS:\s*([0-9,]+)", line)
        if access is None and hit is not None and miss is not None:
            access = hit + miss

    # Fallback: search larger text for lines containing level and MPKI.
    mpki = None
    for ln in text.splitlines():
        if level in ln and "MPKI" in ln.upper():
            vals = re.findall(r"[0-9]+(?:\.[0-9]+)?", ln)
            if vals:
                mpki = float(vals[-1])
                break

    if mpki is None and miss is not None and instructions:
        mpki = miss * 1000.0 / instructions

    hit_rate = None
    miss_rate = None
    if access and access > 0:
        hit_rate = (hit or 0) / access
        miss_rate = (miss or 0) / access

    prefix = level
    row[f"{prefix}_access"] = "NA" if access is None else str(access)
    row[f"{prefix}_hit"] = "NA" if hit is None else str(hit)
    row[f"{prefix}_miss"] = "NA" if miss is None else str(miss)
    row[f"{prefix}_hit_rate"] = "NA" if hit_rate is None else f"{hit_rate:.6f}"
    row[f"{prefix}_miss_rate"] = "NA" if miss_rate is None else f"{miss_rate:.6f}"
    row[f"{prefix}_MPKI"] = "NA" if mpki is None else f"{mpki:.6f}"
    return row


def parse_prefetch_stats(text: str) -> Dict[str, str]:
    # ChampSim versions differ. Try common labels first.
    requested = first_int(r"prefetch(?:es)? requested[:= ]+([0-9,]+)", text)
    issued = first_int(r"prefetch(?:es)? issued[:= ]+([0-9,]+)", text)
    useful = first_int(r"prefetch(?:es)? useful[:= ]+([0-9,]+)", text)
    useless = first_int(r"prefetch(?:es)? useless[:= ]+([0-9,]+)", text)

    # SPP debug or final lines may contain GHR.pf_issued / GHR.pf_useful.
    if issued is None:
        vals = re.findall(r"GHR\.pf_issued:\s*([0-9,]+)", text)
        if vals:
            issued = int(vals[-1].replace(",", ""))
    if useful is None:
        vals = re.findall(r"GHR\.pf_useful:\s*([0-9,]+)", text)
        if vals:
            useful = int(vals[-1].replace(",", ""))

    if useless is None and issued is not None and useful is not None:
        useless = max(0, issued - useful)

    acc = None
    if issued and issued > 0 and useful is not None:
        acc = useful / issued

    return {
        "prefetch_requested": "NA" if requested is None else str(requested),
        "prefetch_issued": "NA" if issued is None else str(issued),
        "prefetch_useful": "NA" if useful is None else str(useful),
        "prefetch_useless": "NA" if useless is None else str(useless),
        "prefetch_accuracy": "NA" if acc is None else f"{acc:.6f}",
    }


def parse_log(trace: str, log_path: Path) -> Dict[str, str]:
    text = log_path.read_text(errors="replace")

    ipc = first_float(r"cumulative IPC:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if ipc is None:
        ipc = first_float(r"CPU\s+0\s+cumulative IPC:\s*([0-9]+(?:\.[0-9]+)?)", text)

    instructions = first_int(r"instructions:\s*([0-9,]+)", text)
    cycles = first_int(r"cycles:\s*([0-9,]+)", text)

    row: Dict[str, str] = {field: "NA" for field in FIELDS}
    row["trace"] = trace
    row["ipc"] = "NA" if ipc is None else f"{ipc:.6f}"
    row["instructions"] = "NA" if instructions is None else str(instructions)
    row["cycles"] = "NA" if cycles is None else str(cycles)

    for level in CACHE_LEVELS:
        row.update(parse_cache_level(level, text, instructions))

    row.update(parse_prefetch_stats(text))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--append-csv", type=Path, default=None)
    args = ap.parse_args()

    row = parse_log(args.trace, args.log)

    if args.append_csv is not None:
        with args.append_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerow(row)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
