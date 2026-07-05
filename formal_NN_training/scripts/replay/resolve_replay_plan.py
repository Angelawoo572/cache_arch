#!/usr/bin/env python3
"""Validate and resolve one replay-plan CSV for all replay consumers."""
from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path

REQUIRED = set(["tag", "trace", "source_rel"])
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_plan(plan, root):
    plan = Path(plan).resolve()
    root = Path(root).resolve()
    if not plan.is_file():
        raise ValueError("replay plan is missing: {}".format(plan))
    if not root.is_dir():
        raise ValueError("replay plan root is missing: {}".format(root))
    rows = []
    seen = set()
    with plan.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("replay plan missing columns: {}".format(sorted(missing)))
        for line_no, raw in enumerate(reader, start=2):
            tag = (raw.get("tag") or "").strip()
            trace = (raw.get("trace") or "").strip()
            source_rel = (raw.get("source_rel") or "").strip()
            if not tag or not trace or not source_rel:
                raise ValueError("blank tag/trace/source_rel at row {}".format(line_no))
            if not SAFE.match(tag) or not SAFE.match(trace):
                raise ValueError("unsafe tag or trace at row {}".format(line_no))
            if tag in seen:
                raise ValueError("duplicate replay-plan tag: {}".format(tag))
            seen.add(tag)
            source = Path(source_rel)
            source = source if source.is_absolute() else root / source
            source = source.resolve()
            if not source.is_file() or source.stat().st_size <= 0:
                raise ValueError("missing/nonempty rich list for {}: {}".format(tag, source))
            row = dict(raw)
            row["tag"] = tag
            row["trace"] = trace
            row["source_rel"] = source_rel
            row["rich_list"] = str(source)
            rows.append(row)
    if not rows:
        raise ValueError("replay plan has no entries")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = read_plan(args.plan, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write("{}\t{}\t{}\n".format(row["tag"], row["trace"], row["rich_list"]))
    print("[plan] {} entries".format(len(rows)))


if __name__ == "__main__":
    main()
