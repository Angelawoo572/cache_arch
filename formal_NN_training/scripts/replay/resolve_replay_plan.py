#!/usr/bin/env python3
from __future__ import print_function
import argparse
import csv
from pathlib import Path


def read_plan(plan, root):
    plan = Path(plan).resolve()
    root = Path(root).resolve()
    if not plan.is_file():
        raise ValueError("plan is missing: {}".format(plan))
    if not root.is_dir():
        raise ValueError("plan root is missing: {}".format(root))
    rows = []
    seen = set()
    with plan.open(newline="") as h:
        reader = csv.DictReader(h)
        required = set(["tag", "trace", "source_rel"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("plan is missing columns: {}".format(sorted(missing)))
        for line_no, raw in enumerate(reader, start=2):
            tag = (raw.get("tag") or "").strip()
            trace = (raw.get("trace") or "").strip()
            value = (raw.get("source_rel") or "").strip()
            if not tag or not trace or not value:
                raise ValueError("plan has a blank required field at row {}".format(line_no))
            if tag in seen:
                raise ValueError("plan has duplicate tag: {}".format(tag))
            seen.add(tag)
            source = Path(value)
            if not source.is_absolute():
                source = root / source
            source = source.resolve()
            if not source.is_file() or not source.stat().st_size:
                raise ValueError("plan list is missing: {}".format(source))
            row = dict(raw)
            row["tag"] = tag
            row["trace"] = trace
            row["source_rel"] = value
            row["rich_list"] = str(source)
            rows.append(row)
    if not rows:
        raise ValueError("plan has no rows")
    return rows


def rows_from_plan(plan, root):
    return [(row["tag"], row["trace"], Path(row["rich_list"])) for row in read_plan(plan, root)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as h:
        for tag, trace, source in rows_from_plan(a.plan, a.root):
            h.write("{}\t{}\t{}\n".format(tag, trace, source))
    print("[plan] {} entries".format(len(rows_from_plan(a.plan, a.root))))


if __name__ == "__main__":
    main()
