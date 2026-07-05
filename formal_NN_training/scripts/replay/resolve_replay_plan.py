#!/usr/bin/env python3
from __future__ import print_function
import argparse
import csv
from pathlib import Path


def rows_from_plan(plan, root):
    items = []
    with Path(plan).open(newline="") as h:
        for row in csv.DictReader(h):
            tag = (row.get("tag") or "").strip()
            trace = (row.get("trace") or "").strip()
            value = (row.get("source_rel") or "").strip()
            if not tag or not trace or not value:
                raise ValueError("plan has a blank required field")
            source = Path(value)
            if not source.is_absolute():
                source = Path(root) / source
            if not source.is_file() or not source.stat().st_size:
                raise ValueError("plan list is missing: {}".format(source))
            items.append((tag, trace, source.resolve()))
    if not items:
        raise ValueError("plan has no rows")
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with Path(a.out).open("w") as h:
        for tag, trace, source in rows_from_plan(a.plan, a.root):
            h.write("{}\t{}\t{}\n".format(tag, trace, source))


if __name__ == "__main__":
    main()
