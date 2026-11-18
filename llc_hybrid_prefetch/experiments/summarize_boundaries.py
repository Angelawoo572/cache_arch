"""Join synthetic NoPF/Stride cache outcomes to known generator events by original load ID."""
import argparse
import csv
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from matrix import ROOT, SYNTHETIC, SYN_ROI, SYN_WARMUP, TRACE_DIR

FIELDS = ("group", "stage", "n", "base_llc_miss", "stride_llc_miss",
          "miss_to_hit", "miss_to_miss", "miss_to_no_lookup", "hit_to_miss",
          "stride_first_use", "stride_late_merge", "base_head_cycles", "stride_head_cycles")


def rows(path):
    with gzip.open(str(path), "rt") as stream:
        yield from csv.reader(stream)


def case_summary(pair):
    case, runs = pair
    known = {}
    source = TRACE_DIR / ("synthetic-" + case + ".boundaries.csv.gz")
    for row in rows(source):
        if row[3] == "roi":
            known[int(row[0])] = row[2] + ("_cross4k" if row[4] == "1" else "_same4k")
    info = json.loads((TRACE_DIR / ("synthetic-" + case + ".source.json")).read_text())
    interval = info["events_per_load"]
    first = 2 + ((SYN_WARMUP - 2) // interval + 1) * interval
    ids = range(first, SYN_WARMUP + SYN_ROI + 1, interval)
    outcomes = []
    for policy in ("nopf", "stride"):
        path = runs / ("syn-" + case + "-" + policy)
        if not (path / "complete.json").is_file():
            raise RuntimeError("missing simulation run: " + str(path))
        seen = set()
        status = {}
        for row in rows(path / "loads.csv.gz"):
            if row[6:8] == ["CORE", "trace"]:
                seen.add(int(row[1]))
            elif row[6:8] == ["LLC", "lookup"]:
                ordinal = int(row[1])
                if ordinal in status:
                    raise ValueError("duplicate LLC lookup")
                status[ordinal] = int(row[9])
        if seen != set(ids):
            raise ValueError("load ID range differs from synthetic generator")
        exposure = {int(i): int(c) for core, i, operand, c in rows(path / "head.csv.gz") if core == "0"}
        timely, late = set(), set()
        for row in rows(path / "candidates.csv.gz"):
            if row[8] in ("timely", "late") and row[10] == "0":
                (timely if row[8] == "timely" else late).add(int(row[11]))
        outcomes.append((status, exposure, timely, late))
    base, stride = outcomes
    groups = {}
    for i in ids:
        level = known.get(i, "other")
        counter = groups.setdefault(level, Counter())
        b, s = base[0].get(i), stride[0].get(i)
        counter.update(n=1, base_llc_miss=int(b == 0), stride_llc_miss=int(s == 0),
                       miss_to_hit=int(b == 0 and s == 1), miss_to_miss=int(b == 0 and s == 0),
                       miss_to_no_lookup=int(b == 0 and s is None), hit_to_miss=int(b == 1 and s == 0),
                       stride_first_use=int(i in stride[2]), stride_late_merge=int(i in stride[3]),
                       base_head_cycles=base[1].get(i, 0), stride_head_cycles=stride[1].get(i, 0))
    return [dict(group=case, stage=label, **counts) for label, counts in sorted(groups.items())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "raw/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "raw/synthetic_boundaries.csv")
    parser.add_argument("--jobs", type=int, default=4)
    opts = parser.parse_args()
    with ProcessPoolExecutor(opts.jobs) as pool:
        groups = list(pool.map(case_summary, [(case, opts.runs) for case in SYNTHETIC]))
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    with opts.out.open("w", newline="") as stream:
        table = csv.DictWriter(stream, FIELDS)
        table.writeheader()
        for group in groups:
            table.writerows(group)
