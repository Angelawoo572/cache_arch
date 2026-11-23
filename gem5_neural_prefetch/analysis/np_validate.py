#!/usr/bin/env python3
"""Validation checks on captures.

    python analysis/np_validate.py structure RUN_DIR [--expect NAME]
        V3: slice the capture, fit relations, print them and, for a validation
        microbenchmark, compare with its constructed structure.
    python analysis/np_validate.py reconcile RUN_DIR
        V4: stream counts against gem5's own statistics.
    python analysis/np_validate.py storage RUN_DIR
        V5: every gzip member inflates and the index sums to the counts.
    python analysis/np_validate.py same RUN_DIR RUN_DIR ...
        V1: equal load hashes across arms or repeats.
    python analysis/np_validate.py intrusion RUN_WITH_CAPTURE RUN_WITHOUT
        V2: every gem5 statistic equal with the capture on and off.
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_relations import fit, read_edges  # noqa: E402
from np_streams import COLUMNS, EV_FILL, EV_HIT, EV_MISS, Capture, Stream, cache_fields  # noqa: E402

# Constructed ground truth of the microbenchmarks: (shift, base) of every
# relation that must exist. "array" stands for a large constant array base.
EXPECT = {
    "micro_stride": set(),
    "micro_list": {(0, 0), (0, 16)},
    "micro_tree": {(0, 0), (0, 8), (0, 16)},
    "micro_csr": {(3, "array")},
    "micro_hash": {(0, 0), (0, 8), (0, 16)},
}


def edges_for(run_dir):
    capture = Path(run_dir) / "capture"
    edges = capture / "edges.u64le"
    if not edges.exists():
        tool = Path(os.environ.get("GEM5_STUDY", HERE.parent)) / "raw" / "tools" / "np_slice"
        subprocess.check_call([str(tool), str(capture), str(edges)])
    return edges


def structure(run_dir, expect):
    capture = Capture(Path(run_dir) / "capture")
    load = capture.table("load")
    rel = fit(load, read_edges(edges_for(run_dir)))
    table = rel["relations"]
    n = len(load["pc"])
    print(f"loads {n}, explained by an affine relation {rel['derived'].mean():.4f}")

    rows = []
    if table:
        order = np.argsort(-table["instances"])
        for r in order[:20]:
            rows.append(
                (
                    int(table["instances"][r]),
                    int(table["producer_pc"][r]),
                    int(table["consumer_pc"][r]),
                    int(table["depth"][r]),
                    int(table["shift"][r]),
                    int(table["base"][r]),
                )
            )
    print("instances   producer pc -> consumer pc   depth shift base")
    for inst, ppc, cpc, depth, shift, base in rows:
        print(f"{inst:9d}   {ppc:#x} -> {cpc:#x}   {depth:3d} {shift:3d}   {base:#x}")

    print("per load PC: loads, explained share")
    pcs, counts = np.unique(load["pc"], return_counts=True)
    for pc, count in sorted(zip(pcs, counts), key=lambda x: -x[1])[:10]:
        mask = load["pc"] == pc
        print(f"  {int(pc):#x}  {int(count):9d}  {rel['derived'][mask].mean():.4f}")

    if expect is None:
        return 0
    wanted = EXPECT[expect]
    found = set()
    if table:
        strong = table["instances"] >= 0.01 * n
        for shift, base in zip(table["shift"][strong], table["base"][strong]):
            base = int(base)
            found.add((int(shift), base if base < 4096 else "array"))
    missing = wanted - found
    extra = found - wanted
    if expect == "micro_stride":
        ok = rel["derived"].mean() < 0.01
    else:
        ok = not missing
    print(f"V3 {expect}: expected {sorted(wanted, key=str)}; found {sorted(found, key=str)}")
    if extra:
        print(f"   additional relations found: {sorted(extra, key=str)}")
    print("V3 PASS" if ok else f"V3 FAIL: missing {sorted(missing, key=str)}")
    return 0 if ok else 1


def stat(text, name):
    match = re.search(rf"^{re.escape(name)}\s+(\d+)", text, re.M)
    return int(match.group(1)) if match else None


def reconcile(run_dir):
    run_dir = Path(run_dir)
    capture = Capture(run_dir / "capture")
    text = (run_dir / "m5out" / "stats.txt").read_text()
    events = capture.table("cache")
    fields = cache_fields(events["code"])
    # gem5 counts by command. A prefetch issued by the level above arrives
    # as a read, so gem5's "demand" counters at L2 and L3 include it, as they
    # include instruction fetches. Evictions are not accesses.
    request = ~fields["evict"] & ~fields["swpf"]
    hierarchy = "board.cache_hierarchy"
    rows = []
    for level, name in ((1, "l1d"), (2, "l2"), (3, "l3")):
        at = request & (fields["level"] == level)
        rows.append(
            (
                f"{name} misses",
                int((at & (fields["event"] == EV_MISS)).sum()),
                stat(text, f"{hierarchy}.{name}.demandMisses::total"),
            )
        )
        if level > 1:  # ordinary L1D hits are not recorded
            rows.append(
                (
                    f"{name} hits",
                    int((at & (fields["event"] == EV_HIT)).sum()),
                    stat(text, f"{hierarchy}.{name}.demandHits::total") or 0,
                )
            )
    rows.append(
        (
            "instructions",
            capture.counts["instructions_at_finish"]
            - capture.counts["instructions_before_window"],
            stat(text, "simInsts"),
        )
    )
    # The statistics window is cut at cycle boundaries and runs 2,000
    # instructions longer than the capture window.
    slack = 2000 / max(rows[-1][1], 1)
    worst = 0.0
    print(f"{'quantity':16s} {'capture':>12s} {'gem5 stats':>12s}")
    for label, ours, theirs in rows:
        diff = abs(ours - theirs) / max(theirs, ours, 1)
        if max(ours, theirs) > 1000:
            worst = max(worst, diff)
        print(f"{label:16s} {ours:12d} {theirs:12d}   {100 * diff:.3f}%")
    ok = worst < 3 * slack + 0.002
    print(f"committed loads  {capture.counts['committed_loads']:12d}")
    print("V4 PASS" if ok else "V4 FAIL")
    return 0 if ok else 1


def storage(run_dir):
    capture = Capture(Path(run_dir) / "capture")
    ok = True
    for name in COLUMNS:
        if not capture.has(name):
            continue
        stream = Stream(capture.folder, name)
        rows = 0
        with stream.path.open("rb") as file:
            for member in stream.members:
                file.seek(member["byte_offset"])
                raw = gzip.decompress(file.read(member["compressed_bytes"]))
                if len(raw) != member["rows"] * stream.width * 8:
                    ok = False
                rows += member["rows"]
        counted = capture.counts["rows"].get(name)
        size = stream.path.stat().st_size
        end = stream.members[-1]["byte_offset"] + stream.members[-1]["compressed_bytes"] if stream.members else 0
        good = rows == counted and size == end
        ok &= good
        print(f"{name:14s} rows {rows:10d}  counted {counted!s:>10s}  bytes {size:11d}  {'ok' if good else 'MISMATCH'}")
    print("V5 PASS" if ok else "V5 FAIL")
    return 0 if ok else 1


def read_stats(run_dir):
    """The last statistics dump of a run, without host-side quantities."""
    text = (Path(run_dir) / "m5out" / "stats.txt").read_text()
    block = text.rsplit("---------- Begin Simulation Statistics ----------", 1)[-1]
    stats = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].startswith(("host", "-")):
            continue
        stats[parts[0]] = parts[1]
    return stats


def intrusion(with_capture, without_capture):
    a, b = read_stats(with_capture), read_stats(without_capture)
    keys = sorted(set(a) | set(b))
    differing = [k for k in keys if a.get(k) != b.get(k)]
    print(f"statistics compared: {len(keys)}; differing: {len(differing)}")
    for key in differing[:20]:
        print(f"  {key}: {a.get(key)} with capture, {b.get(key)} without")
    print("V2 PASS" if not differing else "V2 FAIL")
    return 0 if not differing else 1


def same(run_dirs):
    hashes = {}
    for run_dir in run_dirs:
        counts = json.loads((Path(run_dir) / "capture" / "capture_counts.json").read_text())
        hashes[str(run_dir)] = (counts["v1_load_hash"], counts["committed_loads"], counts["committed_uops"])
        print(f"{counts['v1_load_hash']}  loads {counts['committed_loads']:9d}  uops {counts['committed_uops']:9d}  {Path(run_dir).name}")
    ok = len(set(hashes.values())) == 1
    print("V1 PASS" if ok else "V1 FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="check", required=True)
    p = sub.add_parser("structure")
    p.add_argument("run_dir")
    p.add_argument("--expect", choices=sorted(EXPECT), default=None)
    for name in ("reconcile", "storage"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
    p = sub.add_parser("same")
    p.add_argument("run_dirs", nargs="+")
    p = sub.add_parser("intrusion")
    p.add_argument("with_capture")
    p.add_argument("without_capture")
    args = parser.parse_args()
    if args.check == "intrusion":
        return intrusion(args.with_capture, args.without_capture)
    if args.check == "structure":
        return structure(args.run_dir, args.expect)
    if args.check == "reconcile":
        return reconcile(args.run_dir)
    if args.check == "storage":
        return storage(args.run_dir)
    return same(args.run_dirs)


if __name__ == "__main__":
    sys.exit(main())
