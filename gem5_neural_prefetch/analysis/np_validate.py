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
    python analysis/np_validate.py values RUN_A RUN_B
        Which loads read different values in two arms that both recorded S2.
    python analysis/np_validate.py bound RUN_STOCK RUN_BOUNDED
        Does the bounded signature-path class reproduce gem5's own class?
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
    request = ~fields["evict"] & ~fields["swpf"] & ~fields["upgrade"]
    has_upgrade_flag = "v1_address_hash" in capture.counts
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
        # Captures taken before the upgrade flag existed count upgrade
        # requests among the hits, so their hit rows are not comparable.
        informational = label.endswith("hits") and not has_upgrade_flag
        if max(ours, theirs) > 1000 and not informational:
            worst = max(worst, diff)
        note = "   (includes upgrade requests)" if informational else ""
        print(f"{label:16s} {ours:12d} {theirs:12d}   {100 * diff:.3f}%{note}")
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


def bound(stock, bounded):
    """Does the bounded signature-path class reproduce a run of gem5's own
    class? Compares every statistic both runs report and the capture window."""
    a, b = read_stats(stock), read_stats(bounded)
    common = sorted(set(a) & set(b))
    differing = [k for k in common if a[k] != b[k]]
    stopped = [v for k, v in b.items() if k.endswith("lookaheadsStopped")]
    ca = json.loads((Path(stock) / "capture" / "capture_counts.json").read_text())
    cb = json.loads((Path(bounded) / "capture" / "capture_counts.json").read_text())
    same_window = all(ca[k] == cb[k] for k in ("enable_tick", "finish_tick", "committed_loads"))
    classes = [
        json.loads((Path(d) / "m5out" / "run_info.json").read_text()).get(
            "signature_path_class", "SignaturePathPrefetcherV2"
        )
        for d in (stock, bounded)
    ]
    print(f"classes: {classes[0]} and {classes[1]}")
    print(f"statistics in both runs: {len(common)}; differing: {len(differing)}")
    for key in differing[:12]:
        print(f"  {key}: {a[key]} and {b[key]}")
    print(f"capture window ticks and loads equal: {same_window}")
    print(f"lookahead walks stopped at the limit: {stopped[0] if stopped else 'not reported'}")
    ok = not differing and same_window
    print("BOUND SAME" if ok else "BOUND DIFFERS")
    return 0 if ok else 1


class LoadsOnly:
    """The part of a capture that V1 needs: load addresses and the counts.
    Much lighter than np_join.Arm, which also reads every cache event."""

    def __init__(self, run_dir):
        capture = Capture(Path(run_dir) / "capture")
        self.counts = capture.counts
        self.paddr = Stream(capture.folder, "load_time").column("paddr").copy()
        self.n = len(self.paddr)


def same(run_dirs):
    """V1: equal load PCs and addresses in every run. Values are reported
    separately: a program that reads a clock differs there and nowhere else."""
    from np_join import aligned
    from np_measure import value_hash

    first = LoadsOnly(run_dirs[0])
    ok = True
    hashes = {}  # what the hash covers -> the hashes seen
    for run_dir in run_dirs:
        arm = first if run_dir == run_dirs[0] else LoadsOnly(run_dir)
        same_addresses = aligned(first, arm)
        ok &= same_addresses
        kind, digest = value_hash(arm)
        hashes.setdefault(kind, set()).add(digest)
        print(
            f"loads {arm.n:9d}  uops {arm.counts['committed_uops']:9d}  "
            f"addresses {'same' if same_addresses else 'DIFFER'}  "
            f"value hash {digest}  {Path(run_dir).name}"
        )
    # Captures written before and after the hashes were split cannot be
    # compared with each other, so each kind is compared within itself.
    if any(len(seen) > 1 for seen in hashes.values()):
        print("load values differ across these runs")
    elif len(hashes) > 1:
        print(f"load values equal within each of the {len(hashes)} hash formats")
    else:
        print("load values equal across these runs")
    print("V1 PASS" if ok else "V1 FAIL")
    return 0 if ok else 1


def values(run_a, run_b):
    """Where do two arms read different values? Both must hold stream S2."""
    a = Capture(Path(run_a) / "capture").table("load")
    b = Capture(Path(run_b) / "capture").table("load")
    if len(a["pc"]) != len(b["pc"]):
        print("different numbers of loads")
        return 1
    same_place = bool(np.array_equal(a["pc"], b["pc"]) and np.array_equal(a["vaddr"], b["vaddr"]))
    differ = a["value"] != b["value"]
    print(f"loads {len(differ)}; same PC and address in every row: {same_place}")
    print(f"rows with a different value: {int(differ.sum())} ({100 * differ.mean():.5f}%)")
    pcs, counts = np.unique(a["pc"][differ], return_counts=True)
    print("load PCs that read different values (PC, differing rows, all rows of that PC):")
    for pc, count in sorted(zip(pcs, counts), key=lambda x: -x[1])[:16]:
        print(f"  {int(pc):#x}  {int(count):8d}  {int((a['pc'] == pc).sum()):8d}")
    return 0 if same_place else 1


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
    p = sub.add_parser("values")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p = sub.add_parser("intrusion")
    p.add_argument("with_capture")
    p.add_argument("without_capture")
    p = sub.add_parser("bound")
    p.add_argument("stock")
    p.add_argument("bounded")
    args = parser.parse_args()
    if args.check == "bound":
        return bound(args.stock, args.bounded)
    if args.check == "values":
        return values(args.run_a, args.run_b)
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
