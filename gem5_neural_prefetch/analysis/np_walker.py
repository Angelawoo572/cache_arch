#!/usr/bin/env python3
"""Open-loop bound on what a bounded traversal could cover in time (M3, M9).

    python analysis/np_walker.py [--arms fast] [--only gap_]

For every residual target whose address is affine in a loaded value, walk up
its dependence chain to a root whose own address continues a same-PC delta.
A traversal can start at such a root before the program gets there: it steps
the root `d` instances ahead, then follows the chain, paying one fetch per
link. The target is covered in time if the program takes at least that long to
go from the root instance `d` steps back to the target's issue.

  hops      links between the root and the target (1 = the root's value names
            the target)
  distance  how many instances ahead the root is stepped
  fetch     the latency each line had without any prefetching (measured)
  pace      issue times in the prefetching arm (measured)

A second way to be early needs no address prediction. A line holds several
fields. When the first load of a line returns, the other fields of that line
are visible too, before the loads that will read them. A root that sits in
such a line gives the traversal the time until the program reaches it
("same line").

A third way starts at a fill instead of a load. A line that a conventional
prefetcher brings into L2 is visible there before any load reads it. A root
in such a line gives the traversal the time from that fill to the target
("fill tap"). Only fills inside the window count.

Both are upper bounds at the measured pace. They ignore queue and bandwidth
limits, they assume the right fields are followed and no others, and a
program that speeds up needs a longer distance for the same coverage. Stage 2
replaces them with a closed-loop measurement.

Writes raw/measure/<window>__<arm>.walker.json and
measurement/tables/m3_walker_bound.csv.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_join import Arm, aligned, fast_outcome  # noqa: E402
from np_streams import EV_FILL  # noqa: E402
from np_measure import Window, previous_same, share  # noqa: E402
from np_report import MEASURE, RUNS, TABLES, edges_for, windows  # noqa: E402

MAX_HOPS = 3
DISTANCES = (1, 2, 4, 8, 16, 32, 64)
NEVER = 1 << 30


def equal_delta_run(pc, vaddr):
    """Consecutive equal same-PC address steps that end at each load."""
    n = len(pc)
    prev = previous_same(pc)
    step = vaddr.astype(np.int64) - vaddr[np.maximum(prev, 0)].astype(np.int64)
    prev2 = np.where(prev >= 0, prev[np.maximum(prev, 0)], -1)
    equal = (prev2 >= 0) & (step == step[np.maximum(prev, 0)])
    order = np.lexsort((np.arange(n), pc))
    e = equal[order]
    position = np.arange(n)
    last_reset = np.maximum.accumulate(np.where(~e, position, -1))
    run_sorted = np.where(e, position - last_reset + 1, (prev[order] >= 0).astype(np.int64))
    run = np.zeros(n, dtype=np.int64)
    run[order] = run_sorted
    return run, prev


def walker_bound(window, fast, detail=None):
    """The bound for one window and arm. If `detail` is a dict it receives
    per-target arrays: the load index, its stall, and which ways reach it."""
    nopf, load, relations = window.nopf, window.load, window.relations
    original = nopf.miss[2]
    outcome = fast_outcome(fast, 2)
    residual = original & ((outcome == 1) | (outcome == 2))
    target = np.flatnonzero(residual & relations["derived"])
    weight = fast.stall[target]
    total = float(weight.sum())
    out = {
        "targets": int(len(target)),
        "stall": total,
        "residual_stall_all_mechanisms": float(fast.stall[residual].sum()),
    }
    if not len(target) or total <= 0:
        return out

    run, prev = equal_delta_run(load["pc"], load["vaddr"])
    origin, derived = relations["origin"], relations["derived"]
    issue = fast.t_issue.astype(np.float64) / fast.period
    fetch = nopf.latency
    first = first_touch(fast)
    seen = np.where(fast.t_data[first] > 0, fast.t_data[first], fast.t_issue[first]) / fast.period
    # A field read early is only useful if nothing wrote it afterwards.
    last_store = last_store_before(load, nopf.capture.table("store"))
    load_uop = load["uop"].astype(np.int64)
    unchanged = last_store < load_uop[first]
    # Fill tap: the latest L2 fill of each load's line before the load issues,
    # and whether the field already held its value at that fill.
    filled, has_fill = l2_fill_before(fast)
    committed = np.searchsorted(fast.t_commit, filled, side="right") - 1
    uop_at_fill = np.where(committed >= 0, load_uop[np.maximum(committed, 0)], -1)
    fill_usable = has_fill & ((last_store < 0) | (last_store < uop_at_fill))
    filled = filled / fast.period

    need = fetch[target].copy()  # cycles to fetch the chain below the root
    node = target.copy()
    alive = np.ones(len(target), dtype=bool)
    step_best = {}  # hops -> smallest distance that is in time, over roots within hops
    line_best = {}  # hops -> covered through a root on an already visible line
    step_root = np.full(len(target), -1, dtype=np.int64)
    line_root = np.full(len(target), -1, dtype=np.int64)
    best_all = np.full(len(target), NEVER, dtype=np.int64)
    line_all = np.zeros(len(target), dtype=bool)
    fill_all = np.zeros(len(target), dtype=bool)
    fill_best = {}
    for hops in range(1, MAX_HOPS + 1):
        node = np.where(alive, origin[np.maximum(node, 0)], node)
        at = np.maximum(node, 0)

        # Same line: the root's value has been visible since the first load
        # of its line returned.
        in_line = alive & ~line_all & unchanged[at] & (issue[target] - seen[at] >= need)
        line_root[in_line] = first[at][in_line]
        line_all |= in_line
        line_best[hops] = line_all.copy()

        # Fill tap: the root's line was filled into L2 earlier in the window.
        fill_all |= alive & fill_usable[at] & (issue[target] - filled[at] >= need)
        fill_best[hops] = fill_all.copy()

        # Stepped root: the root itself must be fetched too.
        need_step = need + fetch[at]
        longest = np.where(alive, run[at] - 2, -1)  # usable distance
        required = np.full(len(target), NEVER, dtype=np.int64)
        back = node.copy()
        for distance in range(1, DISTANCES[-1] + 1):
            back = np.where(back >= 0, prev[np.maximum(back, 0)], -1)
            can = alive & (back >= 0) & (distance <= longest) & (required == NEVER)
            in_time = can & (issue[target] - issue[np.maximum(back, 0)] >= need_step)
            required[in_time] = distance
        better = required < best_all
        step_root[better] = node[better]
        best_all = np.minimum(best_all, required)
        step_best[hops] = best_all.copy()

        need = need + np.where(alive, fetch[at], 0.0)
        alive &= derived[at]

    for hops in range(1, MAX_HOPS + 1):
        out[f"same_line.hops_{hops}"] = share(weight[line_best[hops]].sum(), total)
        for distance in DISTANCES:
            stepped = step_best[hops] <= distance
            out[f"stepped.hops_{hops}.distance_{distance}"] = share(weight[stepped].sum(), total)
        for distance in (16, 64):
            either = line_best[hops] | (step_best[hops] <= distance)
            out[f"either.hops_{hops}.distance_{distance}"] = share(weight[either].sum(), total)
        out[f"fill_tap.hops_{hops}"] = share(weight[fill_best[hops]].sum(), total)
        any_way = line_best[hops] | fill_best[hops] | (step_best[hops] <= DISTANCES[-1])
        out[f"any.hops_{hops}"] = share(weight[any_way].sum(), total)

    stepped = best_all <= DISTANCES[-1]
    covered = stepped | line_all | fill_all
    if detail is not None:
        detail.update(target=target, stall=weight, stepped=stepped, same_line=line_all,
                      fill_tap=fill_all, any=covered, stepped_distance=best_all)
    out["covered_targets"] = int(covered.sum())
    # Width: how many targets one start must name.
    for name, chosen, root in (("stepped", stepped, step_root), ("same_line", line_all, line_root)):
        if chosen.any():
            _, per_root = np.unique(root[chosen], return_counts=True)
            out[f"targets_per_start.{name}"] = {
                f"p{p}": float(np.percentile(per_root, p)) for p in (50, 90, 99)
            }
    # The same bound as a share of all residual stall, value-derived or not.
    out["share_of_all_residual_stall.any.hops_3"] = share(
        weight[covered].sum(), out["residual_stall_all_mechanisms"]
    )
    return out


def l2_fill_before(arm):
    """For every load: the tick of the latest L2 data fill of its line at or
    before the load issues, and whether there is one inside the window."""
    events, fields = arm.events, arm.fields
    rows = np.flatnonzero((fields["level"] == 2) & (fields["event"] == EV_FILL) & ~fields["inst"])
    tick = np.zeros(arm.n, dtype=np.float64)
    found = np.zeros(arm.n, dtype=bool)
    if not len(rows):
        return tick, found
    line = arm.paddr >> np.uint64(6)
    t0 = int(min(events["tick"][rows].min(), arm.t_issue[arm.t_issue > 0].min()))
    _, dense = np.unique(np.concatenate((events["paddr_line"][rows], line)), return_inverse=True)
    dense = dense.astype(np.uint64)
    fill_key = np.sort((dense[: len(rows)] << np.uint64(40)) | (events["tick"][rows] - np.uint64(t0)))
    issue = np.maximum(arm.t_issue - t0, 0).astype(np.uint64)
    load_key = (dense[len(rows) :] << np.uint64(40)) | issue
    position = np.searchsorted(fill_key, load_key, side="right") - 1
    at = np.maximum(position, 0)
    found = (position >= 0) & ((fill_key[at] >> np.uint64(40)) == (load_key >> np.uint64(40)))
    tick[found] = (fill_key[at][found] & np.uint64((1 << 40) - 1)).astype(np.float64) + t0
    return tick, found


def last_store_before(load, store):
    """Micro-op ordinal of the latest committed store that wrote any byte of
    the 8-byte word a load reads, before that load. -1 if there is none."""
    result = np.full(len(load["uop"]), -1, dtype=np.int64)
    if not len(store["uop"]):
        return result
    words, ordinals = [], []
    size = (store["size_flags"] & np.uint64(0xFF)).astype(np.int64)
    low = (store["vaddr"] & np.uint64(7)).astype(np.int64)
    first_word = store["vaddr"] >> np.uint64(3)
    for extra in range(8):  # a store of up to 64 bytes covers up to nine words
        covers = low + size > 8 * extra
        if not covers.any():
            break
        words.append(first_word[covers] + np.uint64(extra))
        ordinals.append(store["uop"][covers])
    words, ordinals = np.concatenate(words), np.concatenate(ordinals)
    load_word = load["vaddr"] >> np.uint64(3)
    _, dense = np.unique(np.concatenate((words, load_word)), return_inverse=True)
    dense = dense.astype(np.uint64)
    store_key = np.sort((dense[: len(words)] << np.uint64(36)) | ordinals)
    load_key = (dense[len(words) :] << np.uint64(36)) | load["uop"]
    position = np.searchsorted(store_key, load_key, side="left") - 1
    found = position >= 0
    at = np.maximum(position, 0)
    found &= (store_key[at] >> np.uint64(36)) == (load_key >> np.uint64(36))
    result[found] = (store_key[at][found] & np.uint64((1 << 36) - 1)).astype(np.int64)
    return result


def first_touch(arm):
    """For every load, the first load that touched its line in the current
    L1D residency: the load at which the whole line became visible."""
    n = arm.n
    line = arm.paddr >> np.uint64(6)
    order = np.lexsort((np.arange(n), line))
    sorted_line = line[order]
    start = np.ones(n, dtype=bool)
    start[1:] = sorted_line[1:] != sorted_line[:-1]
    start |= (arm.miss[1] | arm.pf_hit[1])[order]
    position = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    first = np.empty(n, dtype=np.int64)
    first[order] = order[position]
    return first


def flatten(prefix, value, row):
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, inner, row)
    else:
        row[prefix] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="fast")
    parser.add_argument("--only", default="gap_")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for workload, offset in windows():
        if not workload.startswith(args.only):
            continue
        nopf_dir = RUNS / f"{workload}__{offset}__nopf"
        window = None
        for arm in args.arms.split(","):
            fast_dir = RUNS / f"{workload}__{offset}__{arm}"
            if not (fast_dir / "capture" / "capture_counts.json").exists():
                continue
            out = MEASURE / f"{workload}__{offset}__{arm}.walker.json"
            if out.exists() and not args.force:
                continue
            if window is None:
                window = Window(nopf_dir, edges_for(nopf_dir))
            fast = Arm(fast_dir)
            if not aligned(window.nopf, fast):
                raise SystemExit(f"V1 failed for {fast_dir}")
            out.write_text(json.dumps(walker_bound(window, fast), indent=2) + "\n")
            print(f"walker bound {workload} {offset} {arm}", flush=True)

    # The table is rebuilt from every result on disk.
    rows = []
    for path in sorted(MEASURE.glob("*.walker.json")):
        workload, offset, arm = path.name[: -len(".walker.json")].rsplit("__", 2)
        row = {"workload": workload, "offset": offset[1:], "arm": arm}
        flatten("", json.loads(path.read_text()), row)
        rows.append(row)
    if rows:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        TABLES.mkdir(parents=True, exist_ok=True)
        with (TABLES / "m3_walker_bound.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"{len(rows)} rows; {TABLES / 'm3_walker_bound.csv'}")


if __name__ == "__main__":
    main()
