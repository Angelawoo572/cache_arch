#!/usr/bin/env python3
"""Parse Pythia/ChampSim logs into a prefetch behavior audit table.

No pandas. This parser is for the counter-level behavior audit.
"""

import argparse
import csv
import re
from pathlib import Path

KEY_VAL_RE = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED_RE = re.compile(r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)")


def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def div(a, b):
    a = float(a)
    b = float(b)
    return a / b if b else 0.0


def parse_log(path):
    stats = {}
    if not path.exists() or path.stat().st_size == 0:
        stats["log_missing"] = 1.0
        return stats

    with path.open(errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = KEY_VAL_RE.match(line)
            if m:
                stats[m.group(1)] = to_float(m.group(2))
                continue
            m = FINISHED_RE.search(line)
            if m:
                stats["finished_instructions"] = to_float(m.group(1))
                stats["finished_cycles"] = to_float(m.group(2))
                stats["finished_ipc"] = to_float(m.group(3))
    return stats


def get(stats, key):
    return stats.get(key, 0.0)


def summarize_one(trace, prefetcher, log, nodup):
    s = parse_log(log)

    ipc = get(s, "Core_0_IPC") or get(s, "finished_ipc")
    cycles = get(s, "Core_0_cycles") or get(s, "finished_cycles")
    inst = get(s, "Core_0_instructions") or get(s, "finished_instructions")

    l2_loads = get(s, "Core_0_L2C_loads")
    l2_load_miss = get(s, "Core_0_L2C_load_miss")
    l2_load_hit = get(s, "Core_0_L2C_load_hit")

    requested = get(s, "Core_0_L2C_prefetch_requested")
    dropped = get(s, "Core_0_L2C_prefetch_dropped")
    issued = get(s, "Core_0_L2C_prefetch_issued")
    filled = get(s, "Core_0_L2C_prefetch_filled")
    useful = get(s, "Core_0_L2C_prefetch_useful")
    useless = get(s, "Core_0_L2C_prefetch_useless")
    late = get(s, "Core_0_L2C_prefetch_late")
    pq_merged = get(s, "Core_0_L2C_pq_merged")

    nodup_issued = max(issued - pq_merged, 0.0)
    denom_issued = nodup_issued if nodup else issued

    return {
        "trace": trace,
        "prefetcher": prefetcher,
        "log": str(log),
        "log_missing": int(get(s, "log_missing")),
        "ipc": ipc,
        "instructions": int(inst),
        "cycles": int(cycles),
        "l2_loads": int(l2_loads),
        "l2_load_hit": int(l2_load_hit),
        "l2_load_miss": int(l2_load_miss),
        "l2_load_miss_rate": div(l2_load_miss, l2_loads),
        "pf_requested": int(requested),
        "pf_dropped": int(dropped),
        "pf_issued": int(issued),
        "pf_filled": int(filled),
        "pf_useful": int(useful),
        "pf_useless": int(useless),
        "pf_late": int(late),
        "pq_merged_duplicate_proxy": int(pq_merged),
        "nodup_issued": int(nodup_issued),
        "accuracy": div(useful, issued),
        "nodup_accuracy": div(useful, nodup_issued),
        "selected_accuracy": div(useful, denom_issued),
        "timeliness": div(useful, useful + late),
        "late_per_issued": div(late, issued),
        "drop_rate": div(dropped, requested),
        "useless_per_issued": div(useless, issued),
        "useful_per_l2_miss_self": div(useful, l2_load_miss),
        "nodup_mode": int(bool(nodup)),
    }


def add_baseline_metrics(rows):
    by_trace = {}
    for r in rows:
        if r["prefetcher"] in {"no_pref", "none", "nopref"}:
            by_trace[str(r["trace"])] = r

    for r in rows:
        base = by_trace.get(str(r["trace"]))
        if not base:
            r["speedup_vs_no_pref"] = 0.0
            r["coverage_vs_no_pref_l2_miss"] = 0.0
            r["miss_reduction_vs_no_pref"] = 0.0
            continue
        base_ipc = to_float(base.get("ipc"))
        base_miss = to_float(base.get("l2_load_miss"))
        miss = to_float(r.get("l2_load_miss"))
        useful = to_float(r.get("pf_useful"))
        r["speedup_vs_no_pref"] = div(to_float(r.get("ipc")), base_ipc)
        r["coverage_vs_no_pref_l2_miss"] = div(useful, base_miss)
        r["miss_reduction_vs_no_pref"] = div(base_miss - miss, base_miss)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = [
        "trace", "prefetcher", "ipc", "speedup_vs_no_pref",
        "l2_loads", "l2_load_miss", "l2_load_miss_rate", "miss_reduction_vs_no_pref",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued", "pf_filled",
        "pf_useful", "pf_useless", "pf_late", "pq_merged_duplicate_proxy",
        "accuracy", "nodup_accuracy", "selected_accuracy",
        "coverage_vs_no_pref_l2_miss", "useful_per_l2_miss_self",
        "timeliness", "late_per_issued", "drop_rate", "useless_per_issued",
        "nodup_mode", "log_missing", "log",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--traces", required=True, help="space-separated trace names")
    ap.add_argument("--prefetchers", required=True, help="space-separated prefetcher tags")
    ap.add_argument("--nodup", action="store_true", help="use issued - pq_merged as selected accuracy denominator")
    args = ap.parse_args()

    rows = []
    for trace in args.traces.split():
        for pf in args.prefetchers.split():
            log = args.log_root / (trace + "." + pf + ".log")
            rows.append(summarize_one(trace, pf, log, args.nodup))

    add_baseline_metrics(rows)
    write_csv(args.out, rows)

    print("[write] {}".format(args.out))
    for r in rows:
        print(
            "[audit] {trace} {prefetcher} IPC={ipc:.6f} speedup={speedup:.4f} "
            "acc={acc:.4f} nodup_acc={nodup_acc:.4f} coverage={coverage:.4f} "
            "timeliness={timeliness:.4f}".format(
                trace=r["trace"],
                prefetcher=r["prefetcher"],
                ipc=to_float(r["ipc"]),
                speedup=to_float(r.get("speedup_vs_no_pref")),
                acc=to_float(r["accuracy"]),
                nodup_acc=to_float(r["nodup_accuracy"]),
                coverage=to_float(r.get("coverage_vs_no_pref_l2_miss")),
                timeliness=to_float(r["timeliness"]),
            )
        )


if __name__ == "__main__":
    main()
