#!/usr/bin/env python3
"""Parse demand-centric residual audit event CSVs.

No pandas. Supports plain .csv and .csv.gz files.

Key fix compared with the older 19_* parser:
  covered_on_time is counted on demand HIT rows that carry a prefetch-use flag.
  It is not counted only inside demand misses.

This matches the intended accounting:
  original_miss_pool ~= covered_on_time + demand_miss
  coverage_among_misses = covered_on_time / original_miss_pool
  residual_share_of_misses = demand_miss / original_miss_pool
"""

import argparse
import csv
import gzip
import lzma
from collections import Counter
from pathlib import Path


def open_text(path):
    name = str(path)
    if name.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    if name.endswith(".xz"):
        return lzma.open(path, "rt", newline="")
    return open(path, "r", newline="")


def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        s = str(x).strip()
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(float(s))
    except Exception:
        return default


def div(a, b):
    return float(a) / float(b) if b else 0.0


def pick(row, names, default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def norm_event(row):
    return str(pick(row, ["event", "type", "kind", "event_type"], "")).strip().upper()


def is_truthy(row, names):
    return to_int(pick(row, names, 0), 0) != 0


def addr_line(row, addr_names, line_names):
    line = pick(row, line_names, "")
    if line != "":
        return to_int(line, 0)
    addr = pick(row, addr_names, "")
    if addr == "":
        return 0
    return to_int(addr, 0) // 64


def demand_is_covered_on_time(row, hit):
    """Return True when this demand access hit because of a prior prefetch.

    Pythia/ChampSim variants use different names. In normal ChampSim stats,
    a useful prefetch is recognized on the later demand hit, not on the demand
    miss. Our residual logger writes this signal as `was_prefetch`, so that
    alias must be checked here.
    """
    explicit = is_truthy(row, [
        "covered_on_time",
        "was_prefetch", "was_prefetched", "prefetched",
        "prefetch_hit", "hit_prefetch", "pf_hit",
        "useful_prefetch", "prefetch_useful", "was_useful_prefetch",
        "hit_on_prefetch", "line_prefetched", "prefetch_bit",
    ])
    if explicit and hit:
        return True

    # Some loggers put the access type/source instead of a boolean.
    source = str(pick(row, ["hit_source", "source", "fill_source", "line_source"], "")).lower()
    if hit and ("pref" in source or source == "pf"):
        return True

    return False


def parse_one(path):
    out = {
        "demand": 0,
        "demand_hit": 0,
        "demand_miss": 0,
        "covered_on_time": 0,
        "late_prefetch": 0,
        "residual_miss": 0,
        "pf_requested_events": 0,
        "pf_accepted_events": 0,
        "pf_duplicate_events": 0,
        "pf_dropped_events": 0,
        "parse_error": 0,
        "top_residual_pcs": [],
        "top_residual_deltas": [],
    }

    if not path.exists() or path.stat().st_size == 0:
        out["parse_error"] = 1
        return out

    top_pc = Counter()
    top_delta = Counter()

    try:
        with open_text(path) as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                out["parse_error"] = 1
                return out

            for raw in reader:
                row = {str(k).strip(): v for k, v in raw.items() if k is not None}
                ev = norm_event(row)

                if ev in {"DEMAND", "DMD", "ACCESS", "LOAD", "RFO"} or is_truthy(row, ["is_demand", "demand"]):
                    out["demand"] += 1
                    hit = is_truthy(row, ["hit", "cache_hit", "demand_hit", "l2_hit"])
                    late = is_truthy(row, ["late_prefetch", "pf_late", "late"])
                    covered = demand_is_covered_on_time(row, hit)

                    miss_field = pick(row, ["miss", "cache_miss", "demand_miss", "l2_miss"], "")
                    if miss_field != "":
                        miss = to_int(miss_field, 0) != 0
                    else:
                        miss = not hit

                    if hit:
                        out["demand_hit"] += 1
                    if covered:
                        out["covered_on_time"] += 1

                    if miss:
                        out["demand_miss"] += 1
                        if late:
                            out["late_prefetch"] += 1

                        # A miss was not covered on time. Keep it in the residual
                        # target pool. Late misses can become a separate timing label.
                        out["residual_miss"] += 1
                        pc = str(pick(row, ["pc", "ip"], ""))
                        if pc:
                            top_pc[pc] += 1
                        delta = pick(row, ["delta", "demand_delta", "line_delta"], "")
                        if delta == "":
                            cur = addr_line(row, ["addr", "address"], ["line", "line_addr"])
                            prev = addr_line(row, ["prev_addr", "last_addr"], ["prev_line", "last_line"])
                            if cur and prev:
                                delta = cur - prev
                        if delta != "":
                            top_delta[to_int(delta, 0)] += 1

                elif ev in {"PF", "PREFETCH", "PREF", "PF_REQUEST", "PREFETCH_REQUEST"} or is_truthy(row, ["is_prefetch", "prefetch"]):
                    out["pf_requested_events"] += 1
                    accepted = is_truthy(row, ["accepted", "issued", "pf_accepted", "pf_issued", "enqueued"])
                    duplicate = is_truthy(row, ["duplicate", "pf_duplicate", "already_present", "merged", "pq_merged"])
                    dropped = is_truthy(row, ["dropped", "rejected", "pf_dropped"])
                    if accepted:
                        out["pf_accepted_events"] += 1
                    if duplicate:
                        out["pf_duplicate_events"] += 1
                    if dropped:
                        out["pf_dropped_events"] += 1

                elif any(k in row for k in ["pf_addr", "prefetch_addr", "pf_line"]):
                    out["pf_requested_events"] += 1
                    if is_truthy(row, ["accepted", "issued", "pf_accepted", "pf_issued", "enqueued"]):
                        out["pf_accepted_events"] += 1
                    if is_truthy(row, ["duplicate", "pf_duplicate", "already_present", "merged", "pq_merged"]):
                        out["pf_duplicate_events"] += 1

    except Exception:
        out["parse_error"] = 1

    out["top_residual_pcs"] = top_pc.most_common(10)
    out["top_residual_deltas"] = top_delta.most_common(10)
    return out


def summarize(trace, prefetcher, event_root, compressed):
    suffixes = [".events.csv.gz", ".events.csv"] if compressed else [".events.csv", ".events.csv.gz"]
    path = None
    for suffix in suffixes:
        p = event_root / (trace + "." + prefetcher + suffix)
        if p.exists():
            path = p
            break
    if path is None:
        path = event_root / (trace + "." + prefetcher + suffixes[0])

    s = parse_one(path)
    demand = s["demand"]
    demand_miss = s["demand_miss"]
    covered = s["covered_on_time"]
    late = s["late_prefetch"]
    residual = s["residual_miss"]
    pf_req = s["pf_requested_events"]
    pf_dup = s["pf_duplicate_events"]

    original_miss_pool = demand_miss + covered

    return {
        "trace": trace,
        "prefetcher": prefetcher,
        "demand": demand,
        "demand_hit": s["demand_hit"],
        "demand_miss": demand_miss,
        "demand_miss_rate": div(demand_miss, demand),
        "covered_on_time": covered,
        "covered_on_time_rate": div(covered, demand),
        "coverage_among_misses": div(covered, original_miss_pool),
        "late_prefetch": late,
        "late_rate_among_misses": div(late, demand_miss),
        "residual_miss": residual,
        "residual_miss_rate": div(residual, demand),
        "residual_share_of_misses": div(residual, original_miss_pool),
        "original_miss_pool": original_miss_pool,
        "pf_requested_events": pf_req,
        "pf_accepted_events": s["pf_accepted_events"],
        "pf_duplicate_events": pf_dup,
        "pf_duplicate_rate": div(pf_dup, pf_req),
        "pf_dropped_events": s["pf_dropped_events"],
        "pf_dropped_rate": div(s["pf_dropped_events"], pf_req),
        "top_residual_pcs": str(s["top_residual_pcs"]),
        "top_residual_deltas": str(s["top_residual_deltas"]),
        "parse_error": s["parse_error"],
        "event_file": str(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--prefetchers", required=True)
    ap.add_argument("--compressed", action="store_true")
    args = ap.parse_args()

    rows = []
    for trace in args.traces.split():
        for pf in args.prefetchers.split():
            rows.append(summarize(trace, pf, args.event_root, args.compressed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "prefetcher", "demand", "demand_hit", "demand_miss", "demand_miss_rate",
        "covered_on_time", "covered_on_time_rate", "coverage_among_misses",
        "late_prefetch", "late_rate_among_misses",
        "residual_miss", "residual_miss_rate", "residual_share_of_misses",
        "original_miss_pool",
        "pf_requested_events", "pf_accepted_events", "pf_duplicate_events", "pf_duplicate_rate",
        "pf_dropped_events", "pf_dropped_rate",
        "top_residual_pcs", "top_residual_deltas", "parse_error", "event_file",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[write] {args.out}")
    for row in rows:
        print(
            "[residual] {trace} {prefetcher} miss_rate={demand_miss_rate:.4f} "
            "covered={coverage_among_misses:.4f} late={late_rate_among_misses:.4f} "
            "residual_share={residual_share_of_misses:.4f} pf_dup={pf_duplicate_rate:.4f}".format(**row)
        )


if __name__ == "__main__":
    main()
