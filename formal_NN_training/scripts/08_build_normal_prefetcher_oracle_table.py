#!/usr/bin/env python3
"""Build per-access oracle tables from normal-prefetcher residual event logs.

This is the LSTM-ready data bridge.

Input:
  One residual event CSV per trace/prefetcher produced by
  05_run_residual_demand_audit.sh.

Output:
  One oracle CSV per trace. Each row is a demand access from the no-prefetch
  stream with raw demand-stream features plus teacher/oracle labels derived from
  the normal prefetchers.

Important design:
  Normal prefetcher outputs are used as teacher labels/diagnostics only. They are
  not required runtime model inputs. The LSTM can train on raw stream features
  such as pc, line, delta, page, offset, hit/miss, and future target deltas.
"""

import argparse
import csv
import gzip
import lzma
from collections import Counter, deque
from pathlib import Path


DEMAND_EVENTS = {"DEMAND", "DMD", "ACCESS", "LOAD", "RFO"}


def open_text(path, mode="rt"):
    name = str(path)
    if name.endswith(".gz"):
        return gzip.open(path, mode, newline="")
    if name.endswith(".xz"):
        return lzma.open(path, mode, newline="")
    return open(path, mode, newline="")


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


def is_truthy(row, names):
    return to_int(pick(row, names, 0), 0) != 0


def norm_event(row):
    return str(pick(row, ["event", "type", "kind", "event_type"], "")).strip().upper()


def addr_line(row):
    line = pick(row, ["line", "line_addr", "base_line", "addr_line"], "")
    if line != "":
        return to_int(line, 0)
    addr = pick(row, ["addr", "address", "base_addr", "demand_addr"], "")
    if addr == "":
        return 0
    return to_int(addr, 0) // 64


def byte_addr(row):
    addr = pick(row, ["addr", "address", "base_addr", "demand_addr"], "")
    if addr != "":
        return to_int(addr, 0)
    line = addr_line(row)
    return line * 64 if line else 0


def demand_is_covered_on_time(row, hit):
    explicit = is_truthy(row, [
        "covered_on_time",
        "was_prefetch", "was_prefetched", "prefetched",
        "prefetch_hit", "hit_prefetch", "pf_hit",
        "useful_prefetch", "prefetch_useful", "was_useful_prefetch",
        "hit_on_prefetch", "line_prefetched", "prefetch_bit",
    ])
    if explicit and hit:
        return True
    source = str(pick(row, ["hit_source", "source", "fill_source", "line_source"], "")).lower()
    return bool(hit and ("pref" in source or source == "pf"))


def is_demand_row(row):
    ev = norm_event(row)
    return ev in DEMAND_EVENTS or is_truthy(row, ["is_demand", "demand"])


def event_path(event_root, trace, prefetcher, compressed=True):
    suffixes = [".events.csv.gz", ".events.csv", ".events.csv.xz"] if compressed else [".events.csv", ".events.csv.gz", ".events.csv.xz"]
    for suffix in suffixes:
        p = event_root / f"{trace}.{prefetcher}{suffix}"
        if p.exists():
            return p
    return event_root / f"{trace}.{prefetcher}{suffixes[0]}"


def parse_demand_rows(path):
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows, {"missing": 1, "fieldnames": []}

    try:
        with open_text(path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            prev_line = None
            for raw in reader:
                row = {str(k).strip(): v for k, v in raw.items() if k is not None}
                if not is_demand_row(row):
                    continue
                line = addr_line(row)
                addr = byte_addr(row)
                pc = to_int(pick(row, ["pc", "ip"], 0), 0)
                cycle = to_int(pick(row, ["cycle", "timestamp", "time"], 0), 0)
                hit = is_truthy(row, ["hit", "cache_hit", "demand_hit", "l2_hit"])
                miss_field = pick(row, ["miss", "cache_miss", "demand_miss", "l2_miss"], "")
                if miss_field != "":
                    miss = to_int(miss_field, 0) != 0
                else:
                    miss = not hit
                covered = demand_is_covered_on_time(row, hit)
                late = is_truthy(row, ["late_prefetch", "pf_late", "late"])
                delta = 0 if prev_line is None or not line else line - prev_line
                prev_line = line if line else prev_line
                rows.append({
                    "cycle": cycle,
                    "pc": pc,
                    "addr": addr,
                    "line": line,
                    "page": line // 64 if line else 0,
                    "page_offset": line % 64 if line else 0,
                    "hit": int(hit),
                    "miss": int(miss),
                    "covered_on_time": int(covered),
                    "late": int(late),
                    "delta": delta,
                })
        return rows, {"missing": 0, "fieldnames": fieldnames}
    except Exception as e:
        return rows, {"missing": 1, "fieldnames": [], "error": str(e)[:180]}


def compute_future_targets(base_rows, max_lookahead):
    """For each row i, find the next no-prefetch demand miss within max_lookahead rows."""
    n = len(base_rows)
    target_idx = [-1] * n
    q = deque()

    # Walk backwards while keeping future miss indices within the lookahead window.
    for i in range(n - 1, -1, -1):
        while q and q[0] - i > max_lookahead:
            q.popleft()
        if q:
            target_idx[i] = q[0]
        if base_rows[i]["miss"]:
            q.appendleft(i)

    return target_idx


def best_prefetcher_for_index(per_pf_rows, prefetchers, idx):
    for pf in prefetchers:
        rows = per_pf_rows.get(pf, [])
        if idx < len(rows) and rows[idx].get("covered_on_time", 0):
            return pf
    return ""


def summarize_oracle(trace, out_path, base_rows, per_pf_rows, prefetchers, future_idx, mismatch_counts):
    n = len(base_rows)
    covered_any = 0
    residual_all = 0
    future_target_count = 0
    future_target_covered_any = 0
    cover_by_pf = Counter()

    for i, base in enumerate(base_rows):
        any_cov = False
        for pf in prefetchers:
            rows = per_pf_rows.get(pf, [])
            cov = bool(i < len(rows) and rows[i].get("covered_on_time", 0))
            if cov:
                any_cov = True
                cover_by_pf[pf] += 1
        if any_cov:
            covered_any += 1
        if base["miss"] and not any_cov:
            residual_all += 1
        j = future_idx[i]
        if j >= 0:
            future_target_count += 1
            if any(j < len(per_pf_rows.get(pf, [])) and per_pf_rows[pf][j].get("covered_on_time", 0) for pf in prefetchers):
                future_target_covered_any += 1

    return {
        "trace": trace,
        "rows": n,
        "out_file": str(out_path),
        "normal_prefetchers": " ".join(prefetchers),
        "covered_by_any": covered_any,
        "covered_by_any_rate": div(covered_any, n),
        "residual_all_normal": residual_all,
        "residual_all_normal_rate": div(residual_all, n),
        "future_target_count": future_target_count,
        "future_target_covered_any": future_target_covered_any,
        "future_target_covered_any_rate": div(future_target_covered_any, future_target_count),
        "mismatch_counts": dict(mismatch_counts),
        "cover_by_prefetcher": dict(cover_by_pf),
    }


def build_trace(trace, event_root, out_root, prefetchers, max_lookahead, compressed=True):
    base_path = event_path(event_root, trace, "no_pref", compressed)
    base_rows, base_meta = parse_demand_rows(base_path)
    if not base_rows:
        raise RuntimeError(f"no demand rows for {trace} no_pref at {base_path}")

    per_pf_rows = {}
    pf_meta = {}
    for pf in prefetchers:
        p = event_path(event_root, trace, pf, compressed)
        rows, meta = parse_demand_rows(p)
        per_pf_rows[pf] = rows
        pf_meta[pf] = {**meta, "path": str(p), "rows": len(rows)}

    future_idx = compute_future_targets(base_rows, max_lookahead)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{trace}.oracle.csv.gz"

    fixed_fields = [
        "trace", "demand_idx", "cycle", "pc", "addr", "line", "page", "page_offset",
        "delta", "no_pref_hit", "no_pref_miss",
        "covered_by_any_normal", "cover_count", "teacher_prefetcher_class",
        "residual_after_all_normal", "late_by_any_normal",
        "future_target_idx", "future_distance", "future_line", "future_delta",
        "future_pc", "future_covered_by_any_normal", "future_teacher_prefetcher_class",
        "future_residual_after_all_normal",
    ]
    pf_fields = []
    for pf in prefetchers:
        pf_fields.extend([
            f"{pf}_hit", f"{pf}_miss", f"{pf}_covered_on_time", f"{pf}_late", f"{pf}_mismatch",
        ])
    fields = fixed_fields + pf_fields

    mismatch_counts = Counter()

    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, base in enumerate(base_rows):
            cover_count = 0
            late_any = False
            teacher = ""
            row = {
                "trace": trace,
                "demand_idx": i,
                "cycle": base["cycle"],
                "pc": base["pc"],
                "addr": base["addr"],
                "line": base["line"],
                "page": base["page"],
                "page_offset": base["page_offset"],
                "delta": base["delta"],
                "no_pref_hit": base["hit"],
                "no_pref_miss": base["miss"],
            }

            for pf in prefetchers:
                rows = per_pf_rows.get(pf, [])
                if i < len(rows):
                    r = rows[i]
                    mismatch = int((base["line"] and r["line"] and base["line"] != r["line"]) or (base["pc"] and r["pc"] and base["pc"] != r["pc"]))
                    if mismatch:
                        mismatch_counts[pf] += 1
                    cov = int(r.get("covered_on_time", 0))
                    late = int(r.get("late", 0))
                    row[f"{pf}_hit"] = r.get("hit", 0)
                    row[f"{pf}_miss"] = r.get("miss", 0)
                    row[f"{pf}_covered_on_time"] = cov
                    row[f"{pf}_late"] = late
                    row[f"{pf}_mismatch"] = mismatch
                    if cov:
                        cover_count += 1
                        if not teacher:
                            teacher = pf
                    if late:
                        late_any = True
                else:
                    row[f"{pf}_hit"] = ""
                    row[f"{pf}_miss"] = ""
                    row[f"{pf}_covered_on_time"] = 0
                    row[f"{pf}_late"] = 0
                    row[f"{pf}_mismatch"] = 1
                    mismatch_counts[pf] += 1

            row["covered_by_any_normal"] = int(cover_count > 0)
            row["cover_count"] = cover_count
            row["teacher_prefetcher_class"] = teacher
            row["residual_after_all_normal"] = int(base["miss"] and cover_count == 0)
            row["late_by_any_normal"] = int(late_any)

            j = future_idx[i]
            if j >= 0:
                target = base_rows[j]
                future_teacher = best_prefetcher_for_index(per_pf_rows, prefetchers, j)
                row["future_target_idx"] = j
                row["future_distance"] = j - i
                row["future_line"] = target["line"]
                row["future_delta"] = target["line"] - base["line"] if target["line"] and base["line"] else 0
                row["future_pc"] = target["pc"]
                row["future_covered_by_any_normal"] = int(bool(future_teacher))
                row["future_teacher_prefetcher_class"] = future_teacher
                row["future_residual_after_all_normal"] = int(not bool(future_teacher))
            else:
                row["future_target_idx"] = -1
                row["future_distance"] = 0
                row["future_line"] = 0
                row["future_delta"] = 0
                row["future_pc"] = 0
                row["future_covered_by_any_normal"] = 0
                row["future_teacher_prefetcher_class"] = ""
                row["future_residual_after_all_normal"] = 0

            writer.writerow(row)

    summary = summarize_oracle(trace, out_path, base_rows, per_pf_rows, prefetchers, future_idx, mismatch_counts)
    summary["base_event_file"] = str(base_path)
    summary["pf_event_meta"] = str(pf_meta)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--traces", required=True, help="space-separated trace names")
    ap.add_argument("--prefetchers", required=True, help="space-separated working normal prefetchers; exclude no_pref")
    ap.add_argument("--max-lookahead", type=int, default=128)
    ap.add_argument("--compressed", action="store_true")
    args = ap.parse_args()

    prefetchers = [p for p in args.prefetchers.split() if p not in {"no_pref", "none", "nopref"}]
    summaries = []
    for trace in args.traces.split():
        print(f"[oracle] build trace={trace} prefetchers={' '.join(prefetchers)}")
        s = build_trace(trace, args.event_root, args.out_root, prefetchers, args.max_lookahead, args.compressed)
        summaries.append(s)
        print(f"[write] {s['out_file']} rows={s['rows']} covered_any={s['covered_by_any_rate']:.4f} residual_all={s['residual_all_normal_rate']:.4f}")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "rows", "out_file", "normal_prefetchers",
        "covered_by_any", "covered_by_any_rate",
        "residual_all_normal", "residual_all_normal_rate",
        "future_target_count", "future_target_covered_any", "future_target_covered_any_rate",
        "mismatch_counts", "cover_by_prefetcher", "base_event_file", "pf_event_meta",
    ]
    with args.summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    print(f"[write] {args.summary_out}")


if __name__ == "__main__":
    main()
