#!/usr/bin/env python3
"""Build keyed rich lists for explicit replay ceilings.

``omniscient`` schedules each no-prefetch miss exactly N demand events before
its target.  ``bank`` emits only oracle-positive rows already present in a full
candidate ledger.  Both modes write the normal rich-list schema consumed by
07_prepare_keyed_replay_input.py.

This script intentionally uses only Python's standard library so it works on
Sacramento's Python 3.6 installation.  It does not import pandas.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

CACHE_LINE_BYTES = 64


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="")


def as_int(value, field):
    if value is None or str(value).strip() == "":
        raise ValueError("missing {}".format(field))
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def read_oracle(path):
    rows = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = set(["trace", "demand_idx", "pc", "line", "addr", "no_pref_miss"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("oracle missing columns: {}".format(sorted(missing)))
        expected_idx = 0
        for raw in reader:
            idx = as_int(raw.get("demand_idx"), "demand_idx")
            if idx != expected_idx:
                raise ValueError("oracle demand_idx must be contiguous: expected {}, got {}".format(expected_idx, idx))
            expected_idx += 1
            line = as_int(raw.get("line"), "line")
            addr = as_int(raw.get("addr"), "addr")
            if addr <= 0:
                addr = line * CACHE_LINE_BYTES
            if addr % CACHE_LINE_BYTES:
                raise ValueError("unaligned addr at demand_idx={}".format(idx))
            rows.append({
                "trace": str(raw.get("trace") or ""),
                "demand_idx": idx,
                "pc": as_int(raw.get("pc"), "pc"),
                "line": line,
                "addr": addr,
                "no_pref_miss": as_int(raw.get("no_pref_miss"), "no_pref_miss"),
                "cycle": as_int(raw.get("cycle"), "cycle") if raw.get("cycle") not in (None, "") else 0,
            })
    if not rows:
        raise ValueError("oracle has no rows: {}".format(path))
    traces = set(row["trace"] for row in rows)
    if len(traces) != 1 or not next(iter(traces)):
        raise ValueError("oracle must contain one nonempty trace, got {}".format(traces))
    return rows


def make_rich_row(trace, trigger, target_line, target_addr, rank, source, lead_note, future_label, future_cycle_label):
    return {
        "trace": trace,
        "order": int(trigger.get("cycle", trigger["demand_idx"])),
        "demand_idx": int(trigger["demand_idx"]),
        "pc": "0x{:x}".format(int(trigger["pc"])),
        "line": int(trigger["line"]),
        "candidate_rank": int(rank),
        "candidate_delta": int(target_line) - int(trigger["line"]),
        "candidate_source": source,
        "utility_prob": 1.0,
        "far_prob": 1.0,
        "issue_prob": 1.0,
        "predicted_lead_lo": lead_note,
        "predicted_cycle_lo": 0,
        "candidate_score": 1.0,
        "prefetch_addr": "0x{:x}".format(int(target_addr)),
        "future_label": int(future_label),
        "future_cycle_label": int(future_cycle_label),
        "ceiling_source": source,
    }


def write_rich_list(path, rows):
    if not rows:
        raise RuntimeError("ceiling construction emitted zero legal rows")
    fields = [
        "trace", "order", "demand_idx", "pc", "line", "candidate_rank",
        "candidate_delta", "candidate_source", "utility_prob", "far_prob",
        "issue_prob", "predicted_lead_lo", "predicted_cycle_lo", "candidate_score",
        "prefetch_addr", "future_label", "future_cycle_label", "ceiling_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_omniscient(oracle, min_lead_events, max_degree):
    if min_lead_events <= 0:
        raise ValueError("min-lead-events must be positive")
    trace = oracle[0]["trace"]
    by_trigger = defaultdict(list)
    no_pref_misses = 0
    skipped_early = 0
    skipped_same_line = 0
    for target in oracle:
        if not target["no_pref_miss"]:
            continue
        no_pref_misses += 1
        trigger_idx = target["demand_idx"] - min_lead_events
        if trigger_idx < 0:
            skipped_early += 1
            continue
        trigger = oracle[trigger_idx]
        if trigger["line"] == target["line"]:
            skipped_same_line += 1
            continue
        by_trigger[trigger_idx].append(target)

    rows = []
    dropped_by_degree = 0
    for trigger_idx in sorted(by_trigger):
        trigger = oracle[trigger_idx]
        seen_lines = set()
        rank = 0
        for target in by_trigger[trigger_idx]:
            if target["line"] in seen_lines:
                continue
            seen_lines.add(target["line"])
            if rank >= max_degree:
                dropped_by_degree += 1
                continue
            rows.append(make_rich_row(
                trace, trigger, target["line"], target["addr"], rank,
                "oracle_future_no_pref_miss", "events_{}".format(min_lead_events), 1, 0
            ))
            rank += 1
    return rows, {
        "mode": "omniscient",
        "trace": trace,
        "oracle_rows": len(oracle),
        "no_pref_miss_targets": no_pref_misses,
        "min_lead_events": min_lead_events,
        "max_degree": max_degree,
        "scheduled_rows": len(rows),
        "scheduled_target_fraction": float(len(rows)) / no_pref_misses if no_pref_misses else 0.0,
        "skipped_too_early": skipped_early,
        "skipped_same_trigger_line": skipped_same_line,
        "dropped_by_degree": dropped_by_degree,
        "semantics": "offline omniscient replay ceiling; not an online NN policy",
    }


def build_bank(oracle, candidate_ledger, min_lead_bin, max_degree, require_full_coverage):
    trace = oracle[0]["trace"]
    by_idx = dict((row["demand_idx"], row) for row in oracle)
    by_identity = dict((
        (row["demand_idx"], row["pc"], row["line"]), row
    ) for row in oracle)
    by_trigger = defaultdict(list)
    ledger_events = set()
    unmatched = 0

    with open_text(candidate_ledger) as handle:
        reader = csv.DictReader(handle)
        required = set([
            "trace", "demand_idx", "pc", "line", "candidate_line", "candidate_valid",
            "future_label", "candidate_score", "candidate_rank",
        ])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("candidate ledger missing columns: {}".format(sorted(missing)))
        for raw in reader:
            if str(raw.get("trace") or "") != trace:
                raise ValueError("candidate ledger trace mismatch: {}".format(raw.get("trace")))
            if not as_int(raw.get("candidate_valid"), "candidate_valid"):
                continue
            future_label = as_int(raw.get("future_label"), "future_label")
            if future_label < min_lead_bin:
                continue
            demand_idx = as_int(raw.get("demand_idx"), "demand_idx")
            pc = as_int(raw.get("pc"), "pc")
            line = as_int(raw.get("line"), "line")
            trigger = by_identity.get((demand_idx, pc, line))
            if trigger is None:
                unmatched += 1
                continue
            ledger_events.add(demand_idx)
            target_line = as_int(raw.get("candidate_line"), "candidate_line")
            if target_line == line:
                continue
            by_trigger[demand_idx].append({
                "line": target_line,
                "rank": as_int(raw.get("candidate_rank"), "candidate_rank"),
                "score": float(raw.get("candidate_score") or 0.0),
                "future_label": future_label,
                "future_cycle_label": as_int(raw.get("future_cycle_label"), "future_cycle_label") if raw.get("future_cycle_label") not in (None, "") else 0,
            })

    if require_full_coverage and len(ledger_events) != len(oracle):
        raise RuntimeError(
            "ledger covers {}/{} oracle events; run the notebook with LEDGER_SCOPE=full and LEDGER_CSV=full".format(
                len(ledger_events), len(oracle)
            )
        )

    rows = []
    dropped_by_degree = 0
    for demand_idx in sorted(by_trigger):
        trigger = by_idx[demand_idx]
        candidates = sorted(by_trigger[demand_idx], key=lambda c: (c["rank"], -c["score"], c["line"]))
        seen_lines = set()
        output_rank = 0
        for cand in candidates:
            if cand["line"] in seen_lines:
                continue
            seen_lines.add(cand["line"])
            if output_rank >= max_degree:
                dropped_by_degree += 1
                continue
            rows.append(make_rich_row(
                trace, trigger, cand["line"], cand["line"] * CACHE_LINE_BYTES,
                output_rank, "oracle_positive_fixed_candidate_bank",
                "ledger_bin_{}".format(cand["future_label"]),
                cand["future_label"], cand["future_cycle_label"]
            ))
            output_rank += 1
    return rows, {
        "mode": "bank",
        "trace": trace,
        "candidate_ledger": str(candidate_ledger),
        "oracle_rows": len(oracle),
        "ledger_events_seen": len(ledger_events),
        "ledger_full_scope_verified": len(ledger_events) == len(oracle),
        "min_lead_bin": min_lead_bin,
        "max_degree": max_degree,
        "scheduled_rows": len(rows),
        "dropped_by_degree": dropped_by_degree,
        "unmatched_ledger_rows": unmatched,
        "semantics": "oracle-positive fixed-candidate-bank replay ceiling; not an online NN policy",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--mode", choices=("omniscient", "bank"), required=True)
    ap.add_argument("--min-lead-events", type=int, default=8)
    ap.add_argument("--min-lead-bin", type=int, default=1)
    ap.add_argument("--max-degree", type=int, default=1)
    ap.add_argument("--ledger-candidates", type=Path, default=None)
    ap.add_argument("--require-full-coverage", action="store_true")
    args = ap.parse_args()

    if args.max_degree <= 0:
        raise SystemExit("max-degree must be positive")
    oracle = read_oracle(args.oracle)
    if args.mode == "omniscient":
        rows, metadata = build_omniscient(oracle, args.min_lead_events, args.max_degree)
    else:
        if args.ledger_candidates is None:
            raise SystemExit("--ledger-candidates is required for bank mode")
        rows, metadata = build_bank(
            oracle, args.ledger_candidates, args.min_lead_bin,
            args.max_degree, args.require_full_coverage
        )
    write_rich_list(args.out, rows)
    metadata["oracle"] = str(args.oracle)
    metadata["out"] = str(args.out)
    metadata["rows_written"] = len(rows)
    metadata_path = args.meta_out or args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[ceiling list] " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
