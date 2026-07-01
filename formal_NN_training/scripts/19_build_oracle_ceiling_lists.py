#!/usr/bin/env python3
"""Build keyed rich lists for explicit prefetch ceilings.

Two ceilings are intentionally separate:

* ``omniscient``: each no-prefetch miss is scheduled exactly N demand events
  earlier.  It answers whether the *keyed-list replay mechanism plus cache and
  queue configuration* can in principle beat a normal prefetcher.
* ``bank``: selects oracle-positive rows from a complete decision-ledger
  candidate table.  It answers whether the fixed candidate bank has enough
  reachable targets before model/policy errors are considered.

Neither output is a deployable prefetcher.  The rich-list schema is compatible
with 07_prepare_keyed_replay_input.py and therefore with the normal keyed
ListReplayer replay driver.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CACHE_LINE_BYTES = 64


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return path.open(mode.replace("t", ""), newline="")


def as_int(value, name: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {name}")
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def read_oracle(path: Path) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"trace", "demand_idx", "pc", "line", "addr", "no_pref_miss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"oracle missing columns: {sorted(missing)}")
        expected_idx = 0
        for raw in reader:
            idx = as_int(raw.get("demand_idx"), "demand_idx")
            if idx != expected_idx:
                raise ValueError(f"oracle demand_idx must be contiguous from zero; expected {expected_idx}, got {idx}")
            expected_idx += 1
            line = as_int(raw.get("line"), "line")
            addr = as_int(raw.get("addr"), "addr")
            if addr <= 0:
                addr = line * CACHE_LINE_BYTES
            if addr % CACHE_LINE_BYTES:
                raise ValueError(f"unaligned address at demand_idx={idx}: {addr}")
            rows.append({
                "trace": str(raw.get("trace") or ""),
                "demand_idx": idx,
                "pc": as_int(raw.get("pc"), "pc"),
                "line": line,
                "addr": addr,
                "no_pref_miss": as_int(raw.get("no_pref_miss"), "no_pref_miss"),
                "cycle": as_int(raw.get("cycle", 0), "cycle") if raw.get("cycle", "") != "" else 0,
            })
    if not rows:
        raise ValueError(f"oracle has no rows: {path}")
    trace_names = {row["trace"] for row in rows}
    if len(trace_names) != 1 or not next(iter(trace_names)):
        raise ValueError(f"oracle must contain exactly one nonempty trace name, got {trace_names}")
    return rows


def rich_row(trace: str, trigger: Dict[str, int], target_line: int, target_addr: int,
             rank: int, source: str, note: str, *, future_label: int = 1,
             future_cycle_label: int = 0) -> Dict[str, object]:
    delta = int(target_line) - int(trigger["line"])
    return {
        "trace": trace,
        "order": int(trigger.get("cycle", trigger["demand_idx"])),
        "demand_idx": int(trigger["demand_idx"]),
        "pc": f"0x{int(trigger['pc']):x}",
        "line": int(trigger["line"]),
        "candidate_rank": int(rank),
        "candidate_delta": int(delta),
        "candidate_source": source,
        "utility_prob": 1.0,
        "far_prob": 1.0,
        "issue_prob": 1.0,
        "predicted_lead_lo": note,
        "predicted_cycle_lo": 0,
        "candidate_score": 1.0,
        "prefetch_addr": f"0x{int(target_addr):x}",
        "future_label": int(future_label),
        "future_cycle_label": int(future_cycle_label),
        "ceiling_source": source,
    }


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        raise RuntimeError("ceiling construction emitted zero legal rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "order", "demand_idx", "pc", "line", "candidate_rank",
        "candidate_delta", "candidate_source", "utility_prob", "far_prob",
        "issue_prob", "predicted_lead_lo", "predicted_cycle_lo", "candidate_score",
        "prefetch_addr", "future_label", "future_cycle_label", "ceiling_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_omniscient(oracle: List[Dict[str, int]], min_lead_events: int,
                      max_degree: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if min_lead_events <= 0:
        raise ValueError("--min-lead-events must be positive")
    trace = oracle[0]["trace"]
    grouped: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    target_misses = 0
    skipped_early = 0
    skipped_same_line = 0
    for target in oracle:
        if not target["no_pref_miss"]:
            continue
        target_misses += 1
        trigger_idx = target["demand_idx"] - min_lead_events
        if trigger_idx < 0:
            skipped_early += 1
            continue
        trigger = oracle[trigger_idx]
        if trigger["line"] == target["line"]:
            skipped_same_line += 1
            continue
        grouped[trigger_idx].append(target)

    rows: List[Dict[str, object]] = []
    dropped_degree = 0
    for trigger_idx in sorted(grouped):
        trigger = oracle[trigger_idx]
        seen_lines = set()
        rank = 0
        for target in grouped[trigger_idx]:
            if target["line"] in seen_lines:
                continue
            seen_lines.add(target["line"])
            if rank >= max_degree:
                dropped_degree += 1
                continue
            rows.append(rich_row(
                trace, trigger, target["line"], target["addr"], rank,
                "oracle_future_no_pref_miss", f"events_{min_lead_events}",
            ))
            rank += 1
    meta = {
        "mode": "omniscient",
        "trace": trace,
        "oracle_rows": len(oracle),
        "no_pref_miss_targets": target_misses,
        "min_lead_events": min_lead_events,
        "max_degree": max_degree,
        "scheduled_rows": len(rows),
        "scheduled_target_fraction": len(rows) / target_misses if target_misses else 0.0,
        "skipped_too_early": skipped_early,
        "skipped_same_trigger_line": skipped_same_line,
        "dropped_by_degree": dropped_degree,
        "semantics": (
            "Offline omniscient target list. Each retained no-prefetch miss is emitted exactly "
            "min_lead_events earlier. This is a mechanism ceiling, not an online predictor."
        ),
    }
    return rows, meta


def parse_ledger_candidates(path: Path, oracle: List[Dict[str, int]], min_lead_bin: int,
                            max_degree: int, require_full_coverage: bool) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    trace = oracle[0]["trace"]
    oracle_by_key = {(row["pc"], row["line"], row["demand_idx"]): row for row in oracle}
    oracle_by_idx = {row["demand_idx"]: row for row in oracle}
    grouped: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    seen_events = set()
    unmatched_ledger_events = 0
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "pc", "line", "candidate_line", "candidate_valid",
            "future_label", "candidate_score", "candidate_rank",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ledger candidate CSV missing columns: {sorted(missing)}")
        for raw in reader:
            if str(raw.get("trace") or "") != trace:
                raise ValueError(f"ledger trace mismatch: expected {trace}, got {raw.get('trace')}")
            if as_int(raw.get("candidate_valid"), "candidate_valid") == 0:
                continue
            lead_bin = as_int(raw.get("future_label"), "future_label")
            if lead_bin < min_lead_bin:
                continue
            demand_idx = as_int(raw.get("demand_idx"), "demand_idx")
            pc = as_int(raw.get("pc"), "pc")
            line = as_int(raw.get("line"), "line")
            trigger = oracle_by_key.get((pc, line, demand_idx)) or oracle_by_idx.get(demand_idx)
            if trigger is None or trigger["pc"] != pc or trigger["line"] != line:
                unmatched_ledger_events += 1
                continue
            seen_events.add(demand_idx)
            target_line = as_int(raw.get("candidate_line"), "candidate_line")
            if target_line == line:
                continue
            grouped[demand_idx].append({
                "target_line": target_line,
                "candidate_rank": as_int(raw.get("candidate_rank"), "candidate_rank"),
                "candidate_score": float(raw.get("candidate_score") or 0.0),
                "future_label": lead_bin,
                "future_cycle_label": as_int(raw.get("future_cycle_label", 0), "future_cycle_label") if raw.get("future_cycle_label", "") != "" else 0,
            })

    if require_full_coverage:
        missing = sorted(set(oracle_by_idx).difference(seen_events))
        if missing:
            raise RuntimeError(
                f"ledger is not full-scope: it covers {len(seen_events)}/{len(oracle_by_idx)} oracle events; "
                "rerun the notebook with LEDGER_SCOPE=all before using --require-full-coverage"
            )

    rows: List[Dict[str, object]] = []
    dropped_degree = 0
    for demand_idx, candidates in sorted(grouped.items()):
        trigger = oracle_by_idx[demand_idx]
        candidates.sort(key=lambda x: (x["candidate_rank"], -x["candidate_score"], x["target_line"]))
        seen_lines = set()
        rank = 0
        for cand in candidates:
            line = cand["target_line"]
            if line in seen_lines:
                continue
            seen_lines.add(line)
            if rank >= max_degree:
                dropped_degree += 1
                continue
            rows.append(rich_row(
                trace, trigger, line, line * CACHE_LINE_BYTES, rank,
                "oracle_positive_fixed_candidate_bank", f"ledger_bin_{cand['future_label']}",
                future_label=cand["future_label"], future_cycle_label=cand["future_cycle_label"],
            ))
            rank += 1
    meta = {
        "mode": "bank",
        "trace": trace,
        "ledger_candidates": str(path),
        "ledger_events_seen": len(seen_events),
        "oracle_rows": len(oracle),
        "min_lead_bin": min_lead_bin,
        "max_degree": max_degree,
        "scheduled_rows": len(rows),
        "dropped_by_degree": dropped_degree,
        "unmatched_ledger_events": unmatched_ledger_events,
        "ledger_full_scope_verified": bool(len(seen_events) == len(oracle)),
        "semantics": (
            "Oracle-positive candidates from the learned fixed candidate bank. Model score and policy "
            "are bypassed; candidate representation, keyed replay, queues, and cache remain active."
        ),
    }
    return rows, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=("omniscient", "bank"), required=True)
    ap.add_argument("--min-lead-events", type=int, default=8,
                    help="Only omniscient mode: exact demand-event separation before target miss.")
    ap.add_argument("--min-lead-bin", type=int, default=1,
                    help="Only bank mode: keep oracle-positive candidates with future_label >= this bin.")
    ap.add_argument("--max-degree", type=int, default=1)
    ap.add_argument("--ledger-candidates", type=Path, default=None)
    ap.add_argument("--require-full-coverage", action="store_true")
    ap.add_argument("--meta-out", type=Path, default=None)
    args = ap.parse_args()
    if args.max_degree <= 0:
        raise SystemExit("--max-degree must be positive")
    oracle = read_oracle(args.oracle)
    if args.mode == "omniscient":
        rows, meta = build_omniscient(oracle, args.min_lead_events, args.max_degree)
    else:
        if args.ledger_candidates is None:
            raise SystemExit("--ledger-candidates is required in bank mode")
        rows, meta = parse_ledger_candidates(
            args.ledger_candidates, oracle, args.min_lead_bin, args.max_degree,
            args.require_full_coverage,
        )
    count = write_rows(args.out, rows)
    meta.update({"out": str(args.out), "rows_written": count, "oracle": str(args.oracle)})
    meta_out = args.meta_out or args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("[ceiling list] " + json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
