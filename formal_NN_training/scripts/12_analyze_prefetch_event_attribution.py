#!/usr/bin/env python3
"""Compare normal and standalone prefetchers at L2C demand-event granularity.

Normal results are analysis references only. Existing standalone rich exports
contain selected entries, not every rejected candidate. Therefore
no_earlier_selected_standalone_export means only that the target was absent
from the selected frozen list; it does not prove candidate-bank absence.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("r", newline="")


def to_int(value, default=None):
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("missing integer")
        return default
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def write_rows(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_oracle(path: Path):
    by_key = {}
    rows = []
    with open_text(path) as f:
        for raw in csv.DictReader(f):
            row = {
                "demand_idx": to_int(raw["demand_idx"]),
                "pc": to_int(raw["pc"]),
                "line": to_int(raw["line"]),
                "addr": to_int(raw["addr"]),
                "delta": to_int(raw.get("delta"), 0),
                "page_offset": to_int(raw.get("page_offset"), 0),
                "no_pref_miss": to_int(raw.get("no_pref_miss"), 0),
            }
            key = (row["pc"], row["line"], to_int(raw["pc_line_occ"]))
            by_key[key] = row
            rows.append(row)
    return by_key, rows


def load_l2_events(path: Path, oracle_by_key):
    occurrence = Counter()
    events = {}
    unmatched = 0
    with open_text(path) as f:
        for raw in csv.DictReader(f):
            if raw.get("event", "").upper() != "DEMAND":
                continue
            if raw.get("cache", "").upper() != "L2C":
                continue
            if str(raw.get("type", "")).strip() not in {"0", "0.0"}:
                continue
            pc = to_int(raw.get("ip"), 0)
            line = to_int(raw.get("line"), 0)
            key = (pc, line, occurrence[(pc, line)])
            occurrence[(pc, line)] += 1
            if key not in oracle_by_key:
                unmatched += 1
                continue
            events[key] = {
                "hit": to_int(raw.get("hit"), 0),
                "was_prefetch": to_int(raw.get("was_prefetch"), 0),
                "late": to_int(raw.get("late"), 0),
                "pq_occ": to_int(raw.get("pq_occ"), 0),
                "pq_size": to_int(raw.get("pq_size"), 0),
                "mshr_occ": to_int(raw.get("mshr_occ"), 0),
                "mshr_size": to_int(raw.get("mshr_size"), 0),
            }
    return events, unmatched


def outcome(no_pref_row, event):
    if not no_pref_row["no_pref_miss"]:
        return "not_no_pref_miss"
    if event is None:
        return "unmatched"
    if event["hit"] and event["was_prefetch"]:
        return "timely"
    if event["late"]:
        return "late"
    if event["hit"]:
        return "hit_not_marked_prefetch"
    return "residual"


def find_rich_list(artifact_dir: Path, trace: str, chunk_len: int, suffix: str) -> Path:
    exact = artifact_dir / f"prefetch_list_{trace}_cl{chunk_len}_{suffix}.csv"
    if exact.is_file():
        return exact
    candidates = sorted(artifact_dir.glob(f"prefetch_list_{trace}_cl*_*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return exact
    raise RuntimeError(f"ambiguous rich-list files for {trace}: {candidates}")


def selected_by_address(path: Path):
    selected = defaultdict(list)
    if not path.is_file():
        return selected
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                address = to_int(row.get("prefetch_addr"))
                index = to_int(row.get("replay_idx") or row.get("demand_idx"))
            except Exception:
                continue
            selected[address].append(index)
    for values in selected.values():
        values.sort()
    return selected


def selected_before(selected, address: int, demand_idx: int) -> bool:
    return bisect.bisect_left(selected.get(address, []), demand_idx) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", required=True, type=Path)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--normal-prefetchers", required=True)
    parser.add_argument("--lstm-artifact", action="append", default=[],
                        help="LABEL=ARTIFACT_DIR; repeat for each frozen NN family")
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--export-suffix", default="pure_balanced_lru256")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    variants = {}
    for item in args.lstm_artifact:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory:
            raise ValueError(f"invalid --lstm-artifact: {item}")
        variants[label] = Path(directory)

    run_summary = []
    pair_summary = []
    residual_context = Counter()
    missing_reason = Counter()

    for trace in args.traces.split():
        oracle_by_key, oracle_rows = load_oracle(args.oracle_dir / f"{trace}.oracle.csv.gz")
        no_pref_misses = sum(row["no_pref_miss"] for row in oracle_rows)

        normal_events = {}
        for prefetcher in args.normal_prefetchers.split():
            event_file = args.event_root / "normal" / "events" / f"{trace}.{prefetcher}.events.csv.gz"
            events, unmatched = load_l2_events(event_file, oracle_by_key)
            normal_events[prefetcher] = events
            counts = Counter(outcome(row, events.get(key)) for key, row in oracle_by_key.items())
            run_summary.append({
                "trace": trace, "family": "normal", "variant": prefetcher,
                "event_file": str(event_file), "rich_list": "", "unmatched_event_keys": unmatched,
                "no_pref_miss_events": no_pref_misses,
                "timely_events": counts["timely"], "late_events": counts["late"],
                "residual_events": counts["residual"],
                "unique_event_coverage": counts["timely"] / no_pref_misses if no_pref_misses else 0.0,
                "timeliness_over_covered": counts["timely"] / (counts["timely"] + counts["late"])
                    if counts["timely"] + counts["late"] else 0.0,
            })
            for key, row in oracle_by_key.items():
                if row["no_pref_miss"] and outcome(row, events.get(key)) == "residual":
                    residual_context[(trace, "normal", prefetcher, row["pc"], row["delta"], row["page_offset"])] += 1

        lstm_events = {}
        selected = {}
        for label, artifact_dir in variants.items():
            event_file = args.event_root / "lstm" / label / "events" / f"{trace}.events.csv.gz"
            events, unmatched = load_l2_events(event_file, oracle_by_key)
            lstm_events[label] = events
            rich_list = find_rich_list(artifact_dir, trace, args.chunk_len, args.export_suffix)
            selected[label] = selected_by_address(rich_list)
            counts = Counter(outcome(row, events.get(key)) for key, row in oracle_by_key.items())
            run_summary.append({
                "trace": trace, "family": "standalone", "variant": label,
                "event_file": str(event_file), "rich_list": str(rich_list), "unmatched_event_keys": unmatched,
                "no_pref_miss_events": no_pref_misses,
                "timely_events": counts["timely"], "late_events": counts["late"],
                "residual_events": counts["residual"],
                "unique_event_coverage": counts["timely"] / no_pref_misses if no_pref_misses else 0.0,
                "timeliness_over_covered": counts["timely"] / (counts["timely"] + counts["late"])
                    if counts["timely"] + counts["late"] else 0.0,
            })
            for key, row in oracle_by_key.items():
                if row["no_pref_miss"] and outcome(row, events.get(key)) == "residual":
                    residual_context[(trace, "standalone", label, row["pc"], row["delta"], row["page_offset"])] += 1

        for label, standalone in lstm_events.items():
            for prefetcher, normal in normal_events.items():
                categories = Counter()
                reasons = Counter()
                for key, row in oracle_by_key.items():
                    if not row["no_pref_miss"]:
                        continue
                    normal_state = outcome(row, normal.get(key))
                    standalone_state = outcome(row, standalone.get(key))
                    if normal_state == "timely" and standalone_state == "timely":
                        category = "both_timely"
                    elif normal_state == "timely":
                        category = "normal_only_timely"
                        if standalone_state == "late":
                            reason = "standalone_selected_but_late"
                        elif selected_before(selected[label], row["addr"], row["demand_idx"]):
                            reason = "standalone_selected_earlier_but_not_timely"
                        else:
                            reason = "no_earlier_selected_standalone_export"
                        reasons[reason] += 1
                    elif standalone_state == "timely":
                        category = "standalone_only_timely"
                    elif normal_state == "late" and standalone_state == "late":
                        category = "both_late"
                    elif normal_state == "late":
                        category = "normal_late_standalone_not_timely"
                    elif standalone_state == "late":
                        category = "standalone_late_normal_not_timely"
                    else:
                        category = "neither_timely"
                    categories[category] += 1
                pair_summary.append({
                    "trace": trace, "standalone_variant": label, "normal_prefetcher": prefetcher,
                    **categories, **{f"reason_{name}": count for name, count in reasons.items()},
                })
                for reason, count in reasons.items():
                    missing_reason[(trace, label, prefetcher, reason)] += count

    contexts = []
    for (trace, family, variant, pc, delta, page_offset), count in residual_context.items():
        contexts.append({
            "trace": trace, "family": family, "variant": variant, "pc": f"0x{pc:x}",
            "delta": delta, "page_offset": page_offset,
            "residual_no_pref_miss_events": count,
        })
    contexts.sort(key=lambda row: (row["trace"], row["family"], row["variant"],
                                   -row["residual_no_pref_miss_events"]))
    top_contexts = []
    seen = Counter()
    for row in contexts:
        group = (row["trace"], row["family"], row["variant"])
        if seen[group] < args.top_k:
            top_contexts.append(row)
            seen[group] += 1

    reasons = [
        {"trace": trace, "standalone_variant": label, "normal_prefetcher": prefetcher,
         "reason": reason, "normal_only_timely_miss_events": count}
        for (trace, label, prefetcher, reason), count in sorted(missing_reason.items())
    ]
    write_rows(args.out_dir / "run_unique_event_outcomes.csv", run_summary, [
        "trace", "family", "variant", "event_file", "rich_list", "unmatched_event_keys",
        "no_pref_miss_events", "timely_events", "late_events", "residual_events",
        "unique_event_coverage", "timeliness_over_covered",
    ])
    fields = sorted({key for row in pair_summary for key in row})
    write_rows(args.out_dir / "normal_vs_standalone_target_attribution.csv", pair_summary, fields)
    write_rows(args.out_dir / "top_residual_contexts.csv", top_contexts, [
        "trace", "family", "variant", "pc", "delta", "page_offset",
        "residual_no_pref_miss_events",
    ])
    write_rows(args.out_dir / "normal_only_timely_standalone_reason.csv", reasons, [
        "trace", "standalone_variant", "normal_prefetcher", "reason",
        "normal_only_timely_miss_events",
    ])
    (args.out_dir / "README.json").write_text(json.dumps({
        "scope": "L2C demand-event analysis",
        "limit": "Existing frozen exports contain selections only. no_earlier_selected_standalone_export is not proof of candidate-bank absence.",
        "next_colab_requirement": "Export a full candidate decision ledger with scores, probabilities, and reject_reason.",
    }, indent=2) + "\n")
    print("[write]", args.out_dir)


if __name__ == "__main__":
    main()
