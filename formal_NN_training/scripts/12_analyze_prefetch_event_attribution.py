#!/usr/bin/env python3
"""Compare normal and frozen standalone prefetchers at L2C demand-event granularity.

Normal-prefetcher results are analysis references only. Existing standalone rich
exports contain only selected list entries, not every candidate that the notebook
considered. Consequently, an address absent from an earlier selected export is
not proof that it was absent from the candidate bank.
"""
import argparse
import bisect
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="")


def to_int(value, default=None):
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("missing integer")
        return default
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_oracle(path):
    by_key = {}
    rows = []
    with open_text(path) as handle:
        for raw in csv.DictReader(handle):
            row = {
                "demand_idx": to_int(raw.get("demand_idx")),
                "base_event_id": to_int(raw.get("base_event_id"), 0),
                "pc": to_int(raw.get("pc")),
                "line": to_int(raw.get("line")),
                "addr": to_int(raw.get("addr")),
                "delta": to_int(raw.get("delta"), 0),
                "page_offset": to_int(raw.get("page_offset"), 0),
                "no_pref_miss": to_int(raw.get("no_pref_miss"), 0),
            }
            key = (row["pc"], row["line"], to_int(raw.get("pc_line_occ")))
            by_key[key] = row
            rows.append((key, row))
    return by_key, rows


def load_l2_events(path, oracle_by_key):
    if not path.is_file():
        return {}, 0, 0
    occurrence = Counter()
    events = {}
    unmatched = 0
    observed = 0
    with open_text(path) as handle:
        for raw in csv.DictReader(handle):
            if raw.get("event", "").upper() != "DEMAND":
                continue
            if raw.get("cache", "").upper() != "L2C":
                continue
            if str(raw.get("type", "")).strip() not in ("0", "0.0"):
                continue
            observed += 1
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
    return events, unmatched, observed


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


def find_rich_list(artifact_dir, trace, chunk_len, suffix):
    exact = artifact_dir / ("prefetch_list_%s_cl%s_%s.csv" % (trace, chunk_len, suffix))
    if exact.is_file():
        return exact
    candidates = sorted(artifact_dir.glob("prefetch_list_%s_cl*_*.csv" % trace))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return exact
    raise RuntimeError("ambiguous rich-list files for %s: %s" % (trace, candidates))


def selected_by_address(path):
    selected = defaultdict(list)
    if not path.is_file():
        return selected
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                address = to_int(raw.get("prefetch_addr") or raw.get("addr"))
                index = to_int(raw.get("replay_idx") or raw.get("demand_idx"))
            except (ValueError, TypeError):
                continue
            selected[address].append(index)
    for indexes in selected.values():
        indexes.sort()
    return selected


def selected_before(selected, address, demand_idx):
    indexes = selected.get(address, [])
    return bisect.bisect_left(indexes, demand_idx) > 0


def event_category(normal_state, standalone_state, selected_earlier):
    if normal_state == "timely" and standalone_state == "timely":
        return "both_timely", ""
    if normal_state == "timely":
        if standalone_state == "late":
            return "normal_only_timely", "standalone_selected_but_late"
        if selected_earlier:
            return "normal_only_timely", "standalone_selected_earlier_but_not_timely"
        return "normal_only_timely", "no_earlier_selected_standalone_export"
    if standalone_state == "timely":
        return "standalone_only_timely", ""
    if normal_state == "late" and standalone_state == "late":
        return "both_late", ""
    if normal_state == "late":
        return "normal_late_standalone_not_timely", ""
    if standalone_state == "late":
        return "standalone_late_normal_not_timely", ""
    return "neither_timely", ""


def parse_variants(items):
    variants = {}
    for item in items:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory:
            raise ValueError("invalid --lstm-artifact: %s" % item)
        variants[label] = Path(directory)
    return variants


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
    parser.add_argument("--write-event-rows", action="store_true",
                        help="Write a potentially large per-demand comparison CSV.GZ.")
    args = parser.parse_args()

    variants = parse_variants(args.lstm_artifact)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_summary = []
    pair_summary = []
    residual_context = Counter()
    missing_reason = Counter()
    detail_path = args.out_dir / "normal_vs_standalone_event_rows.csv.gz"
    detail_handle = None
    detail_writer = None
    if args.write_event_rows:
        detail_handle = open_text(detail_path, "wt")
        fields = [
            "trace", "standalone_variant", "normal_prefetcher", "demand_idx",
            "base_event_id", "pc", "line", "delta", "page_offset",
            "normal_state", "standalone_state", "standalone_selected_before",
            "category", "normal_only_reason",
        ]
        detail_writer = csv.DictWriter(detail_handle, fieldnames=fields)
        detail_writer.writeheader()

    try:
        for trace in args.traces.split():
            oracle_path = args.oracle_dir / (trace + ".oracle.csv.gz")
            oracle_by_key, oracle_rows = load_oracle(oracle_path)
            no_pref_misses = sum(row["no_pref_miss"] for _, row in oracle_rows)

            normal_events = {}
            for prefetcher in args.normal_prefetchers.split():
                event_file = args.event_root / "normal" / "events" / (trace + "." + prefetcher + ".events.csv.gz")
                events, unmatched, observed = load_l2_events(event_file, oracle_by_key)
                normal_events[prefetcher] = events
                outcomes = Counter(outcome(row, events.get(key)) for key, row in oracle_rows)
                run_summary.append({
                    "trace": trace, "family": "normal", "variant": prefetcher,
                    "event_file": str(event_file), "rich_list": "",
                    "event_rows_observed": observed, "unmatched_event_keys": unmatched,
                    "no_pref_miss_events": no_pref_misses,
                    "timely_events": outcomes["timely"], "late_events": outcomes["late"],
                    "residual_events": outcomes["residual"],
                    "unique_event_coverage": float(outcomes["timely"]) / no_pref_misses if no_pref_misses else 0.0,
                    "timeliness_over_covered": float(outcomes["timely"]) / (outcomes["timely"] + outcomes["late"])
                        if outcomes["timely"] + outcomes["late"] else 0.0,
                })
                for key, row in oracle_rows:
                    if row["no_pref_miss"] and outcome(row, events.get(key)) == "residual":
                        residual_context[(trace, "normal", prefetcher, row["pc"], row["delta"], row["page_offset"])] += 1

            lstm_events = {}
            selected = {}
            for label, artifact_dir in variants.items():
                event_file = args.event_root / "lstm" / label / "events" / (trace + ".events.csv.gz")
                events, unmatched, observed = load_l2_events(event_file, oracle_by_key)
                lstm_events[label] = events
                rich_list = find_rich_list(artifact_dir, trace, args.chunk_len, args.export_suffix)
                selected[label] = selected_by_address(rich_list)
                outcomes = Counter(outcome(row, events.get(key)) for key, row in oracle_rows)
                run_summary.append({
                    "trace": trace, "family": "standalone", "variant": label,
                    "event_file": str(event_file), "rich_list": str(rich_list),
                    "event_rows_observed": observed, "unmatched_event_keys": unmatched,
                    "no_pref_miss_events": no_pref_misses,
                    "timely_events": outcomes["timely"], "late_events": outcomes["late"],
                    "residual_events": outcomes["residual"],
                    "unique_event_coverage": float(outcomes["timely"]) / no_pref_misses if no_pref_misses else 0.0,
                    "timeliness_over_covered": float(outcomes["timely"]) / (outcomes["timely"] + outcomes["late"])
                        if outcomes["timely"] + outcomes["late"] else 0.0,
                })
                for key, row in oracle_rows:
                    if row["no_pref_miss"] and outcome(row, events.get(key)) == "residual":
                        residual_context[(trace, "standalone", label, row["pc"], row["delta"], row["page_offset"])] += 1

            for label, standalone in lstm_events.items():
                for prefetcher, normal in normal_events.items():
                    categories = Counter()
                    reasons = Counter()
                    for key, row in oracle_rows:
                        if not row["no_pref_miss"]:
                            continue
                        normal_state = outcome(row, normal.get(key))
                        standalone_state = outcome(row, standalone.get(key))
                        earlier = selected_before(selected[label], row["addr"], row["demand_idx"])
                        category, reason = event_category(normal_state, standalone_state, earlier)
                        categories[category] += 1
                        if reason:
                            reasons[reason] += 1
                        if detail_writer is not None:
                            detail_writer.writerow({
                                "trace": trace, "standalone_variant": label,
                                "normal_prefetcher": prefetcher, "demand_idx": row["demand_idx"],
                                "base_event_id": row["base_event_id"], "pc": "0x%x" % row["pc"],
                                "line": row["line"], "delta": row["delta"], "page_offset": row["page_offset"],
                                "normal_state": normal_state, "standalone_state": standalone_state,
                                "standalone_selected_before": int(earlier), "category": category,
                                "normal_only_reason": reason,
                            })
                    output = {"trace": trace, "standalone_variant": label,
                              "normal_prefetcher": prefetcher}
                    output.update(categories)
                    for name, count in reasons.items():
                        output["reason_" + name] = count
                    pair_summary.append(output)
                    for reason, count in reasons.items():
                        missing_reason[(trace, label, prefetcher, reason)] += count
    finally:
        if detail_handle is not None:
            detail_handle.close()

    contexts = []
    for key, count in residual_context.items():
        trace, family, variant, pc, delta, page_offset = key
        contexts.append({
            "trace": trace, "family": family, "variant": variant,
            "pc": "0x%x" % pc, "delta": delta, "page_offset": page_offset,
            "residual_no_pref_miss_events": count,
        })
    contexts.sort(key=lambda row: (row["trace"], row["family"], row["variant"],
                                   -row["residual_no_pref_miss_events"]))
    top_contexts = []
    group_count = Counter()
    for row in contexts:
        group = (row["trace"], row["family"], row["variant"])
        if group_count[group] < args.top_k:
            top_contexts.append(row)
            group_count[group] += 1

    reason_rows = []
    for key, count in sorted(missing_reason.items()):
        trace, label, prefetcher, reason = key
        reason_rows.append({
            "trace": trace, "standalone_variant": label,
            "normal_prefetcher": prefetcher, "reason": reason,
            "normal_only_timely_miss_events": count,
        })

    write_rows(args.out_dir / "run_unique_event_outcomes.csv", run_summary, [
        "trace", "family", "variant", "event_file", "rich_list", "event_rows_observed",
        "unmatched_event_keys", "no_pref_miss_events", "timely_events", "late_events",
        "residual_events", "unique_event_coverage", "timeliness_over_covered",
    ])
    fields = sorted({name for row in pair_summary for name in row})
    write_rows(args.out_dir / "normal_vs_standalone_target_attribution.csv", pair_summary, fields)
    write_rows(args.out_dir / "top_residual_contexts.csv", top_contexts, [
        "trace", "family", "variant", "pc", "delta", "page_offset",
        "residual_no_pref_miss_events",
    ])
    write_rows(args.out_dir / "normal_only_timely_standalone_reason.csv", reason_rows, [
        "trace", "standalone_variant", "normal_prefetcher", "reason",
        "normal_only_timely_miss_events",
    ])
    metadata = {
        "scope": "L2C demand-event analysis",
        "event_rows_written": bool(args.write_event_rows),
        "limit": (
            "Frozen standalone exports contain selections only. "
            "no_earlier_selected_standalone_export is not proof of candidate-bank absence."
        ),
        "next_notebook_requirement": (
            "Export a full decision ledger with every candidate, score/probabilities, "
            "candidate-source, and reject_reason to distinguish candidate absence from policy rejection."
        ),
    }
    (args.out_dir / "README.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[write] %s" % args.out_dir)


if __name__ == "__main__":
    main()
