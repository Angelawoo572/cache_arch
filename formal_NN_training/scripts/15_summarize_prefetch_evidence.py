#!/usr/bin/env python3
"""Build an evidence report from trace, normal-prefetcher, and LSTM experiments.

This script never creates training labels. It turns existing experiment outputs into
measured evidence: trace composition, normal counter behavior, event-level timely
coverage, residual contexts, and normal-vs-standalone demand-outcome overlap.
"""
from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


def number(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_label_path(value):
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise ValueError("expected LABEL=PATH, got %r" % value)
    return label, Path(path)


def find_profile(profile_dir, trace):
    matches = sorted(profile_dir.glob(trace + "*.trace_profile.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text())


def baseline_best(rows):
    result = {}
    for row in rows:
        trace = row.get("trace", "")
        prefetcher = row.get("prefetcher", "")
        if not trace or prefetcher in ("no_pref", "none", "nopref"):
            continue
        if number(row.get("run_failed")):
            continue
        current = result.get(trace)
        if current is None or number(row.get("ipc")) > number(current.get("ipc")):
            result[trace] = row
    return result


def tags_for_run(row):
    tags = []
    coverage = number(row.get("unique_event_coverage"))
    timely = number(row.get("timeliness_over_covered"))
    issued = number(row.get("pf_issued"))
    useless = number(row.get("pf_useless"))
    if coverage and coverage < 0.25:
        tags.append("low_unique_event_coverage")
    if coverage and coverage < 0.10:
        tags.append("very_low_unique_event_coverage")
    if timely and timely < 0.90:
        tags.append("late_sensitive")
    if issued and useless / issued > 0.25:
        tags.append("high_useless_fraction")
    return ";".join(tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-profile-dir", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--attribution-dir", required=True, type=Path)
    parser.add_argument("--replay-summary", action="append", default=[],
                        help="LABEL=SUMMARY_CSV; repeat for each standalone family")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--traces", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_csv(args.baseline_summary)
    best_normal = baseline_best(baseline_rows)
    outcome_rows = read_csv(args.attribution_dir / "run_unique_event_outcomes.csv")
    pair_rows = read_csv(args.attribution_dir / "normal_vs_standalone_target_attribution.csv")
    residual_rows = read_csv(args.attribution_dir / "top_residual_contexts.csv")

    replay_rows = []
    for item in args.replay_summary:
        label, path = parse_label_path(item)
        for row in read_csv(path):
            copied = dict(row)
            # Plan-mode replay summaries already carry a unique candidate tag.
            # Preserve it so the event-attribution label and replay row can join.
            copied["standalone_variant"] = (
                copied.get("standalone_variant")
                or copied.get("candidate_tag")
                or label
            )
            copied["summary_path"] = str(path)
            replay_rows.append(copied)

    profile_rows = []
    for trace in args.traces.split():
        profile = find_profile(args.trace_profile_dir, trace)
        profile_rows.append({
            "trace": trace,
            "records_read": profile.get("profile_window_records_read", ""),
            "full_trace_counted": profile.get("full_trace_counted", ""),
            "compressed_trace_bytes": profile.get("compressed_trace_bytes", ""),
            "unique_pcs": profile.get("unique_pcs_in_window", ""),
            "unique_load_pages": profile.get("unique_load_pages_observed", ""),
            "branch_fraction": profile.get("branch_fraction", ""),
            "taken_fraction": profile.get("taken_fraction_of_branches", ""),
            "load_instruction_fraction": profile.get("load_instruction_fraction", ""),
            "store_instruction_fraction": profile.get("store_instruction_fraction", ""),
            "memory_instruction_fraction": profile.get("memory_instruction_fraction", ""),
        })
    write_csv(args.out_dir / "trace_profile_summary.csv", profile_rows, list(profile_rows[0].keys()))

    outcome_lookup = {(row.get("trace"), row.get("family"), row.get("variant")): row
                      for row in outcome_rows}
    normal_evidence = []
    for row in baseline_rows:
        trace = row.get("trace", "")
        prefetcher = row.get("prefetcher", "")
        joined = dict(row)
        outcome = outcome_lookup.get((trace, "normal", prefetcher), {})
        joined.update({
            "unique_event_coverage": outcome.get("unique_event_coverage", ""),
            "event_timeliness_over_covered": outcome.get("timeliness_over_covered", ""),
            "event_timely": outcome.get("timely_events", ""),
            "event_late": outcome.get("late_events", ""),
            "event_residual": outcome.get("residual_events", ""),
        })
        joined["evidence_tags"] = tags_for_run(joined)
        normal_evidence.append(joined)
    normal_fields = sorted({key for row in normal_evidence for key in row})
    write_csv(args.out_dir / "normal_prefetcher_evidence.csv", normal_evidence, normal_fields)

    standalone_evidence = []
    for row in replay_rows:
        trace = row.get("trace", "")
        label = row.get("standalone_variant", "")
        joined = dict(row)
        outcome = outcome_lookup.get((trace, "standalone", label), {})
        joined.update({
            "unique_event_coverage": outcome.get("unique_event_coverage", ""),
            "event_timeliness_over_covered": outcome.get("timeliness_over_covered", ""),
            "event_timely": outcome.get("timely_events", ""),
            "event_late": outcome.get("late_events", ""),
            "event_residual": outcome.get("residual_events", ""),
        })
        joined["evidence_tags"] = tags_for_run(joined)
        standalone_evidence.append(joined)
    standalone_fields = sorted({key for row in standalone_evidence for key in row})
    write_csv(args.out_dir / "standalone_prefetcher_evidence.csv", standalone_evidence, standalone_fields)

    pair_evidence = []
    for row in pair_rows:
        trace = row.get("trace", "")
        normal = row.get("normal_prefetcher", "")
        label = row.get("standalone_variant", "")
        enriched = dict(row)
        normal_row = next((x for x in normal_evidence
                           if x.get("trace") == trace and x.get("prefetcher") == normal), {})
        standalone_row = next((x for x in standalone_evidence
                               if x.get("trace") == trace and x.get("standalone_variant") == label), {})
        enriched.update({
            "normal_ipc": normal_row.get("ipc", ""),
            "standalone_ipc": standalone_row.get("ipc", ""),
            "ipc_delta_standalone_minus_normal": number(standalone_row.get("ipc")) - number(normal_row.get("ipc")),
            "normal_event_coverage": normal_row.get("unique_event_coverage", ""),
            "standalone_event_coverage": standalone_row.get("unique_event_coverage", ""),
            "normal_event_timeliness": normal_row.get("event_timeliness_over_covered", ""),
            "standalone_event_timeliness": standalone_row.get("event_timeliness_over_covered", ""),
        })
        pair_evidence.append(enriched)
    pair_fields = sorted({key for row in pair_evidence for key in row})
    write_csv(args.out_dir / "normal_vs_standalone_evidence.csv", pair_evidence, pair_fields)

    report = []
    report.append("# Trace and prefetch evidence report")
    report.append("")
    report.append("This report records measured associations from trace profiling, counter summaries, and L2C demand-event attribution. It does not claim a causal explanation beyond what these counters and event outcomes establish.")
    report.append("")
    report.append("## Inputs")
    report.append("")
    report.append("- Trace profiles: `%s`" % args.trace_profile_dir)
    report.append("- Normal baseline summary: `%s`" % args.baseline_summary)
    report.append("- Event attribution: `%s`" % args.attribution_dir)
    for item in args.replay_summary:
        report.append("- Standalone replay summary: `%s`" % item)
    report.append("")

    for trace in args.traces.split():
        profile = next((row for row in profile_rows if row["trace"] == trace), {})
        best = best_normal.get(trace, {})
        report.append("## %s" % trace)
        report.append("")
        report.append("Trace window: %s instructions profiled; unique PCs=%s; branch fraction=%s; load-instruction fraction=%s; memory-instruction fraction=%s." % (
            profile.get("records_read", ""), profile.get("unique_pcs", ""),
            profile.get("branch_fraction", ""), profile.get("load_instruction_fraction", ""),
            profile.get("memory_instruction_fraction", ""),
        ))
        report.append("")
        if best:
            report.append("Best normal IPC: `%s` at `%s`; issued=%s, useful=%s, useless=%s, late=%s, selected accuracy=%s, timeliness=%s." % (
                best.get("prefetcher"), best.get("ipc"), best.get("pf_issued"),
                best.get("pf_useful"), best.get("pf_useless"), best.get("pf_late"),
                best.get("selected_accuracy"), best.get("timeliness"),
            ))
            report.append("")

        report.append("| standalone | IPC | delta vs best normal | selected accuracy | counter timeliness | unique demand-event coverage | event timeliness | issued | useful | useless | late |")
        report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        trace_lstm_rows = [row for row in standalone_evidence if row.get("trace") == trace]
        for row in trace_lstm_rows:
            report.append("| %s | %s | %.6f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("standalone_variant", ""), row.get("ipc", ""),
                number(row.get("ipc")) - number(best.get("ipc")),
                row.get("selected_accuracy", ""), row.get("timeliness", ""),
                row.get("unique_event_coverage", ""), row.get("event_timeliness_over_covered", ""),
                row.get("pf_issued", ""), row.get("pf_useful", ""),
                row.get("pf_useless", ""), row.get("pf_late", ""),
            ))
        report.append("")

        best_pairs = [row for row in pair_evidence
                      if row.get("trace") == trace and row.get("normal_prefetcher") == best.get("prefetcher")]
        if best_pairs:
            report.append("Demand-event overlap against the best normal prefetcher:")
            report.append("")
            report.append("| standalone | both timely | normal-only timely | standalone-only timely | both late | neither timely | selected but late | no earlier selected export |")
            report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for row in best_pairs:
                report.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    row.get("standalone_variant", ""), row.get("both_timely", 0),
                    row.get("normal_only_timely", 0), row.get("standalone_only_timely", 0),
                    row.get("both_late", 0), row.get("neither_timely", 0),
                    row.get("reason_standalone_selected_but_late", 0),
                    row.get("reason_no_earlier_selected_standalone_export", 0),
                ))
            report.append("")

        trace_residual = [row for row in residual_rows if row.get("trace") == trace]
        trace_residual.sort(key=lambda row: -number(row.get("residual_no_pref_miss_events")))
        if trace_residual:
            report.append("Top residual contexts are preserved in `top_residual_contexts.csv`; they are evidence for later candidate-representation analysis, not NN labels.")
            report.append("")

    (args.out_dir / "trace_prefetch_evidence_report.md").write_text("\n".join(report) + "\n")
    print("[write] %s" % args.out_dir)


if __name__ == "__main__":
    main()
