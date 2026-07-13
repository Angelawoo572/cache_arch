#!/usr/bin/env python3
"""Build measured normal-vs-standalone prefetch evidence.

This report does not create NN labels and does not claim causal mechanisms from
counters alone.  It joins trace profiles, normal counter summaries, keyed replay
summaries, and L2C demand-event attribution.  In addition to IPC and prefetch
accuracy, it writes an explicit cache-miss comparison so a high-precision
prefetcher is not incorrectly assumed to cover the same misses as another
high-precision prefetcher.
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
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def parse_label_path(value):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError("expected LABEL=PATH, got {!r}".format(value))
    return label, Path(path)


def find_profile(profile_dir, trace):
    matches = sorted(Path(profile_dir).glob(trace + "*.trace_profile.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text())


def baseline_best(rows):
    output = {}
    for row in rows:
        trace = row.get("trace", "")
        prefetcher = row.get("prefetcher", "")
        if not trace or prefetcher in ("no_pref", "none", "nopref"):
            continue
        if number(row.get("run_failed")):
            continue
        current = output.get(trace)
        if current is None or number(row.get("ipc")) > number(current.get("ipc")):
            output[trace] = row
    return output


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


def miss_reading(row):
    """A descriptive guardrail, not a causal explanation."""
    delta = number(row.get("l2_miss_delta_standalone_minus_normal"))
    normal_cov = number(row.get("normal_event_coverage"))
    standalone_cov = number(row.get("standalone_event_coverage"))
    normal_time = number(row.get("normal_event_timeliness"))
    standalone_time = number(row.get("standalone_event_timeliness"))
    if delta < 0 and standalone_cov >= normal_cov and standalone_time >= normal_time:
        return "NN has fewer L2 misses and no lower measured timely-event coverage/timeliness."
    if delta < 0:
        return "NN has fewer L2 misses; inspect overlap and resource counters before assigning cause."
    if delta > 0 and standalone_cov < normal_cov:
        return "NN has more L2 misses and lower measured timely-event coverage."
    if delta > 0 and standalone_time < normal_time:
        return "NN has more L2 misses and lower measured event timeliness."
    if delta > 0:
        return "NN has more L2 misses despite its selected-precision value; inspect target overlap, dedup, and resource pressure."
    return "Equal L2 miss count in this measured window; compare event overlap and resource counters."


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
            copied["standalone_variant"] = (
                copied.get("standalone_variant") or copied.get("candidate_tag") or label
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
    write_csv(args.out_dir / "trace_profile_summary.csv", profile_rows)

    outcome_lookup = {
        (row.get("trace"), row.get("family"), row.get("variant")): row
        for row in outcome_rows
    }

    normal_evidence = []
    for row in baseline_rows:
        joined = dict(row)
        outcome = outcome_lookup.get((row.get("trace"), "normal", row.get("prefetcher")), {})
        joined.update({
            "unique_event_coverage": outcome.get("unique_event_coverage", ""),
            "event_timeliness_over_covered": outcome.get("timeliness_over_covered", ""),
            "event_timely": outcome.get("timely_events", ""),
            "event_late": outcome.get("late_events", ""),
            "event_residual": outcome.get("residual_events", ""),
        })
        joined["evidence_tags"] = tags_for_run(joined)
        normal_evidence.append(joined)
    write_csv(args.out_dir / "normal_prefetcher_evidence.csv", normal_evidence)

    standalone_evidence = []
    for row in replay_rows:
        joined = dict(row)
        trace = row.get("trace", "")
        variant = row.get("standalone_variant", "")
        outcome = outcome_lookup.get((trace, "standalone", variant), {})
        joined.update({
            "unique_event_coverage": outcome.get("unique_event_coverage", ""),
            "event_timeliness_over_covered": outcome.get("timeliness_over_covered", ""),
            "event_timely": outcome.get("timely_events", ""),
            "event_late": outcome.get("late_events", ""),
            "event_residual": outcome.get("residual_events", ""),
        })
        joined["evidence_tags"] = tags_for_run(joined)
        standalone_evidence.append(joined)
    write_csv(args.out_dir / "standalone_prefetcher_evidence.csv", standalone_evidence)

    normal_index = {
        (row.get("trace"), row.get("prefetcher")): row for row in normal_evidence
    }
    standalone_index = {
        (row.get("trace"), row.get("standalone_variant")): row for row in standalone_evidence
    }

    pair_evidence = []
    cache_miss_rows = []
    for row in pair_rows:
        trace = row.get("trace", "")
        normal_name = row.get("normal_prefetcher", "")
        standalone_name = row.get("standalone_variant", "")
        normal_row = normal_index.get((trace, normal_name), {})
        standalone_row = standalone_index.get((trace, standalone_name), {})
        enriched = dict(row)
        normal_miss = number(normal_row.get("l2_load_miss"))
        standalone_miss = number(standalone_row.get("l2_load_miss"))
        normal_rate = number(normal_row.get("l2_load_miss_rate"))
        standalone_rate = number(standalone_row.get("l2_load_miss_rate"))
        enriched.update({
            "normal_ipc": normal_row.get("ipc", ""),
            "standalone_ipc": standalone_row.get("ipc", ""),
            "ipc_delta_standalone_minus_normal": number(standalone_row.get("ipc")) - number(normal_row.get("ipc")),
            "normal_event_coverage": normal_row.get("unique_event_coverage", ""),
            "standalone_event_coverage": standalone_row.get("unique_event_coverage", ""),
            "normal_event_timeliness": normal_row.get("event_timeliness_over_covered", ""),
            "standalone_event_timeliness": standalone_row.get("event_timeliness_over_covered", ""),
            "normal_l2_load_miss": int(normal_miss),
            "standalone_l2_load_miss": int(standalone_miss),
            "l2_miss_delta_standalone_minus_normal": int(standalone_miss - normal_miss),
            "normal_l2_load_miss_rate": normal_rate,
            "standalone_l2_load_miss_rate": standalone_rate,
            "l2_miss_rate_delta_standalone_minus_normal": standalone_rate - normal_rate,
            "normal_selected_accuracy": normal_row.get("selected_accuracy", ""),
            "standalone_selected_accuracy": standalone_row.get("selected_accuracy", ""),
            "normal_pf_issued": normal_row.get("pf_issued", ""),
            "standalone_pf_issued": standalone_row.get("pf_issued", ""),
            "normal_pf_useless": normal_row.get("pf_useless", ""),
            "standalone_pf_useless": standalone_row.get("pf_useless", ""),
            "normal_pf_late": normal_row.get("pf_late", ""),
            "standalone_pf_late": standalone_row.get("pf_late", ""),
        })
        enriched["cache_miss_reading"] = miss_reading(enriched)
        pair_evidence.append(enriched)
        cache_miss_rows.append({
            "trace": trace,
            "normal_prefetcher": normal_name,
            "standalone_variant": standalone_name,
            "normal_l2_load_miss": int(normal_miss),
            "standalone_l2_load_miss": int(standalone_miss),
            "l2_miss_delta_standalone_minus_normal": int(standalone_miss - normal_miss),
            "normal_l2_load_miss_rate": normal_rate,
            "standalone_l2_load_miss_rate": standalone_rate,
            "l2_miss_rate_delta_standalone_minus_normal": standalone_rate - normal_rate,
            "normal_selected_accuracy": normal_row.get("selected_accuracy", ""),
            "standalone_selected_accuracy": standalone_row.get("selected_accuracy", ""),
            "normal_event_coverage": normal_row.get("unique_event_coverage", ""),
            "standalone_event_coverage": standalone_row.get("unique_event_coverage", ""),
            "normal_event_timeliness": normal_row.get("event_timeliness_over_covered", ""),
            "standalone_event_timeliness": standalone_row.get("event_timeliness_over_covered", ""),
            "both_timely": row.get("both_timely", 0),
            "normal_only_timely": row.get("normal_only_timely", 0),
            "standalone_only_timely": row.get("standalone_only_timely", 0),
            "both_late": row.get("both_late", 0),
            "neither_timely": row.get("neither_timely", 0),
            "cache_miss_reading": enriched["cache_miss_reading"],
        })
    write_csv(args.out_dir / "normal_vs_standalone_evidence.csv", pair_evidence)
    write_csv(args.out_dir / "cache_miss_comparison.csv", cache_miss_rows)

    report = [
        "# Trace and prefetch evidence report",
        "",
        "This report records measured associations from trace profiling, counter summaries, and L2C demand-event attribution. It does not claim a causal explanation beyond these measured counters and outcomes.",
        "",
        "## Inputs",
        "",
        "- Trace profiles: `{}`".format(args.trace_profile_dir),
        "- Normal baseline summary: `{}`".format(args.baseline_summary),
        "- Event attribution: `{}`".format(args.attribution_dir),
    ]
    for item in args.replay_summary:
        report.append("- Standalone replay summary: `{}`".format(item))
    report.append("")

    for trace in args.traces.split():
        profile = next((row for row in profile_rows if row["trace"] == trace), {})
        best = best_normal.get(trace, {})
        report.extend([
            "## {}".format(trace),
            "",
            "Trace window: {} instructions profiled; unique PCs={}; branch fraction={}; load-instruction fraction={}; memory-instruction fraction={} .".format(
                profile.get("records_read", ""), profile.get("unique_pcs", ""),
                profile.get("branch_fraction", ""), profile.get("load_instruction_fraction", ""),
                profile.get("memory_instruction_fraction", ""),
            ),
            "",
        ])
        if best:
            report.extend([
                "Best normal IPC: `{}` at `{}`; L2 misses={}; selected accuracy={}; timeliness={}.".format(
                    best.get("prefetcher"), best.get("ipc"), best.get("l2_load_miss"),
                    best.get("selected_accuracy"), best.get("timeliness"),
                ),
                "",
            ])
        report.extend([
            "| standalone | IPC | delta vs best normal | selected accuracy | counter timeliness | unique demand-event coverage | event timeliness | L2 misses |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in standalone_evidence:
            if row.get("trace") != trace:
                continue
            report.append("| {} | {} | {:.6f} | {} | {} | {} | {} | {} |".format(
                row.get("standalone_variant", ""), row.get("ipc", ""),
                number(row.get("ipc")) - number(best.get("ipc")),
                row.get("selected_accuracy", ""), row.get("timeliness", ""),
                row.get("unique_event_coverage", ""), row.get("event_timeliness_over_covered", ""),
                row.get("l2_load_miss", ""),
            ))
        report.append("")

        best_pairs = [
            row for row in pair_evidence
            if row.get("trace") == trace and row.get("normal_prefetcher") == best.get("prefetcher")
        ]
        if best_pairs:
            report.extend([
                "Demand-event and cache-miss comparison against the best normal prefetcher:",
                "",
                "| standalone | normal L2 misses | standalone L2 misses | NN - normal misses | both timely | normal-only timely | standalone-only timely | reading |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ])
            for row in best_pairs:
                report.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    row.get("standalone_variant", ""), row.get("normal_l2_load_miss", ""),
                    row.get("standalone_l2_load_miss", ""),
                    row.get("l2_miss_delta_standalone_minus_normal", ""),
                    row.get("both_timely", 0), row.get("normal_only_timely", 0),
                    row.get("standalone_only_timely", 0), row.get("cache_miss_reading", ""),
                ))
            report.append("")

        residual = [row for row in residual_rows if row.get("trace") == trace]
        if residual:
            report.extend([
                "Top residual contexts are preserved in `top_residual_contexts.csv`; they are evidence for later candidate-representation analysis, not NN labels.",
                "",
            ])

    (args.out_dir / "trace_prefetch_evidence_report.md").write_text("\n".join(report) + "\n")
    print("[write] {}".format(args.out_dir))


if __name__ == "__main__":
    main()
