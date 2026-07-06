#!/usr/bin/env python3
"""Build replay-gated V4.8 analysis tables with only the Python standard library.

This is deliberately general: it has no trace ID, PC, delta, local path, or
experiment-tag special case.  It joins a keyed replay summary, a fixed normal
matrix, Colab metadata, optional event attribution, and optional PQ/MSHR data.
Offline metrics never produce a claim that an NN beats a normal prefetcher.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def num(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def integer(value, default=0):
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def ratio(a, b):
    a, b = num(a), num(b)
    return a / b if b else float("nan")


def metric(row, names):
    for name in names:
        if row.get(name) not in (None, ""):
            return num(row.get(name))
    return 0.0


def keyed_ok(row):
    return integer(row.get("replay_transport_ok"), 1) == 1 and integer(row.get("run_failed")) == 0


def load_criteria(value):
    default = {
        "ipc_ratio_min": 0.995, "accuracy_ratio_min": 0.95,
        "coverage_ratio_min": 0.95, "timeliness_ratio_min": 0.95,
        "issue_ratio_max": 1.10, "pq_p95_ratio_max": 1.10,
        "mshr_p95_ratio_max": 1.10, "rejected_delta_max": 0.05,
        "duplicate_delta_max": 0.05,
    }
    if not value:
        return default
    path = Path(value)
    loaded = json.loads(path.read_text()) if path.is_file() else json.loads(value)
    default.update(loaded)
    return default


def normal_by_trace(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("trace") and not integer(row.get("run_failed")):
            grouped[row["trace"]].append(row)
    out = {}
    for trace, values in grouped.items():
        no_pref = next((r for r in values if r.get("prefetcher") in ("no_pref", "none", "nopref")), {})
        normals = [r for r in values if r.get("prefetcher") not in ("no_pref", "none", "nopref")]
        out[trace] = {"no_pref": no_pref, "best": max(normals, key=lambda r: num(r.get("ipc"))) if normals else {}, "all": values}
    return out


def resource_index(rows):
    out = {}
    for row in rows:
        tag = row.get("candidate_tag") or row.get("variant") or row.get("tag")
        if tag and tag not in out:
            out[tag] = row
    return out


def merge(metadata, replay, attribution):
    meta = {r.get("tag") or r.get("candidate_tag"): r for r in metadata}
    attr = {(r.get("trace", ""), r.get("variant", "")): r for r in attribution if r.get("family") == "standalone"}
    out = []
    for row in replay:
        tag = row.get("candidate_tag") or row.get("tag")
        merged = dict(meta.get(tag, {}))
        merged.update(row)
        merged["candidate_tag"] = tag
        event = attr.get((merged.get("trace", ""), tag), {})
        if event:
            merged.update({
                "nn_event_coverage": event.get("unique_event_coverage", ""),
                "nn_event_timeliness": event.get("timeliness_over_covered", ""),
                "nn_event_timely": event.get("timely_events", ""),
                "nn_event_late": event.get("late_events", ""),
                "nn_event_residual": event.get("residual_events", ""),
            })
        out.append(merged)
    return out


def enrich(rows, normals):
    out = []
    for row in rows:
        trace = row.get("trace", "")
        ref = normals.get(trace, {})
        no_pref, best = ref.get("no_pref", {}), ref.get("best", {})
        ipc = num(row.get("replay_ipc", row.get("ipc")))
        copied = dict(row)
        copied.update({
            "keyed_replay_valid": int(keyed_ok(row)),
            "no_pref_ipc_fixed_matrix": num(no_pref.get("ipc")),
            "best_normal_prefetcher": best.get("prefetcher", ""),
            "best_normal_ipc_fixed_matrix": num(best.get("ipc")),
            "ipc_delta_vs_no_pref_fixed_matrix": ipc - num(no_pref.get("ipc")),
            "ipc_delta_vs_best_normal_fixed_matrix": ipc - num(best.get("ipc")),
            "beats_best_normal_by_keyed_replay": int(keyed_ok(row) and ipc > num(best.get("ipc"))),
        })
        out.append(copied)
    return out


def all_normal_rows(rows, normals):
    out = []
    for row in rows:
        trace = row.get("trace", "")
        ipc = num(row.get("replay_ipc", row.get("ipc")))
        for normal in normals.get(trace, {}).get("all", []):
            copied = dict(row)
            copied.update({
                "normal_prefetcher": normal.get("prefetcher", ""),
                "normal_ipc": num(normal.get("ipc")), "nn_ipc": ipc,
                "ipc_delta_nn_minus_normal": ipc - num(normal.get("ipc")),
                "nn_beats_this_normal_by_keyed_replay": int(keyed_ok(row) and ipc > num(normal.get("ipc"))),
            })
            out.append(copied)
    return out


def seed_variance(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("trace", ""), row.get("route_id", row.get("bank_id", "")), row.get("policy_tag", ""), row.get("model_size", ""), row.get("stage", row.get("phase", "")))
        groups[key].append(row)
    out = []
    for key, values in sorted(groups.items()):
        ipcs = [num(v.get("replay_ipc", v.get("ipc"))) for v in values if keyed_ok(v)]
        if not ipcs:
            continue
        out.append({
            "trace": key[0], "route_id": key[1], "policy_tag": key[2], "model_size": key[3], "stage": key[4],
            "keyed_replay_count": len(ipcs), "replay_ipc_mean": statistics.mean(ipcs),
            "replay_ipc_min": min(ipcs), "replay_ipc_max": max(ipcs),
            "replay_ipc_stdev": statistics.pstdev(ipcs) if len(ipcs) > 1 else 0.0,
            "seeds": ";".join(str(v.get("seed", "")) for v in values if keyed_ok(v)),
        })
    return out


def rsrc(row, names):
    return metric(row, names)


def stage_b(rows, resources, criteria):
    by_tag = {r.get("candidate_tag"): r for r in rows if r.get("candidate_tag")}
    out = []
    for row in rows:
        if row.get("stage", row.get("phase", "")) != "stage_b":
            continue
        ref_tag = row.get("selected_full_reference_tag", "")
        ref = by_tag.get(ref_tag)
        copied = dict(row)
        if ref is None:
            copied.update({"accepted": 0, "failure_reason": "missing selected-full keyed replay reference"})
            out.append(copied)
            continue
        tag = row.get("candidate_tag")
        cur_resource, ref_resource = resources.get(tag, {}), resources.get(ref_tag, {})
        ipc_ratio = ratio(row.get("replay_ipc", row.get("ipc")), ref.get("replay_ipc", ref.get("ipc")))
        acc_ratio = ratio(metric(row, ["selected_accuracy", "accuracy"]), metric(ref, ["selected_accuracy", "accuracy"]))
        cov_ratio = ratio(metric(row, ["nn_event_coverage", "unique_event_coverage", "coverage"]), metric(ref, ["nn_event_coverage", "unique_event_coverage", "coverage"]))
        time_ratio = ratio(metric(row, ["nn_event_timeliness", "event_timeliness_over_covered", "timeliness"]), metric(ref, ["nn_event_timeliness", "event_timeliness_over_covered", "timeliness"]))
        issue_ratio = ratio(metric(row, ["nn_issue_per_l2_load", "issue_per_event", "val_policy_issue_per_event"]), metric(ref, ["nn_issue_per_l2_load", "issue_per_event", "val_policy_issue_per_event"]))
        pq_ratio = ratio(rsrc(cur_resource, ["demand_pq_occ_p95", "pf_pq_occ_p95", "pq_occ_p95"]), rsrc(ref_resource, ["demand_pq_occ_p95", "pf_pq_occ_p95", "pq_occ_p95"]))
        mshr_ratio = ratio(rsrc(cur_resource, ["demand_mshr_occ_p95", "pf_mshr_occ_p95", "mshr_occ_p95"]), rsrc(ref_resource, ["demand_mshr_occ_p95", "pf_mshr_occ_p95", "mshr_occ_p95"]))
        issued = max(metric(row, ["pf_issued", "issued", "replayer_emitted_candidates"]), 1.0)
        ref_issued = max(metric(ref, ["pf_issued", "issued", "replayer_emitted_candidates"]), 1.0)
        rejected_delta = metric(row, ["pf_rejected", "rejected", "pf_dropped"]) / issued - metric(ref, ["pf_rejected", "rejected", "pf_dropped"]) / ref_issued
        duplicate_delta = metric(row, ["pf_duplicate", "duplicate", "duplicates"]) / issued - metric(ref, ["pf_duplicate", "duplicate", "duplicates"]) / ref_issued
        failures = []
        if not keyed_ok(row): failures.append("candidate keyed replay invalid or failed")
        if not keyed_ok(ref): failures.append("selected-full keyed replay invalid or failed")
        for label, value, limit, op in [
            ("IPC ratio", ipc_ratio, num(criteria["ipc_ratio_min"]), "min"),
            ("selected-accuracy ratio", acc_ratio, num(criteria["accuracy_ratio_min"]), "min"),
            ("coverage ratio", cov_ratio, num(criteria["coverage_ratio_min"]), "min"),
            ("timeliness ratio", time_ratio, num(criteria["timeliness_ratio_min"]), "min"),
            ("issue ratio", issue_ratio, num(criteria["issue_ratio_max"]), "max"),
        ]:
            if not math.isfinite(value) or (op == "min" and value < limit) or (op == "max" and value > limit):
                failures.append("{} {:.6f} {} {:.6f}".format(label, value, "<" if op == "min" else ">", limit))
        for label, value, limit in [("PQ p95 ratio", pq_ratio, criteria.get("pq_p95_ratio_max")), ("MSHR p95 ratio", mshr_ratio, criteria.get("mshr_p95_ratio_max"))]:
            if limit is not None and math.isfinite(value) and value > num(limit):
                failures.append("{} {:.6f} > {:.6f}".format(label, value, num(limit)))
        if rejected_delta > num(criteria["rejected_delta_max"]): failures.append("rejected-rate delta {:.6f} > {:.6f}".format(rejected_delta, num(criteria["rejected_delta_max"])))
        if duplicate_delta > num(criteria["duplicate_delta_max"]): failures.append("duplicate-rate delta {:.6f} > {:.6f}".format(duplicate_delta, num(criteria["duplicate_delta_max"])))
        copied.update({
            "selected_full_reference_tag": ref_tag, "selected_full_ipc": num(ref.get("replay_ipc", ref.get("ipc"))),
            "ipc_ratio_vs_selected_full": ipc_ratio, "selected_accuracy_ratio_vs_selected_full": acc_ratio,
            "coverage_ratio_vs_selected_full": cov_ratio, "timeliness_ratio_vs_selected_full": time_ratio,
            "issue_ratio_vs_selected_full": issue_ratio,
            "pq_p95_ratio_vs_selected_full": pq_ratio if math.isfinite(pq_ratio) else "",
            "mshr_p95_ratio_vs_selected_full": mshr_ratio if math.isfinite(mshr_ratio) else "",
            "rejected_rate_delta_vs_selected_full": rejected_delta, "duplicate_rate_delta_vs_selected_full": duplicate_delta,
            "accepted": int(not failures), "failure_reason": "; ".join(failures),
        })
        out.append(copied)
    return out


def final_five(rows, normals, stage_rows):
    accepted = {r.get("candidate_tag") for r in stage_rows if integer(r.get("accepted"))}
    candidates = defaultdict(list)
    for row in rows:
        stage = row.get("stage", row.get("phase", ""))
        if not keyed_ok(row) or stage == "selected_full_reference":
            continue
        if stage == "stage_b" and row.get("candidate_tag") not in accepted:
            continue
        candidates[row.get("trace", "")].append(row)
    out = []
    for trace, ref in sorted(normals.items()):
        values = candidates.get(trace, [])
        if not values:
            out.append({"trace": trace, "winner_status": "no_valid_v4_8_keyed_candidate", "best_normal_prefetcher": ref["best"].get("prefetcher", ""), "best_normal_ipc_fixed_matrix": num(ref["best"].get("ipc"))})
            continue
        winner = max(values, key=lambda r: num(r.get("replay_ipc", r.get("ipc"))))
        copied = dict(winner)
        ipc = num(winner.get("replay_ipc", winner.get("ipc")))
        copied.update({
            "winner_status": "max_valid_keyed_replay_ipc_among_eligible_v4_8_candidates",
            "no_pref_ipc_fixed_matrix": num(ref["no_pref"].get("ipc")),
            "best_normal_prefetcher": ref["best"].get("prefetcher", ""),
            "best_normal_ipc_fixed_matrix": num(ref["best"].get("ipc")),
            "ipc_delta_vs_no_pref_fixed_matrix": ipc - num(ref["no_pref"].get("ipc")),
            "ipc_delta_vs_best_normal_fixed_matrix": ipc - num(ref["best"].get("ipc")),
            "beats_best_normal_by_keyed_replay": int(ipc > num(ref["best"].get("ipc"))),
        })
        out.append(copied)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-summary", required=True, type=Path)
    parser.add_argument("--normal-summary", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resource-summary", type=Path, default=None)
    parser.add_argument("--attribution-summary", type=Path, default=None)
    parser.add_argument("--criteria", default="")
    args = parser.parse_args()
    criteria = load_criteria(args.criteria)
    merged = merge(read_csv(args.metadata), read_csv(args.replay_summary), read_csv(args.attribution_summary) if args.attribution_summary else [])
    normals = normal_by_trace(read_csv(args.normal_summary))
    final = enrich(merged, normals)
    resources = resource_index(read_csv(args.resource_summary) if args.resource_summary else [])
    stage = stage_b(final, resources, criteria)
    decisions = []
    for row in final:
        if row.get("stage", row.get("phase", "")) == "stage_b":
            continue
        coverage = metric(row, ["nn_event_coverage", "unique_event_coverage", "coverage"])
        timing = metric(row, ["nn_event_timeliness", "event_timeliness_over_covered", "timeliness"])
        focus = "candidate-bank recall" if coverage and coverage < 0.50 else ("timing" if timing and timing < 0.90 else "ranking/calibration or cache/resource behavior")
        decisions.append({"trace": row.get("trace", ""), "candidate_tag": row.get("candidate_tag", ""), "stage": row.get("stage", row.get("phase", "")), "replay_ipc": row.get("replay_ipc", row.get("ipc", "")), "ipc_delta_vs_best_normal_fixed_matrix": row.get("ipc_delta_vs_best_normal_fixed_matrix", ""), "next_focus": focus})
    write_csv(args.out_dir / "v4_8_all_candidate_replay_comparison.csv", final)
    write_csv(args.out_dir / "v4_8_nn_vs_every_normal_ipc.csv", all_normal_rows(final, normals))
    write_csv(args.out_dir / "v4_8_route_seed_variance.csv", seed_variance(final))
    write_csv(args.out_dir / "v4_8_stage_b_acceptance.csv", stage)
    accepted = [r for r in stage if integer(r.get("accepted"))]
    smallest = []
    by_trace = defaultdict(list)
    for row in accepted: by_trace[row.get("trace", "")].append(row)
    for trace, values in sorted(by_trace.items()): smallest.append(min(values, key=lambda r: integer(r.get("parameters"), 10 ** 30)))
    write_csv(args.out_dir / "v4_8_smallest_accepted_model_by_trace.csv", smallest)
    write_csv(args.out_dir / "v4_8_causal_route_decisions.csv", decisions)
    write_csv(args.out_dir / "v4_8_final_five_trace_comparison.csv", final_five(final, normals, stage))
    print("[write] {}".format(args.out_dir))


if __name__ == "__main__":
    main()
