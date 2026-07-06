#!/usr/bin/env python3
"""General standard-library V4.8 replay analysis.

Joins notebook metadata with keyed replay, completed normal baselines, event
attribution, and resource summaries. Offline metrics are diagnostic only: an
NN is marked as beating a normal prefetcher only after valid keyed replay.
"""
from __future__ import print_function
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path


def num(x, default=0.0):
    try:
        return float(x) if x not in (None, "") else default
    except (TypeError, ValueError):
        return default


def integer(x, default=0):
    try:
        return int(float(x)) if x not in (None, "") else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader(); writer.writerows(rows)


def metric(row, names, default=0.0):
    for name in names:
        if row.get(name) not in (None, ""):
            return num(row[name], default)
    return default


def keyed_ok(row):
    return integer(row.get("replay_transport_ok"), 0) == 1 and integer(row.get("run_failed"), 1) == 0


def criteria(value):
    result = {"ipc_ratio_min": .995, "accuracy_ratio_min": .95, "coverage_ratio_min": .95,
              "timeliness_ratio_min": .95, "issue_ratio_max": 1.10,
              "pq_p95_ratio_max": 1.10, "mshr_p95_ratio_max": 1.10,
              "rejected_delta_max": .05, "duplicate_delta_max": .05}
    if value:
        source = Path(value)
        result.update(json.loads(source.read_text()) if source.is_file() else json.loads(value))
    return result


def normal_by_trace(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("trace") and not integer(row.get("run_failed"), 0):
            groups[row["trace"]].append(row)
    out = {}
    for trace, values in groups.items():
        no_pref = next((r for r in values if r.get("prefetcher") in ("no_pref", "none", "nopref")), {})
        normal = [r for r in values if r.get("prefetcher") not in ("no_pref", "none", "nopref")]
        out[trace] = {"all": values, "no_pref": no_pref,
                      "best": max(normal, key=lambda r: num(r.get("ipc"))) if normal else {}}
    return out


def auxiliary_index(rows, family):
    out = {}
    for row in rows:
        if row.get("family") != family:
            continue
        tag = row.get("variant") or row.get("candidate_tag") or row.get("standalone_variant")
        if tag:
            out[(row.get("trace", ""), tag)] = row
    return out


def merge(metadata, replay, attrs, resources):
    meta = {r.get("tag") or r.get("candidate_tag"): r for r in metadata if r.get("tag") or r.get("candidate_tag")}
    rows = []
    for replay_row in replay:
        tag = replay_row.get("candidate_tag") or replay_row.get("tag") or replay_row.get("standalone_variant")
        row = dict(meta.get(tag, {})); row.update(replay_row); row["candidate_tag"] = tag
        attr = attrs.get((row.get("trace", ""), tag), {})
        resource = resources.get((row.get("trace", ""), tag), {})
        row.update({
            "event_coverage": attr.get("unique_event_coverage", ""),
            "event_timeliness": attr.get("timeliness_over_covered", ""),
            "event_timely": attr.get("timely_events", ""),
            "event_late": attr.get("late_events", ""),
            "event_residual": attr.get("residual_events", ""),
            "resource_issue_per_l2_load": resource.get("prefetch_attempts_per_l2_load", ""),
            "resource_rejected_fraction": resource.get("prefetch_reject_fraction", ""),
            "resource_duplicate_fraction": resource.get("prefetch_duplicate_fraction", ""),
            "resource_pq_p95": resource.get("pf_pq_occ_p95", resource.get("demand_pq_occ_p95", "")),
            "resource_mshr_p95": resource.get("pf_mshr_occ_p95", resource.get("demand_mshr_occ_p95", "")),
            "resource_accepted": resource.get("prefetch_accepted", ""),
            "resource_attempts": resource.get("prefetch_attempts", ""),
            "resource_duplicate": resource.get("prefetch_duplicate", ""),
        })
        rows.append(row)
    return rows


def enrich(rows, normals):
    out = []
    for row in rows:
        ref = normals.get(row.get("trace", ""), {})
        best, no_pref = ref.get("best", {}), ref.get("no_pref", {})
        ipc = metric(row, ["replay_ipc", "ipc"])
        copied = dict(row)
        copied.update({
            "keyed_replay_valid": int(keyed_ok(row)), "replay_ipc": ipc,
            "fixed_no_pref_ipc": num(no_pref.get("ipc")),
            "best_normal_prefetcher": best.get("prefetcher", ""),
            "best_normal_ipc": num(best.get("ipc")),
            "ipc_delta_vs_no_pref": ipc - num(no_pref.get("ipc")),
            "ipc_delta_vs_best_normal": ipc - num(best.get("ipc")),
            "beats_best_normal_by_keyed_replay": int(keyed_ok(row) and ipc > num(best.get("ipc"))),
        })
        out.append(copied)
    return out


def every_normal(rows, normals):
    out = []
    for row in rows:
        for normal in normals.get(row.get("trace", ""), {}).get("all", []):
            copied = dict(row)
            copied.update(normal_prefetcher=normal.get("prefetcher", ""), normal_ipc=num(normal.get("ipc")),
                          ipc_delta_nn_minus_normal=metric(row,["replay_ipc","ipc"]) - num(normal.get("ipc")),
                          nn_beats_this_normal_by_keyed_replay=int(keyed_ok(row) and metric(row,["replay_ipc","ipc"]) > num(normal.get("ipc"))))
            out.append(copied)
    return out


def variance(rows):
    groups = defaultdict(list)
    for row in rows:
        if keyed_ok(row):
            key = (row.get("trace", ""), row.get("route_id", row.get("bank_id", "")), row.get("policy_tag", ""), row.get("model_size", ""), row.get("stage", ""))
            groups[key].append(row)
    out = []
    for key, values in sorted(groups.items()):
        ipcs = [metric(v,["replay_ipc","ipc"]) for v in values]
        out.append(dict(trace=key[0], route_id=key[1], policy_tag=key[2], model_size=key[3], stage=key[4],
                        keyed_replay_count=len(ipcs), replay_ipc_mean=statistics.mean(ipcs), replay_ipc_min=min(ipcs),
                        replay_ipc_max=max(ipcs), replay_ipc_stdev=statistics.pstdev(ipcs) if len(ipcs)>1 else 0.0,
                        seeds=";".join(str(v.get("seed", "")) for v in values)))
    return out


def checked_ratio(failures, label, current, reference, limit, direction):
    if not reference:
        failures.append("missing selected-full metric for " + label); return float("nan")
    value = current / reference
    if not math.isfinite(value):
        failures.append("non-finite ratio for " + label)
    elif direction == "min" and value < limit:
        failures.append("{} {:.6f} < {:.6f}".format(label, value, limit))
    elif direction == "max" and value > limit:
        failures.append("{} {:.6f} > {:.6f}".format(label, value, limit))
    return value


def stage_b(rows, gates):
    by_tag = {r.get("candidate_tag"): r for r in rows if r.get("candidate_tag")}
    out = []
    for row in rows:
        if row.get("stage") != "stage_b":
            continue
        copied, failures = dict(row), []
        reference = by_tag.get(row.get("selected_full_reference_tag", ""))
        if not reference:
            copied.update(accepted=0, failure_reason="missing selected-full keyed replay reference")
            out.append(copied); continue
        if not keyed_ok(row): failures.append("candidate keyed replay invalid or failed")
        if not keyed_ok(reference): failures.append("selected-full keyed replay invalid or failed")
        ipc = checked_ratio(failures, "IPC ratio", metric(row,["replay_ipc","ipc"]), metric(reference,["replay_ipc","ipc"]), num(gates["ipc_ratio_min"]), "min")
        acc = checked_ratio(failures, "selected accuracy ratio", metric(row,["selected_accuracy","accuracy"]), metric(reference,["selected_accuracy","accuracy"]), num(gates["accuracy_ratio_min"]), "min")
        cov = checked_ratio(failures, "coverage ratio", metric(row,["event_coverage","coverage","unique_event_coverage"]), metric(reference,["event_coverage","coverage","unique_event_coverage"]), num(gates["coverage_ratio_min"]), "min")
        timely = checked_ratio(failures, "timeliness ratio", metric(row,["event_timeliness","timeliness","event_timeliness_over_covered"]), metric(reference,["event_timeliness","timeliness","event_timeliness_over_covered"]), num(gates["timeliness_ratio_min"]), "min")
        issue = checked_ratio(failures, "issue ratio", metric(row,["resource_issue_per_l2_load","issue_per_event","val_policy_issue_per_event"]), metric(reference,["resource_issue_per_l2_load","issue_per_event","val_policy_issue_per_event"]), num(gates["issue_ratio_max"]), "max")
        # Missing queue data is a failure, not a pass.
        pq = checked_ratio(failures, "PQ p95 ratio", metric(row,["resource_pq_p95"]), metric(reference,["resource_pq_p95"]), num(gates["pq_p95_ratio_max"]), "max")
        mshr = checked_ratio(failures, "MSHR p95 ratio", metric(row,["resource_mshr_p95"]), metric(reference,["resource_mshr_p95"]), num(gates["mshr_p95_ratio_max"]), "max")
        rejected = metric(row,["resource_rejected_fraction"]) - metric(reference,["resource_rejected_fraction"])
        duplicate = metric(row,["resource_duplicate_fraction"]) - metric(reference,["resource_duplicate_fraction"])
        if rejected > num(gates["rejected_delta_max"]): failures.append("rejected fraction delta {:.6f} > {:.6f}".format(rejected,num(gates["rejected_delta_max"])))
        if duplicate > num(gates["duplicate_delta_max"]): failures.append("duplicate fraction delta {:.6f} > {:.6f}".format(duplicate,num(gates["duplicate_delta_max"])))
        copied.update(selected_full_ipc=metric(reference,["replay_ipc","ipc"]), ipc_ratio_vs_selected_full=ipc,
                      selected_accuracy_ratio_vs_selected_full=acc, coverage_ratio_vs_selected_full=cov,
                      timeliness_ratio_vs_selected_full=timely, issue_ratio_vs_selected_full=issue,
                      pq_p95_ratio_vs_selected_full=pq, mshr_p95_ratio_vs_selected_full=mshr,
                      rejected_rate_delta_vs_selected_full=rejected, duplicate_rate_delta_vs_selected_full=duplicate,
                      accepted=int(not failures), failure_reason="; ".join(failures))
        out.append(copied)
    return out


def decisions(rows):
    out=[]
    for row in rows:
        recall = metric(row,["candidate_recall_before_policy"])
        rank4 = metric(row,["correct_rank_recall_top4"])
        coverage = metric(row,["event_coverage","coverage","unique_event_coverage"])
        timely = metric(row,["event_timeliness","timeliness","event_timeliness_over_covered"])
        if recall and recall < .50: focus="candidate-bank recall / representation"
        elif rank4 and rank4 < .80: focus="ranking / calibration"
        elif timely and timely < .90: focus="timing / lead policy"
        elif coverage and coverage < .50: focus="policy / dedup / rate-limit"
        elif metric(row,["ipc_delta_vs_best_normal"]) < 0 and (metric(row,["resource_pq_p95"]) or metric(row,["resource_mshr_p95"])): focus="cache pollution or PQ/MSHR resource behavior"
        else: focus="replay-confirmed route comparison; do not infer cause from offline scores alone"
        out.append(dict(trace=row.get("trace", ""), candidate_tag=row.get("candidate_tag", ""), route_id=row.get("route_id", ""), stage=row.get("stage", ""),
                        replay_ipc=metric(row,["replay_ipc","ipc"]), ipc_delta_vs_best_normal=metric(row,["ipc_delta_vs_best_normal"]),
                        candidate_recall_before_policy=recall, rank_recall_top4=rank4, event_coverage=coverage,
                        event_timeliness=timely, resource_pq_p95=metric(row,["resource_pq_p95"]), resource_mshr_p95=metric(row,["resource_mshr_p95"]), next_focus=focus))
    return out


def final_five(rows, normals, stages):
    accepted = {r.get("candidate_tag") for r in stages if integer(r.get("accepted"))}
    candidates = defaultdict(list)
    for row in rows:
        if keyed_ok(row) and (row.get("stage") != "stage_b" or row.get("candidate_tag") in accepted):
            candidates[row.get("trace", "")].append(row)
    out=[]
    for trace, ref in sorted(normals.items()):
        values = candidates.get(trace, [])
        if not values:
            out.append(dict(trace=trace, winner_status="no_valid_v4_8_keyed_candidate", best_normal_prefetcher=ref["best"].get("prefetcher", ""), best_normal_ipc=num(ref["best"].get("ipc"))))
            continue
        winner=max(values,key=lambda r: metric(r,["replay_ipc","ipc"]))
        copied=dict(winner); copied.update(winner_status="max_valid_keyed_replay_ipc_among_eligible_v4_8_candidates",
            best_normal_prefetcher=ref["best"].get("prefetcher", ""), best_normal_ipc=num(ref["best"].get("ipc")), fixed_no_pref_ipc=num(ref["no_pref"].get("ipc")),
            ipc_delta_vs_best_normal=metric(winner,["replay_ipc","ipc"])-num(ref["best"].get("ipc")),
            beats_best_normal_by_keyed_replay=int(metric(winner,["replay_ipc","ipc"]) > num(ref["best"].get("ipc"))))
        out.append(copied)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replay-summary",required=True,type=Path); parser.add_argument("--normal-summary",required=True,type=Path)
    parser.add_argument("--metadata",required=True,type=Path); parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--resource-summary",type=Path); parser.add_argument("--attribution-summary",type=Path); parser.add_argument("--criteria",default="")
    args=parser.parse_args()
    normals=normal_by_trace(read_csv(args.normal_summary))
    attrs=auxiliary_index(read_csv(args.attribution_summary),"standalone") if args.attribution_summary else {}
    resources=auxiliary_index(read_csv(args.resource_summary),"standalone") if args.resource_summary else {}
    rows=enrich(merge(read_csv(args.metadata),read_csv(args.replay_summary),attrs,resources),normals)
    gates=criteria(args.criteria); stages=stage_b(rows,gates)
    write_csv(args.out_dir / "v4_8_all_candidate_replay_comparison.csv",rows)
    write_csv(args.out_dir / "v4_8_nn_vs_every_normal_ipc.csv",every_normal(rows,normals))
    write_csv(args.out_dir / "v4_8_route_seed_variance.csv",variance(rows))
    write_csv(args.out_dir / "v4_8_stage_b_acceptance.csv",stages)
    accepted=defaultdict(list)
    for row in stages:
        if integer(row.get("accepted")): accepted[row.get("trace", "")].append(row)
    write_csv(args.out_dir / "v4_8_smallest_accepted_model_by_trace.csv",[min(v,key=lambda r: integer(r.get("parameters"),10**30)) for _,v in sorted(accepted.items())])
    write_csv(args.out_dir / "v4_8_causal_route_decisions.csv",decisions(rows))
    write_csv(args.out_dir / "v4_8_final_five_trace_comparison.csv",final_five(rows,normals,stages))
    print("[write] {}".format(args.out_dir))

if __name__ == "__main__":
    main()
