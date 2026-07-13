#!/usr/bin/env python3
"""Replay-gated V4.8 analysis using only Python's standard library.

The program is intentionally configuration-driven. It has no trace IDs, PC values,
deltas, local paths, or experiment-tag-specific rules. It consumes metadata produced
by the Colab notebook plus keyed replay, normal baseline, event-attribution, and
resource summaries. Offline metrics are diagnostic only: an NN is marked as beating
a normal prefetcher only when its keyed replay is valid.
"""
from __future__ import print_function
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def f(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def i(value, default=0):
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        if fields:
            w.writeheader()
            w.writerows(rows)


def choose(row, names, default=0.0):
    for name in names:
        if row.get(name) not in (None, ""):
            return f(row[name], default)
    return default


def valid_keyed(row):
    # Missing transport evidence is not proof of a valid replay.
    return i(row.get("replay_transport_ok"), 0) == 1 and i(row.get("run_failed"), 1) == 0


def load_criteria(value):
    base = {
        "ipc_ratio_min": .995,
        "accuracy_ratio_min": .95,
        "coverage_ratio_min": .95,
        "timeliness_ratio_min": .95,
        "issue_ratio_max": 1.10,
        "pq_p95_ratio_max": 1.10,
        "mshr_p95_ratio_max": 1.10,
        "rejected_delta_max": .05,
        "duplicate_delta_max": .05,
    }
    if not value:
        return base
    p = Path(value)
    loaded = json.loads(p.read_text()) if p.is_file() else json.loads(value)
    base.update(loaded)
    return base


def normal_index(rows):
    out = defaultdict(list)
    for row in rows:
        if row.get("trace") and not i(row.get("run_failed"), 0):
            out[row["trace"]].append(row)
    final = {}
    for trace, values in out.items():
        no_pref = next((r for r in values if r.get("prefetcher") in ("no_pref", "none", "nopref")), {})
        normal = [r for r in values if r.get("prefetcher") not in ("no_pref", "none", "nopref")]
        final[trace] = dict(no_pref=no_pref, all=values, best=max(normal, key=lambda r: f(r.get("ipc"))) if normal else {})
    return final


def attribution_index(rows):
    out = {}
    for row in rows:
        if row.get("family") != "standalone":
            continue
        tag = row.get("variant") or row.get("candidate_tag") or row.get("standalone_variant")
        if tag:
            out[(row.get("trace", ""), tag)] = row
    return out


def resource_index(rows):
    out = {}
    for row in rows:
        if row.get("family") != "standalone":
            continue
        tag = row.get("variant") or row.get("candidate_tag") or row.get("standalone_variant")
        if tag:
            out[(row.get("trace", ""), tag)] = row
    return out


def merge(metadata, replay, attrs, resources):
    meta = {r.get("tag") or r.get("candidate_tag"): r for r in metadata if r.get("tag") or r.get("candidate_tag")}
    merged = []
    for replay_row in replay:
        tag = replay_row.get("candidate_tag") or replay_row.get("tag") or replay_row.get("standalone_variant")
        row = dict(meta.get(tag, {}))
        row.update(replay_row)
        row["candidate_tag"] = tag
        key = (row.get("trace", ""), tag)
        attr, rsrc = attrs.get(key, {}), resources.get(key, {})
        row.update({
            "event_coverage": attr.get("unique_event_coverage", ""),
            "event_timeliness": attr.get("timeliness_over_covered", ""),
            "event_timely": attr.get("timely_events", ""),
            "event_late": attr.get("late_events", ""),
            "event_residual": attr.get("residual_events", ""),
            "resource_prefetch_attempts_per_l2_load": rsrc.get("prefetch_attempts_per_l2_load", ""),
            "resource_rejected_fraction": rsrc.get("prefetch_reject_fraction", ""),
            "resource_duplicate_fraction": rsrc.get("prefetch_duplicate_fraction", ""),
            "resource_pq_p95": rsrc.get("pf_pq_occ_p95", rsrc.get("demand_pq_occ_p95", "")),
            "resource_mshr_p95": rsrc.get("pf_mshr_occ_p95", rsrc.get("demand_mshr_occ_p95", "")),
            "resource_accepted": rsrc.get("prefetch_accepted", ""),
            "resource_attempts": rsrc.get("prefetch_attempts", ""),
            "resource_duplicate": rsrc.get("prefetch_duplicate", ""),
        })
        merged.append(row)
    return merged


def enrich(rows, normals):
    output = []
    for row in rows:
        ref = normals.get(row.get("trace", ""), {})
        best, no_pref = ref.get("best", {}), ref.get("no_pref", {})
        ipc = choose(row, ["replay_ipc", "ipc"])
        x = dict(row)
        x.update({
            "keyed_replay_valid": int(valid_keyed(row)),
            "replay_ipc": ipc,
            "fixed_no_pref_ipc": f(no_pref.get("ipc")),
            "best_normal_prefetcher": best.get("prefetcher", ""),
            "best_normal_ipc": f(best.get("ipc")),
            "ipc_delta_vs_no_pref": ipc - f(no_pref.get("ipc")),
            "ipc_delta_vs_best_normal": ipc - f(best.get("ipc")),
            "beats_best_normal_by_keyed_replay": int(valid_keyed(row) and ipc > f(best.get("ipc"))),
        })
        output.append(x)
    return output


def all_normal_rows(rows, normals):
    out = []
    for row in rows:
        for normal in normals.get(row.get("trace", ""), {}).get("all", []):
            x = dict(row)
            x.update(normal_prefetcher=normal.get("prefetcher", ""), normal_ipc=f(normal.get("ipc")),
                     ipc_delta_nn_minus_normal=choose(row,["replay_ipc","ipc"])-f(normal.get("ipc")),
                     nn_beats_this_normal_by_keyed_replay=int(valid_keyed(row) and choose(row,["replay_ipc","ipc"]) > f(normal.get("ipc"))))
            out.append(x)
    return out


def seed_variance(rows):
    groups = defaultdict(list)
    for row in rows:
        if valid_keyed(row):
            key = (row.get("trace", ""), row.get("route_id", row.get("bank_id", "")), row.get("policy_tag", ""), row.get("model_size", ""), row.get("stage", ""))
            groups[key].append(row)
    out = []
    for key, values in sorted(groups.items()):
        ipcs = [choose(v,["replay_ipc","ipc"]) for v in values]
        out.append(dict(trace=key[0],route_id=key[1],policy_tag=key[2],model_size=key[3],stage=key[4],
                        keyed_replay_count=len(ipcs),replay_ipc_mean=statistics.mean(ipcs),replay_ipc_min=min(ipcs),replay_ipc_max=max(ipcs),
                        replay_ipc_stdev=statistics.pstdev(ipcs) if len(ipcs)>1 else 0.0,
                        seeds=";".join(str(v.get("seed", "")) for v in values)))
    return out


def ratio(num, den):
    return float("nan") if not den else num / den


def require_ratio(failures, label, cur, ref, limit, mode):
    value = ratio(cur, ref)
    if not math.isfinite(value):
        failures.append("missing or zero denominator for " + label)
    elif (mode == "min" and value < limit) or (mode == "max" and value > limit):
        failures.append("{} {:.6f} {} {:.6f}".format(label, value, "<" if mode=="min" else ">", limit))
    return value


def stage_b(rows, criteria):
    by_tag = {r.get("candidate_tag"): r for r in rows if r.get("candidate_tag")}
    output = []
    for row in rows:
        if row.get("stage") != "stage_b":
            continue
        copied = dict(row)
        ref_tag = row.get("selected_full_reference_tag", "")
        ref = by_tag.get(ref_tag)
        failures = []
        if not ref:
            copied.update(accepted=0, failure_reason="missing selected-full keyed replay reference")
            output.append(copied); continue
        if not valid_keyed(row): failures.append("candidate keyed replay invalid or failed")
        if not valid_keyed(ref): failures.append("selected-full keyed replay invalid or failed")
        ipc = require_ratio(failures,"IPC ratio",choose(row,["replay_ipc","ipc"]),choose(ref,["replay_ipc","ipc"]),f(criteria["ipc_ratio_min"]),"min")
        acc = require_ratio(failures,"selected-accuracy ratio",choose(row,["selected_accuracy","accuracy"]),choose(ref,["selected_accuracy","accuracy"]),f(criteria["accuracy_ratio_min"]),"min")
        cov = require_ratio(failures,"coverage ratio",choose(row,["event_coverage","coverage","unique_event_coverage"]),choose(ref,["event_coverage","coverage","unique_event_coverage"]),f(criteria["coverage_ratio_min"]),"min")
        tim = require_ratio(failures,"timeliness ratio",choose(row,["event_timeliness","timeliness","event_timeliness_over_covered"]),choose(ref,["event_timeliness","timeliness","event_timeliness_over_covered"]),f(criteria["timeliness_ratio_min"]),"min")
        issue = require_ratio(failures,"issue ratio",choose(row,["resource_prefetch_attempts_per_l2_load","issue_per_event","val_policy_issue_per_event"]),choose(ref,["resource_prefetch_attempts_per_l2_load","issue_per_event","val_policy_issue_per_event"]),f(criteria["issue_ratio_max"]),"max")
        # Queue metrics are explicit gates: absent resource evidence fails rather than passing silently.
        pq = require_ratio(failures,"PQ p95 ratio",choose(row,["resource_pq_p95"]),choose(ref,["resource_pq_p95"]),f(criteria["pq_p95_ratio_max"]),"max")
        mshr = require_ratio(failures,"MSHR p95 ratio",choose(row,["resource_mshr_p95"]),choose(ref,["resource_mshr_p95"]),f(criteria["mshr_p95_ratio_max"]),"max")
        reject_delta = choose(row,["resource_rejected_fraction"]) - choose(ref,["resource_rejected_fraction"])
        duplicate_delta = choose(row,["resource_duplicate_fraction"]) - choose(ref,["resource_duplicate_fraction"])
        if reject_delta > f(criteria["rejected_delta_max"]): failures.append("rejected-rate delta {:.6f} > {:.6f}".format(reject_delta,f(criteria["rejected_delta_max"])))
        if duplicate_delta > f(criteria["duplicate_delta_max"]): failures.append("duplicate-rate delta {:.6f} > {:.6f}".format(duplicate_delta,f(criteria["duplicate_delta_max"])))
        copied.update(selected_full_reference_tag=ref_tag, selected_full_ipc=choose(ref,["replay_ipc","ipc"]),
                      ipc_ratio_vs_selected_full=ipc, selected_accuracy_ratio_vs_selected_full=acc,
                      coverage_ratio_vs_selected_full=cov, timeliness_ratio_vs_selected_full=tim,
                      issue_ratio_vs_selected_full=issue, pq_p95_ratio_vs_selected_full=pq,
                      mshr_p95_ratio_vs_selected_full=mshr, rejected_rate_delta_vs_selected_full=reject_delta,
                      duplicate_rate_delta_vs_selected_full=duplicate_delta, accepted=int(not failures),failure_reason="; ".join(failures))
        output.append(copied)
    return output


def causal_decisions(rows):
    out=[]
    for row in rows:
        rec=choose(row,["candidate_recall_before_policy"])
        rank4=choose(row,["correct_rank_recall_top4"])
        cov=choose(row,["event_coverage","coverage","unique_event_coverage"])
        tim=choose(row,["event_timeliness","timeliness","event_timeliness_over_covered"])
        pq=choose(row,["resource_pq_p95"])
        mshr=choose(row,["resource_mshr_p95"])
        if rec and rec < .50:
            focus="candidate-bank recall / representation"
        elif rank4 and rank4 < .80:
            focus="ranking / calibration"
        elif tim and tim < .90:
            focus="timing / lead policy"
        elif cov and cov < .50:
            focus="policy / dedup / rate-limit"
        elif (pq or mshr) and choose(row,["ipc_delta_vs_best_normal"]) < 0:
            focus="cache pollution or PQ/MSHR resource behavior"
        else:
            focus="replay-confirmed route comparison; do not infer a cause from offline scores alone"
        out.append(dict(trace=row.get("trace",""),candidate_tag=row.get("candidate_tag",""),stage=row.get("stage",""),route_id=row.get("route_id",""),
                        replay_ipc=choose(row,["replay_ipc","ipc"]),ipc_delta_vs_best_normal=choose(row,["ipc_delta_vs_best_normal"]),
                        candidate_recall_before_policy=rec,rank_recall_top4=rank4,event_coverage=cov,event_timeliness=tim,
                        resource_pq_p95=pq,resource_mshr_p95=mshr,next_focus=focus))
    return out


def final_five(rows, normals, stage_rows):
    accepted={r.get("candidate_tag") for r in stage_rows if i(r.get("accepted"))}
    groups=defaultdict(list)
    for row in rows:
        if not valid_keyed(row):
            continue
        stage=row.get("stage","")
        if stage == "stage_b" and row.get("candidate_tag") not in accepted:
            continue
        groups[row.get("trace","")].append(row)
    output=[]
    for trace, refs in sorted(normals.items()):
        candidates=groups.get(trace,[])
        if not candidates:
            output.append(dict(trace=trace,winner_status="no_valid_v4_8_keyed_candidate",best_normal_prefetcher=refs["best"].get("prefetcher",""),best_normal_ipc=f(refs["best"].get("ipc"))))
            continue
        win=max(candidates,key=lambda r:choose(r,["replay_ipc","ipc"]))
        x=dict(win); x.update(winner_status="max_valid_keyed_replay_ipc_among_eligible_v4_8_candidates",
                              best_normal_prefetcher=refs["best"].get("prefetcher",""),best_normal_ipc=f(refs["best"].get("ipc")),
                              fixed_no_pref_ipc=f(refs["no_pref"].get("ipc")),
                              ipc_delta_vs_best_normal=choose(win,["replay_ipc","ipc"])-f(refs["best"].get("ipc")),
                              beats_best_normal_by_keyed_replay=int(choose(win,["replay_ipc","ipc"]) > f(refs["best"].get("ipc"))))
        output.append(x)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replay-summary",required=True,type=Path)
    p.add_argument("--normal-summary",required=True,type=Path)
    p.add_argument("--metadata",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--resource-summary",type=Path)
    p.add_argument("--attribution-summary",type=Path)
    p.add_argument("--criteria",default="")
    a=p.parse_args(); criteria=load_criteria(a.criteria)
    attrs=attribution_index(read_csv(a.attribution_summary) if a.attribution_summary else [])
    resources=resource_index(read_csv(a.resource_summary) if a.resource_summary else [])
    rows=enrich(merge(read_csv(a.metadata),read_csv(a.replay_summary),attrs,resources),normal_index(read_csv(a.normal_summary)))
    normals=normal_index(read_csv(a.normal_summary)); stages=stage_b(rows,criteria)
    write_csv(a.out_dir/"v4_8_all_candidate_replay_comparison.csv",rows)
    write_csv(a.out_dir/"v4_8_nn_vs_every_normal_ipc.csv",all_normal_rows(rows,normals))
    write_csv(a.out_dir/"v4_8_route_seed_variance.csv",seed_variance(rows))
    write_csv(a.out_dir/"v4_8_stage_b_acceptance.csv",stages)
    accepted=defaultdict(list)
    for r in stages:
        if i(r.get("accepted")): accepted[r.get("trace","")].append(r)
    smallest=[]
    for trace, values in sorted(accepted.items()):
        smallest.append(min(values,key=lambda r:i(r.get("parameters"),10**30)))
    write_csv(a.out_dir/"v4_8_smallest_accepted_model_by_trace.csv",smallest)
    write_csv(a.out_dir/"v4_8_causal_route_decisions.csv",causal_decisions(rows))
    write_csv(a.out_dir/"v4_8_final_five_trace_comparison.csv",final_five(rows,normals,stages))
    print("[write] {}".format(a.out_dir))

if __name__ == "__main__":
    main()
