#!/usr/bin/env python3
"""Read-only full-ROI analysis for the separate four602 frozen-live protocol."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import html
import json
import math
import os
from pathlib import Path
import re
import struct

EXP = Path(__file__).resolve().parent
POLICIES = ("stride", "streamer", "ampm", "spp")
END_STATES = ("timely", "late", "unused", "redundant_cache", "redundant_inflight", "pending", "resident_unused")
COHORT_FIELDS = ("enqueued",) + END_STATES + ("timely_load", "late_load", "timely_rfo", "late_rfo")
PROFILE = re.compile(r"phase_(\d+)_(raw|payload|learned)_(.+)$")
MISSING = object()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def csv_out(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + "."))
        else:
            out[name] = item
    return out


def configure(out, raw=None, config=None):
    cache = next((p for p in out.resolve().parents if p.name == "cache"), None)
    if cache is None:
        raise ValueError("Output must be inside cache/formal_study_20261017")
    root = cache / "formal_study_20261017"
    if not out.resolve().is_relative_to(root.resolve()):
        raise ValueError("All new analysis outputs must stay inside " + str(root))
    runtime_name=Path((config or {}).get("runtime", "runtime_v2" if raw is not None and raw.name=="raw_v2" else "runtime")).name
    for key, name in (("TMPDIR", "tmp"), ("PYTHONPYCACHEPREFIX", "pycache"), ("MPLCONFIGDIR", "mpl")):
        folder = root / runtime_name / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


class Audit:
    def __init__(self):
        self.checks = []
        self.notes = []

    def check(self, condition, check_name, run_id=None, **details):
        self.checks.append(dict(pass_=bool(condition), check=check_name, run_id=run_id, **details))
        return bool(condition)

    @property
    def failures(self):
        return [c for c in self.checks if not c["pass_"]]


def expected_specs(config, selection, stages):
    rows = []
    def one(stage, roi, policy, method, mode="full", budget=0):
        start, end = config["validation_records"] if stage == "validation" else config["test_records"][roi]
        rows.append(dict(run_id="{}_r{}_{}_{}_{}_cap{}".format(stage, roi, policy, method, mode, budget),
                         protocol=config["protocol"], stage=stage, roi=roi, policy=policy, method=method,
                         mode=mode, budget=budget, skip=start-config["warmup_instructions"],
                         warmup=config["warmup_instructions"], instructions=end-start,
                         measurement_records=[start, end], seed=config["seed"], macs=0, page_policy=config["page_policy"]))
    if "validation" in stages:
        one("validation", 0, "none", "none")
        for p in POLICIES: one("validation", 0, p, "native")
        for p in POLICIES[1:]: one("validation", 0, p, "student")
    if "test" in stages and selection:
        for roi in range(len(config["test_records"])):
            one("test", roi, "none", "none")
            for cap in (0, selection["common_cap"]):
                for p in POLICIES: one("test", roi, p, "native", budget=cap)
                one("test", roi, "stride", "stride_student", budget=cap)
                for p in POLICIES[1:]: one("test", roi, p, "student", budget=cap)
            for p in POLICIES[1:]:
                one("test", roi, p, "student", "fixed_payload")
                one("test", roi, p, "template")
                one("test", roi, p, "student", "reset_state")
            for m in ("forward1", "forward2"): one("test", roi, "stride", "stride_"+m)
    return rows


def protocol_key(spec):
    contract = spec.get("contract", {})
    return tuple(json.dumps(spec.get(k, contract.get(k)), sort_keys=True) for k in
                 ("protocol", "stage", "roi", "skip", "warmup", "instructions", "page_policy", "build_id", "trace"))


def counters(snapshot):
    return snapshot.get("four602_counters", {})


def progress(snapshot, spec):
    c = counters(snapshot)
    # Four602 monitor's empty payload profiles are not the Stride engine's actions.
    completed = snapshot.get("nn_completed") if spec.get("method") == "stride_student" else c.get("completed_updates", 0)
    return dict(demand_callbacks=c.get("demand_callbacks", 0), fill_callbacks=c.get("fill_callbacks", 0),
                completed_updates=completed or 0,
                warmup_completed_updates=snapshot.get("warmup_completed_updates",0) if spec.get("method")=="stride_student" else c.get("warmup_completed_updates",0),
                warmup_demand_callbacks=c.get("warmup_demand_callbacks",0),warmup_fill_callbacks=c.get("warmup_fill_callbacks",0),
                roi_demand_callbacks=c.get("demand_callbacks",0),roi_fill_callbacks=c.get("fill_callbacks",0),roi_completed_updates=completed or 0,
                inference_macs=snapshot.get("inference_macs", 0) if spec.get("method") == "stride_student" else c.get("inference_macs", 0),
                inference_nonlinears=snapshot.get("inference_nonlinears", 0) if spec.get("method") == "stride_student" else c.get("inference_nonlinears", 0))


def cohort(snapshot, target):
    if target == "l2":
        return {k: snapshot.get("cohort_"+k) for k in COHORT_FIELDS}
    return {k: snapshot.get("llc_target_cohort", {}).get(k) for k in COHORT_FIELDS}


def check_cohort(audit, c, target, run_id):
    available = all(isinstance(c.get(k), int) and c[k] >= 0 for k in COHORT_FIELDS)
    audit.check(available, target+"_cohort_fields_present_nonnegative", run_id)
    if available:
        audit.check(c["enqueued"] == sum(c[k] for k in END_STATES), target+"_lifecycle_conservation", run_id,
                    enqueued=c["enqueued"], outcomes=sum(c[k] for k in END_STATES))
        audit.check(c["timely_load"]+c["timely_rfo"] <= c["timely"] and c["late_load"]+c["late_rfo"] <= c["late"],
                    target+"_LOAD_RFO_subsets", run_id)
    return available


def stride_profile_rows(snapshot, spec, audit):
    rows=[];entries=defaultdict(dict)
    for name,hist in snapshot.get("mechanism_distributions_exact",{}).items():
        match=PROFILE.fullmatch(name)
        if not match:
            audit.check(False,"unknown_Stride_profile_name",spec["run_id"],profile_name=name)
            continue
        phase,stream,stat=match.groups();phase=int(phase)
        entries[phase][(stream,stat)]=hist
    for phase,profiles in entries.items():
        for stream in ("raw","payload"):
            deltas=Counter()
            for (st,stat),hist in list(profiles.items()):
                if st==stream and stat.startswith("delta_slot_"):deltas.update(hist)
            if deltas:
                profiles[(stream,"delta")]=dict(deltas)
                profiles[(stream,"fill")]={"2":sum(deltas.values())}
        for (stream,stat),hist in profiles.items():
            total=sum(hist.values())
            for value,count in sorted(hist.items(),key=lambda p:int(p[0])):
                rows.append(dict(run_id=spec["run_id"],stage=spec["stage"],roi=spec["roi"],policy=spec["policy"],
                    method=spec["method"],mode=spec["mode"],seed=spec.get("seed"),budget=spec["budget"],
                    phase=phase,stream=stream,statistic=stat,value=int(value),count=count,within_histogram_fraction=ratio(count,total),
                    scope="Stride_runtime_exact; raw_K_before_existing_decoder_cap; delta_after_decode; fill_fixed_L2_interface"))
    for stream in ("raw","payload"):
        kh=Counter();slots=0
        for profiles in entries.values():
            kh.update(profiles.get((stream,"K"),{}))
            slots+=sum(profiles.get((stream,"delta"),{}).values())
        if kh:
            desired=sum(int(k)*v for k,v in kh.items())
            if stream=="raw":
                audit.check(sum(kh.values())==snapshot.get("nn_completed",0),"Stride_raw_decisions_completed",spec["run_id"])
                audit.check(desired==slots+snapshot.get("decoder_dropped_addresses",0),"Stride_raw_K_decoded_plus_dropped",spec["run_id"])
            else:
                audit.check(desired==slots,"Stride_payload_K_slot_conservation",spec["run_id"])
            audit.check(slots==snapshot.get("mechanism_"+stream+"_generated_addresses",0),
                        "Stride_profile_generated_conservation",spec["run_id"],stream=stream)
    return rows


def profile_rows(snapshot, spec, audit):
    rows = []
    if spec["method"].startswith("stride_"):
        # Never mislabel four602 monitor K=0 as Stride silence.
        return stride_profile_rows(snapshot,spec,audit)
    distributions = snapshot.get("four602_distributions", {})
    by_phase = defaultdict(dict)
    for name, hist in distributions.items():
        match = PROFILE.fullmatch(name)
        if not match:
            audit.check(False, "unknown_four_profile_name", spec["run_id"], name=name)
            continue
        phase, stream, statistic = match.groups()
        n = sum(hist.values())
        audit.check(all(isinstance(v, int) and v >= 0 for v in hist.values()), "profile_counts_nonnegative", spec["run_id"], name=name)
        by_phase[int(phase)][(stream, statistic)] = hist
        for value, count in sorted(hist.items(), key=lambda p: int(p[0])):
            rows.append(dict(run_id=spec["run_id"], stage=spec["stage"], roi=spec["roi"], policy=spec["policy"],
                             method=spec["method"], mode=spec["mode"], seed=spec.get("seed"), budget=spec["budget"],
                             phase=int(phase), stream=stream, statistic=statistic, value=int(value), count=count,
                             within_histogram_fraction=ratio(count, n), scope="full_recorded_phase"))
    for phase, entries in by_phase.items():
        for stream in ("raw", "payload"):
            kh = entries.get((stream, "K"), {})
            eh = entries.get((stream, "emit"), {})
            if not kh and not eh: continue
            decisions = sum(kh.values())
            actions = sum(int(k)*v for k,v in kh.items())
            audit.check(decisions == sum(eh.values()) and eh.get("1", 0) == sum(v for k,v in kh.items() if int(k) > 0),
                        "profile_K_emit_conservation", spec["run_id"], phase=phase, stream=stream)
            for statistic in ("delta", "fill"):
                audit.check(actions == sum(entries.get((stream, statistic), {}).values()),
                            "profile_K_"+statistic+"_conservation", spec["run_id"], phase=phase, stream=stream)
            for slot in range(max([int(k) for k in kh] + [0])):
                expected = sum(v for k,v in kh.items() if int(k) > slot)
                for statistic in ("delta_slot_", "fill_slot_"):
                    audit.check(sum(entries.get((stream, statistic+str(slot)), {}).values()) == expected,
                                "profile_slot_conservation", spec["run_id"], phase=phase, stream=stream, slot=slot, statistic=statistic)
    for stream in ("raw", "payload"):
        kh = {phase: entries.get((stream, "K"), {}) for phase, entries in by_phase.items()}
        decisions = sum(sum(h.values()) for h in kh.values())
        actions = sum(sum(int(k)*v for k,v in h.items()) for h in kh.values())
        if kh and spec["method"] != "none":
            audit.check(decisions == counters(snapshot).get("demand_callbacks", 0),
                        "profile_decisions_equal_demand_callbacks", spec["run_id"], stream=stream, decisions=decisions)
        audit.check(actions == counters(snapshot).get(stream+"_generated", 0),
                    "profile_generated_counter_conservation", spec["run_id"], stream=stream)
    return rows


def classification_rows(snapshot, spec, audit):
    rows = []
    classes = snapshot.get("mechanism_classified_requests", [])
    totals = Counter()
    cohort_totals = {target:Counter() for target in ("l2", "llc")}
    llc_by_group = {item["group"]: item["cohort"] for item in snapshot.get("llc_target_classified_cohorts", [])}
    for item in classes:
        out = dict(run_id=spec["run_id"], stage=spec["stage"], roi=spec["roi"], policy=spec["policy"],
                   method=spec["method"], mode=spec["mode"], seed=spec.get("seed"), budget=spec["budget"], **item)
        target = "l2" if item["fill_scope"] == "l2_target" else "llc"
        c = item.get("l2_cohort") if target == "l2" else llc_by_group.get(item["arm"]+"/"+item["fill_scope"]+"/"+item["key"])
        out["target_cohort"] = c
        for k in ("cache_attempted", "allowed", "cache_accepted_nonmerged", "cache_merged", "cache_rejected", "pq_enqueued",
                  "current_suppressed", "page_suppressed", "budget_suppressed"):
            totals[k] += item[k]
        audit.check(item["cache_attempted"] == item["allowed"]+item["current_suppressed"]+item["page_suppressed"]+item["budget_suppressed"],
                    "classified_gate_conservation", spec["run_id"], key=item["key"])
        audit.check(item["allowed"] == item["cache_accepted_nonmerged"]+item["cache_merged"]+item["cache_rejected"],
                    "classified_cache_result_conservation", spec["run_id"], key=item["key"])
        audit.check(sum(item["actual_attempt_phase_counts"].values()) == item["cache_attempted"] and
                    sum(item["actual_allowed_phase_counts"].values()) == item["allowed"],
                    "classified_actual_phase_conservation", spec["run_id"], key=item["key"])
        if c is not None:
            audit.check(item["pq_enqueued"] == c["enqueued"], "classified_target_enqueue_conservation", spec["run_id"], key=item["key"])
            audit.check(c["enqueued"] == sum(c[k] for k in END_STATES), "classified_target_lifecycle_conservation", spec["run_id"], key=item["key"])
            cohort_totals[target].update({k:c[k] for k in COHORT_FIELDS})
        else:
            audit.check(item["pq_enqueued"] == 0, "nonzero_classified_target_has_cohort", spec["run_id"], key=item["key"])
        rows.append(out)
    translations = {"cache_attempted":"request_gate_generated", "allowed":"request_gate_allowed",
                    "cache_accepted_nonmerged":"request_cache_enqueued", "cache_merged":"request_cache_merged",
                    "cache_rejected":"request_cache_rejected", "pq_enqueued":"request_cache_pq_inserted",
                    "current_suppressed":"request_gate_current_suppressed", "page_suppressed":"request_gate_page_suppressed",
                    "budget_suppressed":"request_gate_budget_suppressed"}
    for key, field in translations.items():
        audit.check(totals[key] == snapshot.get(field), "classified_global_conservation", spec["run_id"], field=field,
                    classified=totals[key], global_count=snapshot.get(field))
    for target in ("l2", "llc"):
        c = cohort(snapshot, target)
        for k in COHORT_FIELDS:
            audit.check(cohort_totals[target][k] == c.get(k), "classified_"+target+"_global_cohort_conservation", spec["run_id"], field=k)
    return rows


def model_storage(run, raw):
    spec, snap = run["spec"], run["summary"]
    result = dict(storage_scope="partial_model_payload_only; excludes inference_scratch/control/native_tables/queues/host_RSS",
                  model_weight_dram_bytes=0 if spec["method"] in ("student", "stride_student") else None,
                  weight_dram_scope="resident_weights_no_modeled_NN_DRAM_transactions" if spec["method"] in ("student", "stride_student") else "not_applicable")
    if spec["method"] == "student":
        path = run["dir"] / "model.bin"
        if not path.is_file(): path = raw / "models" / spec["policy"] / "model.bin"
        if path.is_file():
            with path.open("rb") as stream: head = stream.read(80)
            if len(head) == 80 and head[:8] == b"LV602R1\0":
                values = struct.unpack("<8s16IQ", head)
                h = values[5]
                result.update(model_parameter_count=values[-1], model_weight_bytes=values[-1]*4,
                              recurrent_hc_bytes=8*h, hidden_size=h, global_state_entries=1,
                              model_storage_source=str(path), binary_file_bytes=path.stat().st_size)
        result.update(recurrent_hc_bytes=snap.get("four602_state_bytes", result.get("recurrent_hc_bytes")),
                      global_state_entries=snap.get("four602_global_state_entries", result.get("global_state_entries")))
    elif spec["method"] == "stride_student":
        result.update(model_weight_bytes=snap.get("weight_bytes"), recurrent_state_configured_bytes=snap.get("state_bytes_configured"),
                      recurrent_state_peak_bytes=snap.get("state_bytes_peak"), state_capacity=snap.get("state_capacity"),
                      state_evictions=snap.get("state_evictions"), state_scope="per_PC_old_Stride_runtime")
    return result


def attach_cost(metrics, path, audit):
    if not path.is_file():
        audit.notes.append("Detailed source/host cost record absent; model payload still derived independently from actual binary.")
        return None
    cost=read(path)
    for row in metrics:
        policy=row["policy"]
        if row["method"]=="student":
            source=cost.get("nn_models",{}).get(policy)
            if not source:continue
            model=source["model"];host=source["host_allocations"]
            matches=(row.get("model_weight_bytes")==model["fp32_weight_bytes"] and
                     row.get("recurrent_hc_bytes")==model["global_hc_bytes"])
            audit.check(matches,"cost_record_matches_actual_model_payload",row["run_id"])
            if not matches:continue
            row.update(model_weight_plus_hc_payload_bytes=model["weight_plus_global_hc_payload_bytes"],
                       model_payload_qualifier=model["payload_qualifier"],
                       host_runtime_object_plus_persistent_requested_bytes=host["object_plus_persistent_requested_bytes"],
                       host_persistent_includes_weights_and_hc=True,
                       host_observed_infer_incremental_peak_requested_bytes=host["max_infer_incremental_peak_requested_bytes"],
                       host_observed_fixture_max_K=source["fixture"]["max_count"],
                       host_allocation_scope=source["scope"],cost_record_source=str(path))
        elif row["method"]=="native" and policy in cost.get("native",{}):
            row.update(native_source_cost_inventory=cost["native"][policy],cost_record_source=str(path))
    return cost


def derive(run, baseline, raw):
    s, x = run["spec"], run["summary"]
    ins, cyc = x["instructions"], x["cycles"]
    b = baseline["summary"] if baseline else None
    row = {k:s.get(k) for k in ("run_id","protocol","stage","roi","policy","method","mode","budget","seed","macs","page_policy","skip","warmup","measurement_records","build_id")}
    row.update(status="complete", instructions=ins, cycles=cyc, ipc=ratio(ins,cyc),
               no_pf_run=baseline["spec"]["run_id"] if baseline else None, no_pf_L2_LOAD_misses=b.get("M") if b else None,
               l2_load_accesses=x.get("NL"), l2_load_misses=x.get("M"), l2_load_mpki=1000*ratio(x["M"],ins),
               ipc_speedup_vs_NoPF=ratio(ratio(ins,cyc),ratio(b["instructions"],b["cycles"])) if b else None,
               net_l2_load_miss_reduction=ratio(b["M"]-x["M"], b["M"]) if b else None,
               no_pf_LLC_LOAD_misses=b.get("LLC_LOAD_misses") if b else None,
               net_llc_load_miss_reduction=ratio(b["LLC_LOAD_misses"]-x["LLC_LOAD_misses"],b["LLC_LOAD_misses"]) if b and "LLC_LOAD_misses" in b and "LLC_LOAD_misses" in x else None,
               requests_per_million_instructions=1e6*ratio(x["request_gate_allowed"],ins),
               generated_per_million_instructions=1e6*ratio(x["request_gate_generated"],ins),
               common_ceiling_is_not_realized_request_match=True,
               synchronous_MAC0_assumption=True, finite_cost_IPC_measured=False,
               training_interval_is_not_live_history=True)
    row["recorded_current_line_filter"]=run.get("record",{}).get("environment",{}).get("DYNAMIC602_FILTER_CURRENT")
    row["source_training_instruction_interval"]=[0,1000000] if s["method"]=="stride_student" else [0,20000000] if s["method"]=="student" else None
    row["source_training_epochs"]=12 if s["method"]=="stride_student" else 8 if s["method"]=="student" else None
    row["source_training_seed"]=s.get("seed") if s["method"] in ("student","stride_student") else None
    row["state_organization"]=("per_PC64_existing_decoder32" if s["method"]=="stride_student" else "one_global_hc_no_decoder_cap" if s["method"]=="student" else "native_or_no_neural_state")
    row.update(progress(x,s))
    row["total_callbacks"] = row["demand_callbacks"] + row["fill_callbacks"]
    row["updates_per_demand_callback"] = ratio(row["completed_updates"],row["demand_callbacks"])
    row["mac4_aggregate_dense_lower_bound_cycles"] = (row["inference_macs"]+3)//4
    row["mac4_dense_plus_4cycle_nonlinear_analytical_cycles"] = row["mac4_aggregate_dense_lower_bound_cycles"]+4*row["inference_nonlinears"]
    row["mac4_cost_scope"] = "aggregate_work_only; not latency/queue simulation; not IPC; no control/scratch term"
    for target in ("l2","llc"):
        c = cohort(x,target)
        available = all(c.get(k) is not None for k in COHORT_FIELDS) and (target == "l2" or x.get("llc_only_lifecycle_accuracy_available") is True)
        useful = c["timely_load"]+c["late_load"] if available else None
        prefix=target+"_target_"
        row.update({prefix+k:v for k,v in c.items()})
        row[prefix+"metrics_available"] = available
        row[prefix+"LOAD_useful"] = useful
        row[prefix+"LOAD_accuracy_enqueued"] = ratio(useful,c["enqueued"]) if available else None
        row[prefix+"LOAD_timeliness"] = ratio(c["timely_load"],useful) if available else None
        denominator = b.get("M" if target=="l2" else "LLC_LOAD_misses") if b else None
        row[prefix+"LOAD_useful_over_NoPF_misses"] = ratio(useful,denominator) if available else None
        row[prefix+"LOAD_timely_over_NoPF_misses"] = ratio(c["timely_load"],denominator) if available else None
    row["coverage_scope"] = "L2 and LLC target cohorts kept separate; useful/NoPF misses is target-level opportunity ratio, not causal miss reduction; no summation across levels"
    for k,v in x.items():
        if not isinstance(v,(dict,list)): row["raw__"+k]=v
    row.update(model_storage(run,raw))
    return row


def snapshot_rows(run, audit):
    rows=[]
    final=None
    with (run["dir"]/"snapshots.jsonl").open() as f:
        for line in f:
            x=json.loads(line)
            if x.get("event")!="snapshot": continue
            if x.get("final"): final=x
            rows.append(dict(run_id=run["spec"]["run_id"], stage=run["spec"]["stage"], roi=run["spec"]["roi"],
                             policy=run["spec"]["policy"], method=run["spec"]["method"], mode=run["spec"]["mode"],
                             budget=run["spec"]["budget"], seed=run["spec"].get("seed"),
                             instructions=x["instructions"], cycles=x["cycles"], cumulative_ipc=ratio(x["instructions"],x["cycles"]),
                             l2_load_misses=x["M"], dram_total_bus_bytes=x["dram_total_bus_bytes"], **progress(x,run["spec"])))
    audit.check(final==run["summary"],"summary_equals_actual_final_snapshot",run["spec"]["run_id"])
    audit.check(all(a["instructions"]<=b["instructions"] and a["cycles"]<=b["cycles"] for a,b in zip(rows,rows[1:])),
                "snapshot_progress_monotone",run["spec"]["run_id"])
    return rows


def phase_progress(rows, spec, config):
    # Exact differences at saved boundaries, never interpolated into missing data.
    width=spec.get("contract",{}).get("phase_instructions", config["phase_instructions"])
    out=[]
    if not rows: return out
    previous=rows[0]
    for phase in range(math.ceil(spec["instructions"]/width)):
        boundary=min(spec["instructions"],(phase+1)*width)
        end=next((x for x in rows if x["instructions"]>=boundary),None)
        if end is None: break
        item=dict(run_id=spec["run_id"],stage=spec["stage"],roi=spec["roi"],policy=spec["policy"],method=spec["method"],mode=spec["mode"],
                  budget=spec["budget"],seed=spec.get("seed"),phase=phase,
                  requested_end_instructions=boundary,actual_start_instructions=previous["instructions"],
                  actual_end_instructions=end["instructions"],boundary_scope="first_saved_snapshot_at_or_after_requested_boundary; actual coordinates retained")
        for k in ("instructions","cycles","demand_callbacks","fill_callbacks","completed_updates","inference_macs","inference_nonlinears","l2_load_misses","dram_total_bus_bytes"):
            item["interval_"+k]=end[k]-previous[k]
        item["interval_ipc"]=ratio(item["interval_instructions"],item["interval_cycles"])
        out.append(item);previous=end
    return out


def verify_callbacks(run, audit):
    path=run["dir"]/"callbacks.csv.gz"
    if not path.exists():path=run["dir"]/"callbacks.csv"
    required=run["spec"]["method"]=="student" and run["spec"]["mode"]=="full" and run["spec"]["budget"]==0
    if not path.exists():
        audit.check(not required,"required_actual_callback_log_present",run["spec"]["run_id"])
        return dict(run_id=run["spec"]["run_id"],available=False,required=required)
    opener=gzip.open if path.suffix==".gz" else open
    counts=Counter();last_ins=-1;good=True;rows=0;malformed=[]
    with opener(path,"rt") as stream:
        for i,item in enumerate(csv.DictReader(stream)):
            try:
                instruction=int(item["instructions"]);completed=int(item["completed_updates"])
                sequence=int(item["sequence"]);address=int(item["address"])
                if item["kind"] not in ("D","F"):raise ValueError("unknown event kind")
            except (ValueError,TypeError,KeyError) as error:
                malformed.append(dict(row=i,fields=item,error=str(error)))
                continue
            good &= sequence==rows and completed==rows and instruction>=last_ins and address%64==0 and not malformed
            counts[item["kind"]]+=1;last_ins=instruction;rows+=1
    expected=counters(run["summary"])
    complete=(good and not malformed and counts["D"]==expected.get("demand_callbacks",0) and
              counts["F"]==expected.get("fill_callbacks",0) and rows==expected.get("completed_updates",0))
    audit.check(good,"actual_callback_recorded_prefix_valid",run["spec"]["run_id"],recorded_rows=rows)
    audit.check(complete,"actual_callback_log_complete",run["spec"]["run_id"],recorded_complete_rows=rows,
                expected_updates=expected.get("completed_updates",0),malformed_rows=malformed)
    if not complete:audit.notes.append("Callback log incomplete/corrupt for "+run["spec"]["run_id"]+"; no missing callbacks are synthesized. Full summary axes are independent counters.")
    return dict(run_id=run["spec"]["run_id"],available=True,path=str(path),rows=rows,demand_rows=counts["D"],fill_rows=counts["F"],
                recorded_prefix_valid=bool(good),complete_log=bool(complete),expected_updates=expected.get("completed_updates",0),
                missing_complete_rows=expected.get("completed_updates",0)-rows,malformed_rows=malformed)


def verify_selection(selection, runs, config, audit):
    if not selection: return dict(available=False)
    by_id={r["spec"]["run_id"]:r for r in runs}
    def val(policy,method):
        return by_id.get("validation_r0_{}_{}_full_cap0".format(policy,method))
    result=dict(available=True,source="validation_only",selected=selection,templates={})
    audit.check(selection.get("protocol")==config["protocol"] and selection.get("validation_records")==config["validation_records"],
                "selection_protocol_and_heldout_interval")
    no=val("none","none")
    if no:
        calculated=counters(no["summary"])["demand_callbacks"]*1000000//no["summary"]["instructions"]
        audit.check(calculated==selection["common_cap"],"validation_common_cap_exact",calculated=calculated,selected=selection["common_cap"])
        result["cap_recomputed"]=calculated
    else:
        result["cap_recomputed"]=None
        audit.notes.append("Selection source NoPF validation is absent from requested analysis stages; cap not reverified here.")
    for policy in POLICIES[1:]:
        source=val(policy,"student")
        if source is None: continue
        counts=Counter();slots={}
        for name,hist in source["summary"]["four602_distributions"].items():
            suffix=name.split("_",2)[2]
            if suffix=="raw_K": counts.update({int(k):v for k,v in hist.items()})
            if suffix.startswith("raw_delta_fill_"):
                fill,slot=map(int,suffix[len("raw_delta_fill_"):].split("_slot_"))
                slots.setdefault(slot,Counter()).update({(int(k),fill):v for k,v in hist.items()})
        mode=min(counts,key=lambda k:(-counts[k],k))
        selected=[list(min(slots[i],key=lambda k:(-slots[i][k],k))) for i in range(max(slots)+1)]
        original=selection["templates"][policy]
        audit.check(mode==original["modal_count"] and selected==original["slots"] and original["source_run"]==source["spec"]["run_id"],
                    "validation_joint_delta_fill_template_exact", policy=policy)
        result["templates"][policy]=dict(modal_count_recomputed=mode,slots_recomputed=selected,source_run=source["spec"]["run_id"])
    return result


def pair_rows(metrics):
    rows=[]
    def context(r):
        return tuple(r.get(k) for k in ("stage","roi","policy","seed","protocol","build_id","skip","warmup","page_policy"))
    by_context=defaultdict(list)
    for row in metrics: by_context[context(row)].append(row)
    for group in by_context.values():
        fulls=[x for x in group if x["method"]=="student" and x["mode"]=="full" and x["budget"]==0]
        for full in fulls:
            for other in group:
                mechanism=(other["method"]=="student" and other["mode"] in ("fixed_payload","reset_state")) or other["method"]=="template"
                if not mechanism or other["budget"]!=0: continue
                rows.append(dict(comparison="mechanism_vs_full_student", policy=full["policy"],roi=full["roi"],stage=full["stage"],seed=full["seed"],
                                 control_run=full["run_id"],candidate_run=other["run_id"],candidate_method=other["method"],candidate_mode=other["mode"],
                                 delta_ipc=other["ipc"]-full["ipc"],candidate_over_control_ipc=ratio(other["ipc"],full["ipc"]),
                                 delta_allowed=other["raw__request_gate_allowed"]-full["raw__request_gate_allowed"],
                                 delta_dram_bus_bytes=other["raw__dram_total_bus_bytes"]-full["raw__dram_total_bus_bytes"],
                                 interpretation="Actual intervention: fixed_payload preserves full forward/K/hc at each given callback, but altered cache feedback can change future callbacks and entire hc trajectory; reset_state clears each callback; template removes NN gate/state"))
        for original in group:
            if original["budget"]!=0:continue
            for capped in group:
                if capped["budget"]<=0 or capped["method"]!=original["method"] or capped["mode"]!=original["mode"]:continue
                rows.append(dict(comparison="common_ceiling_vs_uncapped",policy=original["policy"],roi=original["roi"],stage=original["stage"],seed=original["seed"],
                                 control_run=original["run_id"],candidate_run=capped["run_id"],candidate_method=capped["method"],candidate_mode=capped["mode"],
                                 delta_ipc=capped["ipc"]-original["ipc"],candidate_over_control_ipc=ratio(capped["ipc"],original["ipc"]),
                                 delta_allowed=capped["raw__request_gate_allowed"]-original["raw__request_gate_allowed"],
                                 delta_dram_bus_bytes=capped["raw__dram_total_bus_bytes"]-original["raw__dram_total_bus_bytes"],
                                 interpretation="Same validation-selected request-credit ceiling across capped arms; actual allowed requests are measured, not assumed equal"))
    for row in metrics:
        if row["method"] not in ("student","stride_student") or row["mode"]!="full":continue
        matches=[x for x in metrics if context(x)==context(row) and x["method"]=="native" and x["mode"]=="full" and x["budget"]==row["budget"]]
        for native in matches:
            rows.append(dict(comparison="student_vs_native",policy=row["policy"],roi=row["roi"],stage=row["stage"],seed=row["seed"],
                             budget=row["budget"],control_run=native["run_id"],candidate_run=row["run_id"],candidate_method=row["method"],candidate_mode=row["mode"],
                             delta_ipc=row["ipc"]-native["ipc"],candidate_over_control_ipc=ratio(row["ipc"],native["ipc"]),
                             delta_allowed=row["raw__request_gate_allowed"]-native["raw__request_gate_allowed"],
                             realized_allowed_request_ratio=ratio(row["raw__request_gate_allowed"],native["raw__request_gate_allowed"]),
                             delta_dram_bus_bytes=row["raw__dram_total_bus_bytes"]-native["raw__dram_total_bus_bytes"],
                             interpretation=("Native SPP FILTER sees only PF fills; student sees every raw L2-fill callback; equal effective input not claimed" if row["policy"]=="spp" else
                                 "Stride student retains per-PC64 state/decoder32 and recorded current-line filter; comparator native Stride remains legacy" if row["policy"]=="stride" else
                                 "Actual paired native/student route; original checkpoint training budget differs across policies")))
    return rows


def route_rows(manifest, metrics):
    result=[]
    supplied=manifest.get("routes", manifest) if isinstance(manifest,dict) else {}
    for policy in POLICIES:
        for route in ("offline","live","old_formal","new_formal"):
            evidence=supplied.get(policy,{}).get(route,{})
            if isinstance(evidence,str): evidence={"evidence":evidence}
            relevant=[x for x in metrics if x["stage"]=="test" and x["policy"]==policy and x["mode"]=="full" and
                      (x["method"]=="stride_student" if policy=="stride" else x["method"]=="student")]
            status=("completed_"+str(len(relevant))+"_formal_runs") if route=="new_formal" and relevant else evidence.get("status","not_supplied")
            result.append(dict(policy=policy,route=route,status=status,evidence=evidence.get("evidence"),details=evidence.get("details"),
                               comparable_IPC_scope="new_formal shared protocol only; replay/offline/old protocol IPC never cross-compared"))
    return result


def definitions():
    return {
      "protocol":"Separate four602 frozen-live MAC0 study. New protocol only. Existing checkpoints, source and prior results preserved.",
      "ipc":"Retired ROI instructions / measured ROI cycles; tiny retirement overshoot is retained, not rescaled.",
      "mpki":"1000 * L2 LOAD misses M / retired ROI instructions. NL is L2 LOAD accesses.",
      "accuracy":"Separate target-level LOAD-useful / target cohort enqueued; cohort starts on first source L2 PQ insertion; denominator includes ROI-end pending and resident-unused. Cache merges and WQ forwards are not new target-cohort enqueues. Cohorts retain FIRST source-enqueue intent; a later merge may upgrade final fill level. These are neither final-fill groups nor any-level total usefulness.",
      "timeliness":"Separate target-level timely_load / (timely_load + late_load). RFO usefulness excluded. Zero denominator -> null.",
      "coverage":"Separate L2/LLC target LOAD-useful over matched NoPF target-level LOAD misses. L2 target cohort is partial subset if LLC-target requests exist; LLC and L2 ratios never added. Coverage is not a causal claim or net miss reduction.",
      "LLC_availability":"LLC accuracy/timeliness/coverage calculated only if llc_only_lifecycle_accuracy_available is explicitly true. Rejected untracked handoffs or missing IDs disable LLC derived ratios, even if some tracked outcomes exist.",
      "net_miss_reduction":"(NoPF LOAD misses - candidate LOAD misses) / NoPF LOAD misses; separate L2 and LLC, distinct from useful-request coverage.",
      "request_budget":"requests_per_million_instructions = 1e6 * request_gate_allowed / actual retired instructions. Cap selected only from validation NoPF eligible L2 LOAD callbacks; common ceiling does not imply same realized allowed count.",
      "cache_results":"Allowed = accepted_nonmerged + merged + rejected. accepted_nonmerged may include WQ forwarding; actual PQ insertion is a separate field. PQ insertion = L2-target cohort enqueued + LLC-target cohort enqueued.",
      "DRAM":"Use actual DRAM controller queue and actual bus start/completion events, bytes = those transfers * line_bytes. Started and completed bytes separate, with initial and final pending counts. Never issued prefetches * 64. Bus bytes include all traffic, not just prefetches.",
      "NN_traffic_storage":"Frozen resident weights have no modeled weight DRAM fetches. 4*MAC is neither measured weight traffic nor DRAM; biases/reuse/cache behavior are excluded. Weights+h/c is partial payload. Host runtime object+persistent requested heap already includes weights/hc and must not be added to them. Observed Infer heap peaks cover specified correctness fixture only, not all-input bounds. Excludes outer Four602 controller, measurement, stack, malloc metadata and native diagnostic histograms; not complete hardware SRAM, RSS, RTL area or deployment cost.",
      "MAC0":"Measured IPC assumes synchronous zero-cost C++ inference. Actual operation counters retained. MAC4 aggregate work estimates are not finite-cost simulation or predicted IPC; exclude scheduling/queue/control/scratch overhead.",
      "state":"Streamer/AMPM/SPP use one chronological global h/c without decoder cap; Stride student retains its original per-PC64 runtime/decoder32 and recorded current-line filter. reset_state clears before every callback. Training instruction count 0..20M is not online state history.",
      "SPP_inputs":"Student training/live sees every raw CACHE L2-fill callback, using evicted address and event kind. Native SPP preserves existing FILTER feedback on prefetch fills only. Do not claim identical effective input streams.",
      "actions":"Stride raw_K precedes its existing decoder cap; Stride delta comes from actual decoded slots and fill=2 is fixed interface, not a learned fill. Full raw global model actions before validation-selected payload intervention and common output gate; payload after intervention. Four monitor's empty distributions are excluded for separate stride_* engines. SPP fill callbacks update h/c but do not emit and are excluded from demand-action histogram denominator.",
      "observation_axes":"instructions, demand_callbacks and completed_updates shown on curves are ROI-reset counters. h/c already inherits the configured5M warmup except reset_state. warmup_completed_updates/demand_callbacks/fill_callbacks are separately exposed. Completed updates are aggregate callbacks, not per-PC visible history or training samples. These curves do not start from zero-history and do not establish a minimum sample/history requirement.",
      "phases":"Generation profiles follow 5M retired-instruction bins, final bin absorbs retirement overshoot. Classified allowed time uses actual gate phase; useful cohorts retain generation origin. Progress interval values are exact differences of saved snapshots, with actual endpoints recorded; no prefix extrapolation.",
      "splits":"Streamer/AMPM/SPP source training0..20M (8 epochs); Stride retained formal_v2 training0..1M (12 epochs balanced). Different training budgets mean teacher/student paired routes, not a matched four-model learning-algorithm ranking. Shared validation20..25M, tests25..50M and75..100M with common5M cache warmup. 25..50M previously inspected, explicitly a replication ROI; 75..100M future interval. Validation lies before tests, may overlap test cache warmup. No new independent training seeds implied.",
      "causality":"Only actual fixed_payload/template/reset_state contrasts establish intervention outcomes; action labels alone do not establish IPC attribution. Fixed payload preserves the complete forward/K/hc update for a given current callback, but changed prefetches alter later cache behavior, future callbacks and entire hc trajectory; entire-run state equality is not guaranteed. Negative and mixed results retained.",
      "SPP_label_canonicalization":"Offline SPP normalize_actions deduplicates equal target addresses and keeps min fill (L2 has priority), representing canonical queue-effect labels. Native generated profiles count raw source prefetch calls. Their K distributions have different call/canonical-action scopes; never assert exact teacher-action agreement from K alone.",
      "teacher_classification":"Four602 global native/student/template routes set teacher_category=3 (unclassified/not available); no shadow-teacher overlap or silent attribution is inferred. The retained Stride runtime separately records its arrival-time shadow labels 0/1/2.",
      "unused":"ROI-unused and evicted-unused are lifecycle outcomes, not direct pollution measurements.",
      "null":"Missing baseline, missing source evidence or zero denominator -> null; absent counter fields fail applicable schema checks instead of silently becoming zero."
    }


def plot_outputs(out, metrics, snapshots, phases, histograms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors={"stride":"#0072B2","streamer":"#D55E00","ampm":"#009E73","spp":"#CC79A7","none":"#777777"}
    figures=[]
    def finish(fig,name):
        fig.tight_layout()
        for suffix in ("png","svg"):fig.savefig(out/(name+"."+suffix),dpi=180,bbox_inches="tight")
        plt.close(fig);figures.append(name+".png")
    tests=[r for r in metrics if r["stage"]=="test"]
    for roi in sorted({r["roi"] for r in tests}):
        chosen=[r for r in tests if r["roi"]==roi and r["mode"]=="full" and r["method"] in ("native","student","stride_student") and r["budget"]==0]
        if not chosen:continue
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=[]
        for i,r in enumerate(chosen):
            label="{} {} s{}".format(r["policy"],r["method"],r["seed"]);labels.append(label)
            axes[0].bar(i,r["ipc_speedup_vs_NoPF"] or float("nan"),color=colors[r["policy"]],hatch="//" if "student" in r["method"] else "")
            axes[1].bar(i,r["raw__dram_total_bus_bytes"]/2**20,color=colors[r["policy"]],hatch="//" if "student" in r["method"] else "")
        axes[0].axhline(1,color="black",lw=.8);axes[0].set_ylabel("IPC / matched NoPF")
        axes[1].set_ylabel("Actual completed DRAM bus MiB")
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=55,ha="right");ax.grid(axis="y",alpha=.2)
        fig.suptitle("ROI {}: complete uncapped runs; MAC0 synchronous assumption".format(roi))
        finish(fig,"ipc_traffic_roi"+str(roi))
        for budget in sorted({r["budget"] for r in tests if r["roi"]==roi}):
            chosen=[r for r in tests if r["roi"]==roi and r["budget"]==budget and r["mode"]=="full" and r["method"] in ("native","student","stride_student")]
            fig,ax=plt.subplots(figsize=(11,5))
            for r in chosen:
                if r["ipc_speedup_vs_NoPF"] is None: continue
                ax.scatter(r["requests_per_million_instructions"],r["ipc_speedup_vs_NoPF"],s=60,color=colors[r["policy"]],marker="s" if r["method"]=="native" else "o",
                           label="{} {} s{}".format(r["policy"],r["method"],r["seed"]))
            ax.axhline(1,color="black",lw=.8);ax.set_xlabel("Actual allowed requests / million instructions");ax.set_ylabel("IPC / NoPF")
            ax.set_title("ROI {} common ceiling {} (0 = uncapped)".format(roi,budget));ax.grid(alpha=.2)
            if any(r["ipc_speedup_vs_NoPF"] is not None for r in chosen):ax.legend(loc="upper left",bbox_to_anchor=(1.02,1),fontsize=8)
            finish(fig,"requests_ipc_roi{}_cap{}".format(roi,budget))
    selected=[r for r in tests if r["method"] in ("student","stride_student") and r["mode"]=="full" and r["budget"]==0]
    for roi in sorted({r["roi"] for r in selected}):
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for r in selected:
            if r["roi"]!=roi:continue
            for ax,stat in zip(axes,("K","delta","fill")):
                count=Counter()
                for h in histograms:
                    if h["run_id"]==r["run_id"] and h["stream"]=="raw" and h["statistic"]==stat:count[h["value"]]+=h["count"]
                total=sum(count.values())
                if total:ax.plot(sorted(count),[count[k]/total for k in sorted(count)],"o-",label="{} s{}".format(r["policy"],r["seed"]),color=colors[r["policy"]],alpha=.85)
                ax.set_xlabel(stat+" (actual decoded values)");ax.set_ylabel("Whole-ROI observed fraction");ax.grid(alpha=.2)
                if stat=="fill":ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(fontsize=8);fig.suptitle("ROI {} raw demand actions; no prefix extrapolation".format(roi))
        finish(fig,"action_distribution_roi"+str(roi))
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for r in selected:
            if r["roi"]!=roi:continue
            seq=[v for v in snapshots if v["run_id"]==r["run_id"] and v["cycles"]>0]
            for ax,field,label in zip(axes,("instructions","demand_callbacks","completed_updates"),
                                     ("ROI retired instructions","ROI L2 LOAD callbacks","ROI completed updates (aggregate)")):
                ax.plot([x[field] for x in seq],[x["cumulative_ipc"] for x in seq],label="{} s{}".format(r["policy"],r["seed"]),color=colors[r["policy"]])
                ax.set_xlabel(label);ax.set_ylabel("Cumulative ROI IPC at actual snapshot");ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle("ROI {}: warm h/c retained; progress counters restart at ROI".format(roi))
        finish(fig,"progress_axes_roi"+str(roi))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for r in selected:
            if r["roi"]!=roi:continue
            points=[]
            ids=sorted({h["phase"] for h in histograms if h["run_id"]==r["run_id"]})
            for p in ids:
                hs=[h for h in histograms if h["run_id"]==r["run_id"] and h["phase"]==p and h["stream"]=="raw" and h["statistic"]=="emit"]
                total=sum(h["count"] for h in hs);emitted=sum(h["count"] for h in hs if h["value"]==1)
                points.append((p,ratio(emitted,total)))
            axes[0].plot([p for p,v in points],[v for p,v in points],"o-",color=colors[r["policy"]],label="{} s{}".format(r["policy"],r["seed"]))
            ps=[p for p in phases if p["run_id"]==r["run_id"]]
            axes[1].plot([p["phase"] for p in ps],[p["interval_ipc"] for p in ps],"o-",color=colors[r["policy"]],label=r["policy"])
        axes[0].set_ylabel("Raw emit / demand decision");axes[1].set_ylabel("IPC between saved phase boundaries")
        for ax in axes:ax.set_xlabel("5M instruction phase");ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(fontsize=8);fig.suptitle("ROI {} actual phase observations".format(roi))
        finish(fig,"phase_observations_roi"+str(roi))
    return figures


def method_diagram(out):
    # Repo-native vector figure: no generated bitmap or speculative hardware.
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="420" viewBox="0 0 1140 420">
<style>text{font:16px sans-serif;fill:#1e293b}.small{font-size:13px}.box{fill:#f8fafc;stroke:#334155;stroke-width:1.5}.arrow{stroke:#475569;stroke-width:2;fill:none;marker-end:url(#a)}</style>
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#475569"/></marker></defs>
<text x="30" y="30" font-weight="bold">Three global teacher-action students: training, live inference, common measurement</text>
<rect class="box" x="30" y="60" width="245" height="90" rx="8"/><text x="45" y="88">Own teacher action labels</text><text class="small" x="45" y="112">0..20M; models trained separately</text><text class="small" x="45" y="134">Address only; SPP also D/F kind</text>
<path class="arrow" d="M275 105 H325"/><rect class="box" x="325" y="60" width="210" height="90" rx="8"/><text x="340" y="88">Existing seed7 H8 models</text><text class="small" x="340" y="112">Frozen FP32 weights</text><text class="small" x="340" y="134">Python/C++ state/action parity</text>
<path class="arrow" d="M535 105 H585"/><rect class="box" x="585" y="60" width="250" height="90" rx="8"/><text x="600" y="88">Independent C++ live inference</text><text class="small" x="600" y="112">Global h/c; SPP fills advance state</text><text class="small" x="600" y="134">MAC0 synchronous IPC assumption</text>
<path class="arrow" d="M710 150 V205"/><rect class="box" x="585" y="205" width="250" height="90" rx="8"/><text x="600" y="233">Full / fixed payload / reset</text><text class="small" x="600" y="257">Validation selects joint delta/fill</text><text class="small" x="600" y="279">Template also fixes count, no NN</text>
<path class="arrow" d="M998 205 V175 H858 V105 H835"/><text class="small" x="865" y="162">Actual L2 LOAD / SPP raw F callbacks</text>
<path class="arrow" d="M835 250 H885"/><rect class="box" x="885" y="205" width="225" height="90" rx="8"/><text x="900" y="233">Actual cache / DRAM events</text><text class="small" x="900" y="257">Separate L2 and LLC cohorts</text><text class="small" x="900" y="279">NoPF at same protocol + ROI</text>
<rect class="box" x="30" y="205" width="245" height="90" rx="8"/><text x="45" y="233">Validation: 20..25M</text><text class="small" x="45" y="257">NoPF LOAD callbacks define ceiling</text><text class="small" x="45" y="279">No test-selected templates or cap</text>
<path class="arrow" d="M275 250 H585"/>
<text x="30" y="345">Native policies remain independent comparators. Stride is separate: train 0..1M, PC/address input, per-PC h/c.</text>
<text x="30" y="375">Operation counts + resident weights/h/c → MAC4 analytical work estimate only. No finite-cost IPC claim.</text>
<text class="small" x="30" y="404">Tests: 25..50M (previously inspected replication) and 75..100M (future ROI), each with common 5M cache warmup.</text></svg>'''
    (out/"method.svg").write_text(svg)


def report(out, audit_record, metrics, pairs, routes, figures, defs):
    selected=[r for r in metrics if r["stage"]=="test" and r["method"] in ("student","stride_student") and r["mode"]=="full"]
    lines=["# Four-prefetcher frozen live study","","Status: **{}**; {} complete selected runs; {} checks failed.".format(audit_record["status"],len(metrics),audit_record["failure_count"]),"",
           "All measured IPC here assumes synchronous zero-cost inference (MAC0). Existing true seed7 checkpoints are used; this study does not establish three-seed training robustness or finite-cost IPC gains.",
           "Streamer/AMPM/SPP training0..20M (8 epochs) and Stride training0..1M (12 epochs balanced) have different budgets; compare each native/student route, not a matched learning-algorithm ranking. Training coordinates are separate from online recurrent state history. SPP model labels canonicalize duplicate targets with L2 priority, while native generated K counts raw source calls. A fixed-payload intervention preserves each given callback's complete forward/K/hc update, but changed cache feedback can change future callbacks and the full state trajectory. Tests25..50M were already inspected in prior work;75..100M is a distinct future interval. Native SPP preserves prefetch-fill FILTER callbacks; its student's training interface observes every raw L2-fill callback, so equal effective input is not claimed.","",
           "| Policy | ROI | Seed | Method | Ceiling | IPC / NoPF | Allowed / M instructions | L2 LOAD accuracy | LLC LOAD accuracy |",
           "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    def num(x):return "NA" if x is None else "{:.5f}".format(x)
    for r in selected:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(r["policy"],r["roi"],r["seed"],r["method"],r["budget"],num(r["ipc_speedup_vs_NoPF"]),num(r["requests_per_million_instructions"]),num(r["l2_target_LOAD_accuracy_enqueued"]),num(r["llc_target_LOAD_accuracy_enqueued"])))
    comparisons=[p for p in pairs if p["comparison"]=="student_vs_native" and p["stage"]=="test"]
    wins=sum(p["delta_ipc"]>0 for p in comparisons);losses=sum(p["delta_ipc"]<0 for p in comparisons)
    lines += ["","| Claim | Evidence status | Scope |","|---|---|---|",
              "| Existing teacher-action students execute independently online | {} | Exact exported weights and separate parity evidence required; new formal complete rows in per_run.csv |".format("数据支持" if selected else "待验证"),
              "| Students uniformly outperform native teachers | {} | {} positive / {} negative of {} actual paired comparisons; every ROI/cap retained in paired_comparisons.csv |".format("未支持" if losses else "待验证" if not comparisons else "数据支持（仅列出范围）",wins,losses,len(comparisons)),
              "| Learned gate, payload and state explain performance | 待验证 | Read actual fixed_payload/template/reset_state interventions individually; labels alone are not causal evidence |",
              "| Benefits survive finite inference resources | 待验证 | MAC4 outputs are analytical work estimates only; no finite-cost IPC simulation in this protocol |",
              "| Independent training seed robustness | 待验证 | Existing seed7 only; no size sweep mislabeled as seeds |","",
              "L2 and LLC target accuracy/timeliness use separate LOAD-only cohorts. L2 usefulness coverage is a target-subset opportunity ratio when LLC requests exist; it is not total SPP coverage. LLC availability false produces NA. Actual completed DRAM bytes, bus starts, queue arrivals and pending boundary work are separately retained. Unused requests are not called pollution.","",
              "Raw inputs are read-only. The full raw snapshot JSON is retained in raw_snapshots.json. No replay/live or old/new protocol IPC is cross-compared.",
              "","## Route evidence","", "| Policy | Route | Status | Evidence |","|---|---|---|---|"]
    for r in routes:lines.append("| {} | {} | {} | {} |".format(r["policy"],r["route"],r["status"],r["evidence"] or "Not supplied"))
    lines += ["","Plot status: "+audit_record["plot_status"]+". Callback logs checked: "+str(audit_record["callback_logs_checked"])+". Incomplete callback logs are not reconstructed or extrapolated.","","## Outputs","", "[Per-run complete metrics](per_run.csv), [all raw fields](raw_flat.csv), [paired contrasts](paired_comparisons.csv), [audit](audit.json), [definitions](metric_definitions.json), [route evidence](route_status.csv), [selection verification](selection_verification.json).",""]
    lines += ["![{}]({})".format(name,name) for name in figures]
    lines += ["","![Method](method.svg)"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    headers=["Policy","ROI","Seed","Method","Mode","Ceiling","IPC / NoPF","L2 accuracy","LLC accuracy"]
    table="<table><tr>"+"".join("<th>"+x+"</th>" for x in headers)+"</tr>"
    for r in [x for x in metrics if x["stage"]=="test"]:
        values=[r["policy"],r["roi"],r["seed"],r["method"],r["mode"],r["budget"],num(r["ipc_speedup_vs_NoPF"]),num(r["l2_target_LOAD_accuracy_enqueued"]),num(r["llc_target_LOAD_accuracy_enqueued"])]
        table+="<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in values)+"</tr>"
    table+="</table>"
    pictures="".join('<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(name,name,name) for name in figures)
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Four-prefetcher formal study</title>
<style>body{font:16px system-ui;margin:2em auto;max-width:1300px;padding:0 1em;color:#172033}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd4df;padding:.45em}img{max-width:100%}a{color:#07559b}.status{padding:1em;background:#eef2f6}details{margin:1em 0}pre{white-space:pre-wrap}</style>'''
    body+="<h1>Four-prefetcher frozen live study</h1><p class=\"status\">"+html.escape(audit_record["status"])+" · "+str(len(metrics))+" complete runs · "+str(audit_record["failure_count"])+" failed checks.</p>"
    body+="<p>Measured IPC assumes synchronous MAC0 inference. Existing seed7 models; no finite-cost IPC or independent-seed robustness claim. Global-model training0..20M/8epochs and Stride0..1M/12epochs are different budgets; training coordinates are not online state history. Replication25..50M and future75..100M remain separate.</p>"
    body+="<p>L2 and LLC target LOAD-only metrics remain separate. LLC ratios are NA if tracking is incomplete. Native SPP and its student have the documented fill-feedback input difference. Negative and mixed outcomes remain in this table.</p>"+table
    body+="<p>Actual native/student contrasts: "+str(wins)+" positive and "+str(losses)+" negative paired ΔIPC cases; see <a href=\"paired_comparisons.csv\">all paired data</a>.</p>"
    body+="<p>"+" · ".join('<a href="{0}">{1}</a>'.format(path,label) for path,label in [("per_run.csv","Per run"),("raw_flat.csv","Raw fields"),("audit.json","Audit"),("metric_definitions.json","Definitions"),("route_status.csv","Offline/live/old/new evidence"),("report.md","Paper notes")])+"</p>"
    body+='<img src="method.svg" alt="Actual teacher to frozen live method">'+pictures
    body+="<details><summary>Exact metric definitions and limits</summary><pre>"+html.escape(json.dumps(defs,indent=2,ensure_ascii=False))+"</pre></details></html>"
    (out/"index.html").write_text(body)


def analyze(args):
    raw=args.raw.resolve();out=args.out.resolve()
    config=read(args.config)
    configure(out,raw,config)
    if out.exists():raise ValueError("Preserve prior output; choose a fresh --out directory")
    selection=read(raw/"selection.json") if (raw/"selection.json").is_file() else None
    stages=set(args.stages)
    expected=expected_specs(config,selection,stages)
    unresolved_test_matrix="test" in stages and selection is None
    if args.expected_matrix:
        provided=read(args.expected_matrix)
        expected=provided.get("runs",provided.get("specs")) if isinstance(provided,dict) else provided
        if not isinstance(expected,list):raise ValueError("--expected-matrix must be a list or object with runs/specs list")
        expected=[s.get("spec",s) for s in expected if s.get("spec",s).get("stage") in stages]
        unresolved_test_matrix=False
    expected_by_id={s["run_id"]:s for s in expected}
    audit=Audit();runs=[];inventory=[];statuses=Counter()
    train_start,train_end=config["train_records"]
    val_start,val_end=config["validation_records"]
    audit.check(train_end<=val_start and train_start<train_end<=val_start<val_end,
                "train_validation_intervals_disjoint")
    audit.check(all(val_end<=lo<hi for lo,hi in config["test_records"]),
                "validation_test_intervals_disjoint")
    audit.check(len(expected_by_id)==len(expected),"unique_expected_run_ids")
    for path in sorted((raw/"runs").glob("*/run.json")):
        record=read(path);spec=record["spec"]
        if spec.get("stage") not in stages:continue
        statuses[record.get("status","unknown")]+=1
        inventory.append(dict(run_id=spec["run_id"],status=record.get("status"),source=str(path),spec=spec))
        audit.check(path.parent.name==spec["run_id"],"run_id_path_identity",spec["run_id"])
        audit.check(spec.get("protocol")==config["protocol"],"single_new_protocol_only",spec["run_id"])
        if spec["run_id"] in expected_by_id:
            wanted=expected_by_id[spec["run_id"]]
            audit.check(all(spec.get(k)==v for k,v in wanted.items()),"exact_expected_spec",spec["run_id"])
        elif not unresolved_test_matrix:audit.check(False,"unexpected_run_in_selected_stages",spec["run_id"])
        if record.get("status")!="complete":continue
        dest=path.parent
        audit.check(record.get("exit_code")==0,"successful_simulator_exit",spec["run_id"])
        summary=read(dest/"summary.json")
        audit.check(summary.get("final") is True and abs(summary["instructions"]-spec["instructions"])<=4,
                    "full_ROI_final_only",spec["run_id"],actual=summary["instructions"],expected=spec["instructions"])
        audit.check(summary["cycles"]>0,"positive_measured_cycles",spec["run_id"])
        audit.check(summary.get("numerical_failures",0)==0,"no_numerical_failures",spec["run_id"])
        audit.check(summary.get("measurement_unconsumed_context_labels",0)==0,"no_unconsumed_request_context",spec["run_id"])
        audit.check(summary.get("measurement_cohort_scope")=="separate_L2_and_LLC_target_cohorts","new_separate_target_scope",spec["run_id"])
        if spec["budget"]:
            audit.check(summary["request_gate_allowed"] <= summary["instructions"]*spec["budget"]//1000000,
                        "request_credit_ceiling_respected",spec["run_id"])
        if summary.get("llc_only_lifecycle_accuracy_available") is True:
            audit.check(summary.get("llc_target_prefetch_queue_rejected_untracked")==0 and summary.get("llc_target_shortcut_missing_pending_id")==0,
                        "LLC_available_requires_complete_handoff_tracking",spec["run_id"])
        c=counters(summary)
        if spec["method"]=="student":
            audit.check(summary.get("four602_method")=="student","actual_student_method_loaded",spec["run_id"])
            audit.check(c.get("completed_updates",0)==c.get("demand_callbacks",0)+c.get("fill_callbacks",0),"one_global_update_per_callback",spec["run_id"])
            audit.check(c.get("demand_inferences",0)==c.get("demand_callbacks",0) and c.get("fill_inferences",0)==c.get("fill_callbacks",0),
                        "demand_fill_inference_counts",spec["run_id"])
            audit.check(summary.get("four602_global_state_entries")==1 and summary.get("four602_state_evictions")==0,
                        "global_model_state_not_per_PC",spec["run_id"])
            if spec["policy"]=="spp":audit.check(c.get("fill_callbacks",0)>0,"SPP_raw_fill_callbacks_present",spec["run_id"])
        for target in ("l2","llc"):check_cohort(audit,cohort(summary,target),target,spec["run_id"])
        audit.check(summary.get("request_cache_pq_inserted")==summary.get("cohort_enqueued",0)+summary.get("llc_target_cohort",{}).get("enqueued",0),
                    "separate_target_enqueue_partition",spec["run_id"])
        audit.check(summary["request_gate_generated"]==summary["request_gate_allowed"]+summary["request_gate_budget_suppressed"]+summary["request_gate_page_suppressed"]+summary["request_gate_current_suppressed"],
                    "global_gate_conservation",spec["run_id"])
        audit.check(summary["request_gate_allowed"]==summary["request_cache_enqueued"]+summary["request_cache_merged"]+summary["request_cache_rejected"],
                    "global_cache_result_conservation",spec["run_id"])
        for suffix in ("generated","allowed","cache_merged","cache_rejected"):
            audit.check(summary["request_l2_target_"+suffix]+summary["request_llc_only_"+suffix]==summary["request_gate_"+suffix if suffix in ("generated","allowed") else "request_"+suffix],
                        "L2_LLC_scope_partition",spec["run_id"],field=suffix)
        line_bytes=summary["line_bytes"]
        audit.check(summary["dram_total_bus_bytes"]==summary["dram_read_bus_bytes"]+summary["dram_write_bus_bytes"] and
                    summary["dram_read_bus_bytes"]==line_bytes*summary["dram_read_bus_completed_transactions"] and
                    summary["dram_write_bus_bytes"]==line_bytes*summary["dram_write_bus_completed_transactions"],
                    "actual_completed_DRAM_bytes",spec["run_id"])
        audit.check(summary["dram_read_bus_started_bytes"]+summary["dram_write_bus_started_bytes"]-summary["dram_total_bus_bytes"]==line_bytes*summary["dram_bus_transfers_in_progress"],
                    "started_completed_pending_bus_conservation",spec["run_id"])
        runs.append(dict(spec=spec,summary=summary,dir=dest,record=record))
    completed_ids={r["spec"]["run_id"] for r in runs}
    missing=[rid for rid in expected_by_id if rid not in completed_ids]
    baseline_by_key={}
    for run in runs:
        if run["spec"]["method"]=="none" and run["spec"]["policy"]=="none":
            key=protocol_key(run["spec"])
            audit.check(key not in baseline_by_key,"unique_matched_NoPF",run["spec"]["run_id"])
            baseline_by_key[key]=run
    metrics=[];raw_rows=[];hists=[];classified=[];snapshots=[];phases=[];callback_reports=[]
    for run in runs:
        baseline=baseline_by_key.get(protocol_key(run["spec"]))
        if baseline is None:audit.notes.append("No complete matched NoPF for "+run["spec"]["run_id"]+"; baseline-derived metrics remain null.")
        if not missing and not unresolved_test_matrix:
            audit.check(baseline is not None,"complete_matrix_has_matching_NoPF",run["spec"]["run_id"])
        metrics.append(derive(run,baseline,raw))
        raw_rows.append(dict(run_id=run["spec"]["run_id"],**flatten(run["summary"])))
        hists.extend(profile_rows(run["summary"],run["spec"],audit))
        classified.extend(classification_rows(run["summary"],run["spec"],audit))
        seq=snapshot_rows(run,audit);snapshots.extend(seq);phases.extend(phase_progress(seq,run["spec"],config))
        if not args.skip_callback_verification:callback_reports.append(verify_callbacks(run,audit))
    cost_record=attach_cost(metrics,args.cost_fields,audit)
    selection_check=verify_selection(selection,runs,config,audit)
    incomplete=bool(missing or unresolved_test_matrix or any(v for k,v in statuses.items() if k!="complete"))
    status="FAIL" if audit.failures else "INCOMPLETE" if incomplete else "PASS"
    if not expected:status="INCOMPLETE"
    planned_count=len(expected)+(56 if unresolved_test_matrix else 0)
    import importlib.util
    can_plot=importlib.util.find_spec("matplotlib") is not None
    plot_status="disabled_by_flag" if args.no_plots else "generated" if can_plot else "pending_local_render_matplotlib_unavailable"
    record=dict(schema_version=1,status=status,plot_status=plot_status,callback_logs_checked=not args.skip_callback_verification,protocol=config["protocol"],stages=sorted(stages),complete_count=len(runs),
                expected_count=planned_count,resolved_expected_count=len(expected),unresolved_test_selection=unresolved_test_matrix,
                run_status_counts=dict(statuses),missing_or_incomplete_run_ids=missing,failure_count=len(audit.failures),
                checks=audit.checks,failures=audit.failures,notes=audit.notes,
                raw_inputs_modified=False,raw_root=str(raw),actual_full_ROI_only=True)
    out.mkdir(parents=True)
    pairs=pair_rows(metrics);routes=route_rows(read(args.route_manifest) if args.route_manifest else {},metrics)
    if cost_record:save(out/"cost_fields.json",cost_record)
    save(out/"audit.json",record);save(out/"raw_snapshots.json",{r["spec"]["run_id"]:r["summary"] for r in runs})
    save(out/"metric_definitions.json",definitions());save(out/"selection_verification.json",selection_check)
    save(out/"callback_verification.json",callback_reports);save(out/"inventory.json",inventory)
    csv_out(out/"per_run.csv",metrics);csv_out(out/"raw_flat.csv",raw_rows);csv_out(out/"paired_comparisons.csv",pairs)
    aggregate=Counter()
    for item in hists:
        aggregate[(item["run_id"],item["stream"],item["statistic"],item["value"])]+=item["count"]
    by_hist=Counter()
    for (rid,stream,stat,value),count in aggregate.items():by_hist[(rid,stream,stat)]+=count
    csv_out(out/"action_histograms_full_roi.csv",[dict(run_id=rid,stream=stream,statistic=stat,value=value,count=count,
                fraction=ratio(count,by_hist[(rid,stream,stat)]),scope="exact_sum_of_all_full_ROI_phases")
                for (rid,stream,stat,value),count in sorted(aggregate.items())])
    csv_out(out/"action_histograms.csv",hists);csv_out(out/"classified_requests.csv",classified)
    csv_out(out/"progress_snapshots.csv",snapshots);csv_out(out/"phase_progress.csv",phases)
    csv_out(out/"route_status.csv",routes)
    csv_out(out/"storage_cost.csv",[{k:v for k,v in m.items() if k in ("run_id","policy","method","mode","stage","roi","seed","budget","completed_updates","inference_macs","inference_nonlinears") or any(t in k for t in ("model_","recurrent_","storage_","mac4_","global_state","hidden_size","weight_dram","state_capacity","state_evictions","host_","native_source_cost","cost_record"))} for m in metrics])
    method_diagram(out)
    figures=[] if args.no_plots or not can_plot else plot_outputs(out,metrics,snapshots,phases,hists)
    report(out,record,metrics,pairs,routes,figures,definitions())
    print(json.dumps({k:record[k] for k in ("status","complete_count","expected_count","failure_count")},ensure_ascii=False))
    return 1 if audit.failures else 0 if status=="PASS" or args.partial else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True,help="Fresh output directory inside cache/formal_study_20261017")
    parser.add_argument("--config",type=Path,default=EXP/"config.json")
    parser.add_argument("--stages",nargs="+",choices=("test","validation"),default=["test","validation"])
    parser.add_argument("--expected-matrix",type=Path)
    parser.add_argument("--cost-fields",type=Path,default=EXP/"report/cost_fields.json")
    parser.add_argument("--route-manifest",type=Path,help="Optional {routes:{policy:{offline|live|old_formal:{status,evidence,details}}}}")
    parser.add_argument("--partial",action="store_true",help="Emit explicitly INCOMPLETE output while formal jobs are running; never extrapolate")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--skip-callback-verification",action="store_true",help="Skip potentially large actual callback CSV re-read; explicit flag retained in invocation")
    return analyze(parser.parse_args())


if __name__=="__main__":
    raise SystemExit(main())
