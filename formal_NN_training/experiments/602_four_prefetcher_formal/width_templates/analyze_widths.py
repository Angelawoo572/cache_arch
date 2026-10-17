#!/usr/bin/env python3
"""Audit 24 full supplemental runs and explicitly bind compatible archived controls."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CACHE = next(p for p in HERE.parents if p.name == "cache")
BASE = CACHE / "formal_study"
PROTOCOL = "four602_width_templates_20261017_v1"
REFERENCE_PROTOCOL = "four602_frozen_live_20261017_v2"
GLOBAL = ("streamer", "ampm", "spp")
MODULE_SPEC = importlib.util.spec_from_file_location("archived_four602_analysis_read_only", EXP / "analyze.py")
OLD = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(OLD)
read, save, csv_out, ratio = OLD.read, OLD.save, OLD.csv_out, OLD.ratio


def cache_path(path):
    """Resolve recorded Sacramento paths without rewriting historical evidence."""
    path = Path(path)
    text = str(path)
    prefix = "/home/qianruw/cache/"
    if text.startswith(prefix):
        suffix = text[len(prefix):]
        for old, new in (("formal_study_20261016/", "formal_study/archive/20261016/"),
                         ("formal_study_20261017/", "formal_study/archive/20261017/")):
            if suffix.startswith(old):
                suffix = new + suffix[len(old):]
                break
        return CACHE / suffix
    return path


def configure(out):
    resolved = out.resolve()
    if not resolved.is_relative_to((BASE / "runs").resolve()):
        raise ValueError("Fresh analysis output must be inside cache/formal_study/runs")
    for variable, name in (("TMPDIR", "tmp"), ("PYTHONPYCACHEPREFIX", "pycache"), ("MPLCONFIGDIR", "mpl")):
        path = BASE / "runs" / "602_width_templates_20261017" / "runtime" / "analysis" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def expected_specs(config):
    rows = []
    def one(stage, roi, policy, hidden, method, budget):
        lo, hi = config["validation_records"] if stage == "validation" else config["test_records"][roi]
        rows.append(dict(run_id="{}_r{}_{}_h{}_{}_full_cap{}".format(stage, roi, policy, hidden, method, budget),
                         protocol=config["protocol"], stage=stage, roi=roi, policy=policy, hidden_size=hidden,
                         method=method, mode="full", budget=budget, seed=config["seed"], macs=0,
                         page_policy=config["page_policy"], skip=lo-config["warmup_instructions"],
                         warmup=config["warmup_instructions"], instructions=hi-lo, measurement_records=[lo, hi]))
    for policy in ("ampm", "spp"):
        one("validation", 0, policy, 16, "student", 0)
        for roi in range(2):
            for method in ("student", "template"):
                for budget in (0, config["common_cap"]):
                    one("test", roi, policy, 16, method, budget)
    for policy in GLOBAL:
        for roi in range(2):
            one("test", roi, policy, 8, "template", config["common_cap"])
    return rows


def typed(value):
    if value == "": return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return [{k:typed(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def trace(spec):
    return spec.get("trace", spec.get("contract", {}).get("trace"))


def identity(spec):
    return tuple(json.dumps(spec.get(k), sort_keys=True) for k in
                 ("stage", "roi", "skip", "warmup", "instructions", "measurement_records", "page_policy", "seed", "macs")) + (trace(spec),)


def compatible(new_spec, old_spec, config, audit):
    rid = new_spec["run_id"]
    audit.check(identity(new_spec) == identity(old_spec), "explicit_reuse_measurement_contract_matches", rid,
                new_protocol=new_spec["protocol"], reference_protocol=old_spec["protocol"],
                reference_run=old_spec["run_id"])
    audit.check(old_spec["protocol"] == REFERENCE_PROTOCOL and new_spec["protocol"] == PROTOCOL,
                "reference_protocol_remains_distinct", rid)
    contract = old_spec.get("contract", {})
    audit.check(contract.get("simulator_base") == config["base"], "same_configured_ChampSim_base", rid)
    # Archived v2 has no independent per-run host telemetry. Same configured host
    # plus exact actual binary reuse is not an independent hardware certification.
    audit.check(new_spec.get("build_id") == old_spec.get("build_id"),
                "actual_archived_build_id_reused", rid,
                host_scope="same configured Sacramento; historical per-run hardware telemetry unavailable")


def verify_bindings(raw, reference_raw, bindings, refs, config, audit):
    audit.check(bindings.get("protocol") == PROTOCOL
                and bindings.get("reference_protocol") == REFERENCE_PROTOCOL
                and bindings.get("actual_binary_byte_equal") is True,
                "explicit_reference_protocol_and_binary_binding")
    controls = bindings.get("common_controls", {})
    audit.check(all(controls.get(k) == config[k] for k in
                ("trace", "seed", "macs", "page_policy", "validation_records", "test_records",
                 "warmup_instructions", "phase_instructions", "common_cap")),
                "reference_manifest_common_controls_exact")
    entries = bindings.get("references", [])
    by_id = {item["old_run_id"]:item for item in entries}
    audit.check(len(entries) == 64 and len(by_id) == 64 and set(by_id) == set(refs),
                "reference_manifest_exact_64_actual_runs")
    for rid, run in refs.items():
        entry = by_id.get(rid, {})
        audit.check(entry.get("original_protocol") == REFERENCE_PROTOCOL
                    and cache_path(entry.get("raw_run_directory", "")).resolve() == run["dir"].resolve()
                    and all(run["spec"].get(k) == value for k,value in entry.get("fields", {}).items())
                    and bool(entry.get("fields")),
                    "reference_manifest_matches_actual_original_run", rid)
        for name in ("run.json", "summary.json"):
            copy = raw / "reference_snapshot" / rid / name
            source = run["dir"] / name
            audit.check(copy.is_file() and copy.read_bytes() == source.read_bytes(),
                        "reference_original_bytes_bound_in_supplement", rid, file=name)
    state = read(raw / "bindings.json")
    audit.check(state.get("protocol") == PROTOCOL
                and state.get("build_id") == bindings.get("actual_build_id"),
                "supplemental_actual_build_and_source_binding")
    build = state.get("actual_build", {})
    audit.check(build.get("base") == config["base"] and build.get("compiler_flags") == "-O3 -ffp-contract=off"
                and build.get("build_id") == state.get("build_id") and build.get("protocol") == REFERENCE_PROTOCOL,
                "same_actual_archived_compiler_base_build")
    actual_sim = cache_path(build.get("simulator", ""))
    compiled = cache_path(build.get("compiled_source", ""))
    source_files = sorted(p for p in compiled.rglob("*") if p.is_file()) if compiled.is_dir() else []
    different = []
    for source in source_files:
        actual = actual_sim / source.relative_to(compiled)
        if not actual.is_file() or actual.read_bytes() != source.read_bytes():
            different.append(str(source.relative_to(compiled)))
    audit.check(bool(source_files) and not different, "all_actual_selected_compiled_source_bytes_preserved",
                checked_files=len(source_files), mismatched_files=different)
    sources = state.get("bindings", [])
    audit.check(bool(sources), "supplemental_immutable_byte_bindings_present")
    for item in sources:
        source, snapshot = cache_path(item["source"]), cache_path(item["snapshot"])
        inside = source.resolve().is_relative_to(CACHE.resolve()) and snapshot.resolve().is_relative_to(raw.resolve())
        audit.check(inside and source.is_file() and snapshot.is_file()
                    and source.stat().st_size == item["bytes"] and source.read_bytes() == snapshot.read_bytes(),
                    "actual_source_model_binary_byte_binding", source=str(source), snapshot=str(snapshot))
    gate = read(raw / "correctness_gate.json")
    audit.check(gate.get("status") == "PASS" and gate.get("protocol") == PROTOCOL
                and gate.get("build_id") == state.get("build_id")
                and set(gate.get("reports", {})) == {"ampm", "spp"}
                and all(report.get("status") in ("PASS", "verified") for report in gate.get("reports", {}).values()),
                "actual_H16_Python_CPP_correctness_gate_bound")
    # Historical telemetry limits are retained verbatim. No stronger hardware
    # identity assertion is inferred from a matching software configuration.
    return dict(source_bindings=len(sources), reference_records=len(entries),
                actual_build_id=state.get("build_id"), original_protocol=REFERENCE_PROTOCOL)


def reference_runs(raw, config, audit):
    analysis = raw / "analysis_final"
    source_audit = read(analysis / "audit.json")
    audit.check(source_audit.get("status") == "PASS" and source_audit.get("complete_count") == 64
                and source_audit.get("expected_count") == 64 and source_audit.get("failure_count") == 0
                and source_audit.get("callback_logs_checked") is True
                and source_audit.get("protocol") == REFERENCE_PROTOCOL,
                "archived_64_complete_strict_callback_audit")
    rows = read_csv(analysis / "per_run.csv")
    audit.check(len(rows) == 64 and len({r["run_id"] for r in rows}) == 64,
                "archived_reference_64_unique_rows")
    runs, metrics = {}, []
    for row in rows:
        rid = row["run_id"]
        folder = raw / "runs" / rid
        record = read(folder / "run.json")
        summary = read(folder / "summary.json")
        spec = record["spec"]
        audit.check(record.get("status") == "complete" and record.get("exit_code") == 0
                    and spec.get("protocol") == REFERENCE_PROTOCOL and summary.get("final") is True,
                    "reference_actual_completed_record", rid)
        audit.check(row.get("status") == "complete" and row.get("cycles") == summary["cycles"]
                    and row.get("instructions") == summary["instructions"]
                    and row.get("raw__request_gate_allowed") == summary["request_gate_allowed"]
                    and row.get("raw__dram_total_bus_bytes") == summary["dram_total_bus_bytes"]
                    and math.isclose(row["ipc"], summary["instructions"]/summary["cycles"], rel_tol=1e-14),
                    "reference_csv_matches_actual_summary", rid)
        run = dict(spec=spec, summary=summary, dir=folder, record=record)
        runs[rid] = run
        row.update(result_id=REFERENCE_PROTOCOL+"::"+rid, evidence_origin="archived_64_reused",
                   source_raw=str(raw), source_run=str(folder),
                   reference_audit=str(analysis/"audit.json"),
                   hidden_size=8 if row["method"] in ("student", "template") else row.get("hidden_size"),
                   model_width_scope="retained H8 teacher-specific model or validation-derived template"
                   if row["method"] in ("student", "template") else "not_applicable",
                   hardware_reuse_scope="same configured Sacramento; no historical per-run hardware telemetry")
        metrics.append(row)
    for row in metrics:
        run = runs[row["run_id"]]
        spec = run["spec"]
        baseline = runs["{}_r{}_none_none_full_cap0".format(spec["stage"], spec["roi"])]
        actual = OLD.derive(run, baseline, raw)
        mismatches = []
        for field, value in actual.items():
            if field not in row or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                recorded = row[field]
                matches = (isinstance(recorded, (int, float)) and
                           (recorded == value if isinstance(value, int) else
                            math.isclose(recorded, value, rel_tol=1e-13, abs_tol=1e-15)))
                if not matches: mismatches.append(field)
            elif value is None and row[field] is not None:
                mismatches.append(field)
        audit.check(not mismatches, "reference_all_derived_numeric_metrics_match_actual_summary",
                    row["run_id"], mismatched_fields=mismatches)
    return runs, metrics, source_audit


def validate_summary(run, config, audit):
    s, x, record = run["spec"], run["summary"], run["record"]
    rid = s["run_id"]
    audit.check(record.get("exit_code") == 0, "successful_simulator_exit", rid)
    audit.check(x.get("final") is True and abs(x["instructions"]-s["instructions"]) <= 4
                and x["cycles"] > 0, "actual_full_ROI_final", rid)
    audit.check(x.get("numerical_failures", 0) == 0 and x.get("measurement_unconsumed_context_labels", 0) == 0,
                "no_numeric_or_request_context_failure", rid)
    audit.check(x.get("measurement_cohort_scope") == "separate_L2_and_LLC_target_cohorts",
                "separate_target_cohort_scope", rid)
    audit.check(x.get("four602_method") == s["method"] and x.get("four602_policy") == s["policy"]
                and x.get("four602_mode") == s["mode"], "actual_registered_arm", rid)
    environment = record.get("environment", {})
    audit.check(environment.get("FOUR602_METHOD") == s["method"] and environment.get("FOUR602_POLICY") == s["policy"]
                and environment.get("DYNAMIC602_MACS") == "0", "actual_environment_MAC0_arm", rid)
    wanted_environment = {
        "DYNAMIC602_PAGE_POLICY": s["page_policy"], "DYNAMIC602_SKIP_RECORDS": str(s["skip"]),
        "DYNAMIC602_WARMUP_INSTRUCTIONS": str(s["warmup"]), "DYNAMIC602_HISTORY_INSTRUCTIONS": str(s["warmup"]),
        "DYNAMIC602_ROI_INSTRUCTIONS": str(s["instructions"]), "DYNAMIC602_FILTER_CURRENT": "0",
        "DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS": str(s["budget"]),
        "DYNAMIC602_PHASE_INSTRUCTIONS": str(config["phase_instructions"])}
    audit.check(all(environment.get(k) == v for k,v in wanted_environment.items()),
                "actual_environment_measurement_state_page_credit_contract", rid)
    if s["budget"]:
        audit.check(x["request_gate_allowed"] <= x["instructions"]*s["budget"]//1000000,
                    "request_credit_ceiling_respected", rid)
    c = OLD.counters(x)
    if s["method"] == "student":
        audit.check(c.get("completed_updates", 0) == c.get("demand_callbacks", 0)+c.get("fill_callbacks", 0)
                    and c.get("demand_inferences", 0) == c.get("demand_callbacks", 0)
                    and c.get("fill_inferences", 0) == c.get("fill_callbacks", 0),
                    "one_full_global_forward_per_callback", rid)
        audit.check(x.get("four602_global_state_entries") == 1 and x.get("four602_state_evictions") == 0
                    and x.get("four602_state_bytes") == 8*s["hidden_size"],
                    "actual_H16_global_state", rid)
        if s["policy"] == "spp":
            audit.check(c.get("fill_callbacks", 0) > 0, "SPP_all_raw_L2_fill_inputs", rid)
        distributions = x.get("four602_distributions", {})
        phases = {OLD.PROFILE.fullmatch(name).group(1) for name in distributions if OLD.PROFILE.fullmatch(name)}
        audit.check(bool(phases) and all(distributions.get("phase_"+phase+"_learned_K", {}) == distributions.get("phase_"+phase+"_raw_K", {})
                    and distributions.get("phase_"+phase+"_learned_emit", {}) == distributions.get("phase_"+phase+"_raw_emit", {}) for phase in phases),
                    "full_ROI_learned_K_emit_match_all_decoded_actions_no_truncation", rid)
        model = run["dir"] / "model.bin"
        audit.check(model.is_file(), "per_run_actual_model_copy_present", rid)
        if model.is_file():
            from formal_NN_training.common.live602_runtime.model_format import read_model
            header, _ = read_model(model)
            wanted = config["models"][s["policy"]]
            audit.check(header["hidden_size"] == s["hidden_size"]
                        and header["parameter_count"] == wanted["parameters"],
                        "actual_model_header_H16_parameters", rid)
            schema = (64,0,0,0) if s["policy"] == "ampm" else (65,1,2,1)
            audit.check(tuple(header[k] for k in ("feature_count","kind_feature","fill_classes","count_mode")) == schema,
                        "actual_model_policy_input_count_fill_schema", rid)
            frozen = cache_path(s.get("model", ""))
            audit.check(frozen.is_file() and frozen.read_bytes() == model.read_bytes(),
                        "run_model_exact_frozen_source_bytes", rid)
    else:
        audit.check(c.get("completed_updates", 0) == 0 and c.get("inference_macs", 0) == 0
                    and c.get("inference_nonlinears", 0) == 0, "unconditional_template_has_no_NN_forward", rid)
    for target in ("l2", "llc"):
        OLD.check_cohort(audit, OLD.cohort(x, target), target, rid)
    if x.get("llc_only_lifecycle_accuracy_available") is True:
        audit.check(x.get("llc_target_prefetch_queue_rejected_untracked") == 0
                    and x.get("llc_target_shortcut_missing_pending_id") == 0,
                    "available_LLC_metric_requires_complete_handoff", rid)
    audit.check(x["request_cache_pq_inserted"] == x["cohort_enqueued"]+x["llc_target_cohort"]["enqueued"],
                "target_enqueue_partition", rid)
    audit.check(x["request_gate_generated"] == sum(x["request_gate_"+k] for k in
                ("allowed", "budget_suppressed", "page_suppressed", "current_suppressed")),
                "request_gate_conservation", rid)
    audit.check(x["request_gate_allowed"] == x["request_cache_enqueued"]+x["request_cache_merged"]+x["request_cache_rejected"],
                "request_cache_result_conservation", rid)
    for suffix in ("generated", "allowed", "cache_merged", "cache_rejected"):
        total = "request_gate_"+suffix if suffix in ("generated", "allowed") else "request_"+suffix
        audit.check(x["request_l2_target_"+suffix]+x["request_llc_only_"+suffix] == x[total],
                    "L2_LLC_partition", rid, field=suffix)
    line = x["line_bytes"]
    audit.check(x["dram_total_bus_bytes"] == x["dram_read_bus_bytes"]+x["dram_write_bus_bytes"]
                and x["dram_read_bus_bytes"] == line*x["dram_read_bus_completed_transactions"]
                and x["dram_write_bus_bytes"] == line*x["dram_write_bus_completed_transactions"],
                "actual_completed_DRAM_bytes", rid)
    audit.check(x["dram_read_bus_started_bytes"]+x["dram_write_bus_started_bytes"]-x["dram_total_bus_bytes"]
                == line*x["dram_bus_transfers_in_progress"], "DRAM_boundary_bus_conservation", rid)


def template_from_summary(run):
    counts, slots = Counter(), defaultdict(Counter)
    for name, hist in run["summary"]["four602_distributions"].items():
        match = OLD.PROFILE.fullmatch(name)
        if not match:
            raise ValueError("Unknown distribution "+name)
        phase, stream, statistic = match.groups()
        if stream != "raw":
            continue
        if statistic == "K":
            counts.update({int(k):v for k,v in hist.items()})
        if statistic.startswith("delta_fill_"):
            fill, slot = map(int, statistic[len("delta_fill_"):].split("_slot_"))
            slots[slot].update({(int(k), fill):v for k,v in hist.items()})
    if not counts or not slots:
        raise ValueError("Missing full-validation count/delta/fill histogram")
    modal = min(counts, key=lambda k:(-counts[k], k))
    if set(slots) != set(range(max(slots)+1)):
        raise ValueError("Noncontiguous validation slots")
    chosen = [list(min(slots[i], key=lambda k:(-slots[i][k], k))) for i in range(max(slots)+1)]
    return dict(modal_count=modal, slots=chosen, source_run=run["spec"]["run_id"])


def parse_template(path):
    words = Path(path).read_text().split()
    if len(words) < 4 or words[0] != "modal_count" or (len(words)-2) % 2:
        raise ValueError("Invalid template file "+str(path))
    return dict(modal_count=int(words[1]), slots=[[int(words[i]), int(words[i+1])] for i in range(2,len(words),2)])


def verify_selection(selection, new_runs, ref_runs, config, audit):
    audit.check(selection.get("protocol") == PROTOCOL and selection.get("common_cap") == config["common_cap"]
                and selection.get("validation_records") == config["validation_records"],
                "selection_fixed_cap_and_validation_interval")
    old_selection = read(Path(ref_runs["validation_r0_none_none_full_cap0"]["dir"]).parents[1] / "selection.json")
    audit.check(config["common_cap"] == old_selection["common_cap"], "cap_inherited_without_retuning")
    output = dict(source="full_validation_only; no test fitting", cap_source="archived_validation_NoPF",
                  common_cap=config["common_cap"], templates={})
    for policy in ("ampm", "spp"):
        rid = "validation_r0_{}_h16_student_full_cap0".format(policy)
        if rid not in new_runs:
            continue
        actual = template_from_summary(new_runs[rid])
        saved = selection["templates"][policy+"_h16"]
        audit.check(all(saved.get(k) == actual[k] for k in ("modal_count", "slots", "source_run"))
                    and saved.get("hidden_size") == 16 and saved.get("policy") == policy,
                    "H16_template_recomputed_from_entire_own_validation", rid)
        output["templates"][policy+"_h16"] = actual
    for run in new_runs.values():
        spec = run["spec"]
        if spec["method"] != "template":
            continue
        key = spec["policy"]+"_h"+str(spec["hidden_size"])
        expected = output["templates"].get(key) if spec["hidden_size"] == 16 else old_selection["templates"][spec["policy"]]
        if expected is None:
            continue
        path = run["dir"] / "template.txt"
        if not path.is_file():
            path = cache_path(spec.get("template_source", spec.get("template", "")))
        audit.check(path.is_file(), "actual_frozen_template_present", spec["run_id"])
        if path.is_file():
            actual = parse_template(path)
            audit.check(spec.get("frozen_template") == path.read_text(), "run_record_exact_template_bytes", spec["run_id"])
            audit.check(all(actual[k] == expected[k] for k in ("modal_count", "slots")),
                        "actual_template_exact_validation_selection", spec["run_id"], template_key=key)
    return output


def attach_h16_cost(metrics, raw, audit, complete):
    path = raw / "cost/cost_fields_h16.json"
    if not path.is_file():
        if complete: audit.check(False, "final_H16_actual_storage_cost_record_present")
        else: audit.notes.append("H16 host-allocation probe still pending; partial analysis reports only actual binary weight/hc payload.")
        return dict(available=False, source=str(path))
    cost = read(path)
    state = read(raw / "bindings.json")
    audit.check(cost.get("status") == "PASS" and cost.get("protocol") == PROTOCOL
                and cost.get("actual_build", {}).get("build_id") == state["build_id"]
                and cost.get("no_simulator_executed") is True
                and set(cost.get("nn_models", {})) == {"ampm_h16", "spp_h16"},
                "H16_cost_separate_width_keys_actual_build_and_PASS")
    probe_bindings = cost.get("bindings", [])
    audit.check(bool(probe_bindings), "H16_cost_source_model_fixture_bindings_present")
    for item in probe_bindings:
        source, snapshot = cache_path(item["source"]), cache_path(item["snapshot"])
        audit.check(source.is_file() and snapshot.is_file() and source.resolve().is_relative_to(CACHE.resolve())
                    and snapshot.resolve().is_relative_to((raw/"cost").resolve())
                    and source.stat().st_size == item["bytes"] and source.read_bytes() == snapshot.read_bytes(),
                    "H16_cost_actual_source_model_fixture_bytes", source=str(source), snapshot=str(snapshot))
    attached = []
    for row in metrics:
        if row.get("evidence_origin") != "new_supplemental_run" or row["method"] != "student":
            continue
        key = row["policy"]+"_h"+str(row["hidden_size"])
        source = cost.get("nn_models", {}).get(key)
        if not source:
            audit.check(False, "H16_cost_exact_policy_width_key", row["run_id"], requested_key=key)
            continue
        model, host = source["model"], source["host_allocations"]
        exact = (model["hidden_size"] == row["hidden_size"] == 16
                 and model["parameters"] == row["model_parameter_count"]
                 and model["fp32_weight_bytes"] == row["model_weight_bytes"]
                 and model["global_hc_bytes"] == row["recurrent_hc_bytes"] == 128
                 and model["weight_plus_global_hc_payload_bytes"] == row["model_weight_bytes"]+row["recurrent_hc_bytes"])
        audit.check(exact, "H16_cost_matches_actual_per_run_binary_and_state", row["run_id"], cost_key=key)
        fixture = source["fixture"]
        audit.check(host["residual_after_runtime_destruction"] == 0
                    and fixture["real_contiguous_callbacks"] == 4096
                    and fixture["callbacks"] == 4096+fixture["boundary_callbacks"]
                    and fixture["bound_to_passed_correctness_sequence"] is True
                    and source["sizeof"]["live602::Action"] == row.get("actual_template_Action_ABI_bytes_per_slot",32),
                    "H16_actual_host_probe_fixture_and_lifecycle", row["run_id"])
        if not exact: continue
        row.update(model_weight_plus_hc_payload_bytes=model["weight_plus_global_hc_payload_bytes"],
                   model_payload_qualifier=model["payload_qualifier"],
                   host_runtime_object_plus_persistent_requested_bytes=host["object_plus_persistent_requested_bytes"],
                   host_persistent_includes_weights_and_hc=True,
                   host_observed_infer_incremental_peak_requested_bytes=host["max_infer_incremental_peak_requested_bytes"],
                   host_observed_fixture_max_K=source["fixture"]["max_count"],
                   host_allocation_scope=source["scope"],cost_record_source=str(path),cost_model_key=key)
        attached.append(row["run_id"])
    return dict(available=True, source=str(path), separate_H16_record=True, attached_runs=attached,
                scope="actual scalar runtime ordinary new/new[] requested allocations on4096 causal correctness callbacks plus documented boundaries; observed peak is not unbounded-degree worst case; excludes outer adapter/observer/stack/allocator/RSS/PPA")


def comparisons(metrics, audit):
    selected = [r for r in metrics if r["stage"] == "test"]
    lookup = {}
    for r in selected:
        if r["policy"] not in GLOBAL or r["method"] not in ("native", "student", "template") or r["mode"] != "full":
            continue
        h = r.get("hidden_size") if r["method"] != "native" else 0
        key = (r["roi"],r["policy"],h,r["method"],r["budget"])
        audit.check(key not in lookup, "unique_comparison_arm", r["run_id"], key=key)
        lookup[key] = r
    result = []
    def pair(kind, control_key, candidate_key):
        if control_key not in lookup or candidate_key not in lookup:
            return
        a, b = lookup[control_key], lookup[candidate_key]
        exact = all(a.get(k) == b.get(k) for k in ("stage","roi","skip","warmup","measurement_records","seed","macs","page_policy"))
        audit.check(exact and a.get("no_pf_run") == b.get("no_pf_run"), "paired_measurement_and_NoPF_identity", b["run_id"], comparison=kind)
        result.append(dict(comparison=kind,policy=b["policy"],roi=b["roi"],budget=b["budget"],seed=b["seed"],
            control_result_id=a["result_id"],candidate_result_id=b["result_id"],
            control_protocol=a["protocol"],candidate_protocol=b["protocol"],
            control_origin=a["evidence_origin"],candidate_origin=b["evidence_origin"],
            control_hidden_size=a.get("hidden_size"),candidate_hidden_size=b.get("hidden_size"),
            control_method=a["method"],candidate_method=b["method"],
            control_ipc=a["ipc"],candidate_ipc=b["ipc"],candidate_over_control_ipc=ratio(b["ipc"],a["ipc"]),
            delta_ipc=b["ipc"]-a["ipc"],
            control_allowed_requests=a["raw__request_gate_allowed"],candidate_allowed_requests=b["raw__request_gate_allowed"],
            delta_allowed_requests=b["raw__request_gate_allowed"]-a["raw__request_gate_allowed"],
            control_DRAM_bus_bytes=a["raw__dram_total_bus_bytes"],candidate_DRAM_bus_bytes=b["raw__dram_total_bus_bytes"],
            delta_DRAM_bus_bytes=b["raw__dram_total_bus_bytes"]-a["raw__dram_total_bus_bytes"],
            scope="complete paired ROI, same reused actual binary/configuration; common cap is not equal realized traffic; no finite-cost IPC"))
    for roi in (0,1):
        for policy in GLOBAL:
            for h in (8,16):
                for cap in (0,16144):
                    pair("template_vs_full_student",(roi,policy,h,"student",cap),(roi,policy,h,"template",cap))
                    pair("student_vs_native",(roi,policy,0,"native",cap),(roi,policy,h,"student",cap))
                for method in ("student","template"):
                    pair("capped_vs_uncapped",(roi,policy,h,method,0),(roi,policy,h,method,16144))
            if policy in ("ampm","spp"):
                for method in ("student","template"):
                    for cap in (0,16144):
                        pair("H16_vs_H8",(roi,policy,8,method,cap),(roi,policy,16,method,cap))
    return result


def annotate(run, row, raw, baseline):
    s = run["spec"]
    row.update(result_id=PROTOCOL+"::"+s["run_id"], evidence_origin="new_supplemental_run",
               source_raw=str(raw), source_run=str(run["dir"]), hidden_size=s["hidden_size"],
               no_pf_protocol=REFERENCE_PROTOCOL, baseline_reuse_is_explicit=True,
               model_width_scope="same teacher/data/training budget retained width checkpoint, not independent seed",
               hardware_reuse_scope="same configured Sacramento; historical per-run hardware telemetry unavailable")
    if s["method"] == "template":
        row.update(model_weight_bytes=0, recurrent_hc_bytes=0, model_parameter_count=0,
                   inference_macs=0, inference_nonlinears=0, model_weight_dram_bytes=0,
                   state_organization="unconditional_frozen_per_slot_delta_fill_table; no NN state",
                   storage_scope="no NN weights or h/c; template fields separate, adapter/control/container/allocator excluded")
        path = run["dir"]/"template.txt"
        if not path.exists():
            path = cache_path(s.get("template_source", s.get("template","")))
        if path.is_file():
            template = parse_template(path)
            row.update(template_stored_slots=len(template["slots"]), template_modal_count=template["modal_count"],
                       template_prediction_fields_bytes=8+12*len(template["slots"]),
                       template_prediction_fields_scope="essential modalK/delta/fill scalar payload only; excludes unused Action fields, containers and alignment; not actual complete host storage",
                       template_declared_Action_fields_bytes=8+28*len(template["slots"]),
                       template_declared_Action_fields_scope="actual source fields: uint64 modalK plus Action {uint64 address,int64 delta,int fill,int fill_index,float coordinate}; excludes padding/vector capacity/control",
                       template_logical_Action_ABI_plus_modal_bytes=8+32*len(template["slots"]),
                       actual_template_Action_ABI_bytes_per_slot=32,
                       actual_template_Action_ABI_scope="measured archived 64-bit ABI; excludes vector descriptor/capacity/slack/control/allocator",
                       template_source=str(path))
    row["mac4_aggregate_dense_over_measured_MAC0_window"] = ratio(row["mac4_aggregate_dense_lower_bound_cycles"],row["cycles"])
    row["weight_plus_hc_payload_bytes"] = (row.get("model_weight_bytes") or 0)+(row.get("recurrent_hc_bytes") or 0)
    return row


def report(out, audit_record, metrics, pairs):
    new = [r for r in metrics if r["evidence_origin"] == "new_supplemental_run"]
    test = [r for r in new if r["stage"] == "test"]
    fmt = lambda value: "NA" if value is None else "{:.5f}".format(value)
    lines = ["# Width and unconditional-template supplemental study", "",
             "Status: **{}**. New complete runs: {}/24 (22 test + 2 validation required). Archived 64-row dataset is retained separately with its original protocol.".format(audit_record["status"],len(new)), "",
             "All measured IPC uses synchronous MAC0 inference. H16 uses retained real seed7 checkpoints, not retraining or extra seeds. H16 templates use only their own full20–25M validation outputs. H8 capped templates reuse the original validation-selected H8 tables. Cap16144 is inherited unchanged; it does not equalize realized traffic.", "",
             "| Policy | Width | ROI | Arm | Cap | IPC / NoPF | Actual requests/M | DRAM MiB | L2 LOAD accuracy | LLC LOAD accuracy |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(test,key=lambda x:(x["policy"],x["roi"],x["hidden_size"],x["method"],x["budget"])):
        lines.append("| {} | {} | {} | {} | {} | {} | {:.2f} | {:.3f} | {} | {} |".format(r["policy"],r["hidden_size"],r["roi"],r["method"],r["budget"],fmt(r["ipc_speedup_vs_NoPF"]),r["requests_per_million_instructions"],r["raw__dram_total_bus_bytes"]/2**20,fmt(r["l2_target_LOAD_accuracy_enqueued"]),fmt(r["llc_target_LOAD_accuracy_enqueued"])))
    models = {}
    for row in metrics:
        if row["method"] == "student" and row["mode"] == "full": models[(row["policy"],row["hidden_size"])]=row
    lines += ["", "| Actual NN | FP32 weights B | h/c B | Object + persistent requested heap B | Observed extra Infer heap peak B |",
              "|---|---:|---:|---:|---:|"]
    for (policy,h),row in sorted(models.items()):
        lines.append("| {} H{} | {} | {} | {} | {} |".format(policy,h,row.get("model_weight_bytes"),row.get("recurrent_hc_bytes"),
            row.get("host_runtime_object_plus_persistent_requested_bytes","NA"),row.get("host_observed_infer_incremental_peak_requested_bytes","NA")))
    lines += ["", "Host persistent measurement already includes weights and h/c; do not add them again. H8 heap peaks cover its recorded4096 real callbacks; H16 covers4096 real callbacks plus8/16 documented boundaries. Neither is an all-degree worst-case bound. Outer Four602/controller/observer, stack, allocator metadata, RSS and hardware PPA are excluded.", ""]
    tables = {(row["policy"],row["hidden_size"]):row for row in test if row["method"] == "template"}
    if tables:
        lines += ["| Frozen table | Retained slots | Essential prediction fields B | Actual Action scalar fields + modalK B | Padded logical Action array + modalK B |",
                  "|---|---:|---:|---:|---:|"]
        for (policy,h),row in sorted(tables.items()):
            lines.append("| {} H{} | {} | {} | {} | {} |".format(policy,h,row.get("template_stored_slots"),row.get("template_prediction_fields_bytes"),row.get("template_declared_Action_fields_bytes"),row.get("template_logical_Action_ABI_plus_modal_bytes")))
        lines += ["", "The essential payload is a semantic lower-level field inventory, not the current host allocation. Actual Action has address/delta/fill/fill_index/coordinate fields and32B ABI size. Vector descriptor/capacity/slack and outer control are unmeasured; no complete hardware table size is claimed.", ""]
    widths = [r for r in pairs if r["comparison"] == "H16_vs_H8" and r["candidate_method"] == "student"]
    templates = [r for r in pairs if r["comparison"] == "template_vs_full_student"]
    def count(rows):
        return "{} positive, {} negative, {} tied".format(sum(r["delta_ipc"]>0 for r in rows),sum(r["delta_ipc"]<0 for r in rows),sum(r["delta_ipc"]==0 for r in rows))
    lines += ["", "| Claim | Evidence status | Evidence and limit |", "|---|---|---|",
              "| H16 uniformly improves over H8 | {} | {}; all paired ROI/cap cases retained. Width comparisons use the same existing seed, not independent seed robustness. |".format("未支持" if any(r["delta_ipc"]<0 for r in widths) else "数据支持（仅本矩阵）" if widths else "待验证",count(widths)),
              "| Learned payload/state is necessary for the observed benefit | {} | Unconditional template vs full NN: {}. A template near/better than NN is a negative result for necessity; failure to match does not itself establish stateful novelty. |".format("未支持" if any(r["delta_ipc"]>=0 for r in templates) else "待验证",count(templates)),
              "| Gains survive finite compute | 待验证 | MAC0 IPC only. MAC4 aggregate dense work/window is a necessary throughput estimate, without queue, latency, nonlinear throughput or finite IPC simulation. |",
              "| Three independently trained seeds generalize | 待验证 | Existing seed7 only, one workload, two distinct complete25M ROIs. |", "",
              "L2/LLC target cohorts remain separate. Accuracy is target LOAD-useful/enqueued; timeliness is timely LOAD-useful/all LOAD-useful. Useful/NoPF-misses is an opportunity ratio, not causal total coverage. LLC metrics become NA when tracking is incomplete. Pending/resident-unused and evicted-unused remain distinct; unused is not pollution. DRAM is actual completed controller bus bytes, never requests×64.", "",
              "Native SPP FILTER retains prefetch-fill-only feedback; student SPP consumes all raw L2 fill callbacks. Same effective input is not claimed. Full ROI25–50M was already inspected before this supplement;75–100M remains a distinct interval but the supplement is post-inspection analysis. No prefix stands for a whole ROI. The startup plots do not identify a minimum history by first baseline crossing.", "",
              "[New runs](per_run_new.csv), [all explicitly labeled rows](per_run.csv), [unaltered original64 CSV](reference_per_run_original.csv), [paired contrasts](paired_comparisons.csv), [audit](audit.json), [template selection checks](selection_verification.json), [callback checks](callback_verification.json), [storage/cost](storage_cost.csv).", ""]
    lines += ["[One-page paper story](PAPER_STORY.md)", "", "![Actual NN and table routes](method.svg)", ""]
    if (out/"figure_manifest.json").is_file():
        lines += ["![{}]({}.png)".format(name,name) for name in read(out/"figure_manifest.json")["figures"]]
    (out/"FINDINGS.md").write_text("\n".join(lines)+"\n")


def metric_definitions():
    result = OLD.definitions()
    result.update(protocol=PROTOCOL,
        reference_protocol=REFERENCE_PROTOCOL,
        reference_reuse="The original64 rows retain their original protocol, run_id and bytes. New24 rows have new result identities. Only explicit same actual binary, CLI, trace/ROI/cachewarmup/page controls are joined; old per-run independent hardware telemetry is absent.",
        request_budget="Inherited cap16144 allowed requests/M, originally chosen from H8-protocol validation NoPF eligible L2 LOADs; not selected again using H16 or tests. The same ceiling does not equalize actual allowed traffic or DRAM.",
        width="Existing true seed7 H8/H16 AMPM/SPP checkpoints: same per-policy source data and training budget. Width is not an additional independent training seed. Sibling replay/offline IPC is kept outside live comparisons.",
        templates="H16 tables fit only complete H16 validation20–25M raw output: modalK plus joint per-slot modal(delta,fill), numeric ascending tie breaks. H8 tables are original H8 validation choices. Unconditional table performs no NN gate/forward/hc update. H16 and H8 table contents are distinct when selected validation outputs differ.",
        state="Full global H16 models perform one chronological update for each eligible L2 LOAD and, for SPP, every raw L2 fill; h/c is128B, no per-PC state, FIFO, decoder cap or external execution truncation. Full learnedK/emit histograms must equal all raw decoded action histograms phase by phase. h/c inherits5M cache warmup.",
        storage="NN weight bytes derive from actual per-run portable model headers; h/c from actual summary state fields. Those are partial resident payload, not complete deployment storage. H16 host heap/scratch fields, when present, come from the separate4096-callback correctness-fixture probe and are not universal degree/working-set bounds. Table essential prediction fields use8B modalK+12B delta/fill per retained slot; this is not actual host storage. Actual Action source scalar fields total28B/slot, padded to32B/slot in the measured archived ABI, plus8B modalK; vector capacity/descriptor/control/allocator are excluded and unmeasured here. No hardware-optimized SRAM total is inferred.",
        causality="Width and unconditional-template are actual whole-run interventions. Changed prefetch actions can change future cache callbacks and full NN trajectories. Table performance constrains necessity claims; a nonmatching bounded table does not itself establish stateful novelty.",
        observation_axes="All startup points are saved actual retired-instruction/cycle coordinates. Cumulative first crossing is not minimum required history. Phase distributions cover entire callback-assigned5M bins; no prefix substitutes for a complete ROI.",
        post_inspection="Original25–50M ROI was previously observed,75–100M is a distinct interval; the width/template supplement was specified after inspecting the prior study. No untouched-test or independent-seed robustness claim is made.")
    return result


def paper_story(out, record, metrics, pairs):
    width = [r for r in pairs if r["comparison"] == "H16_vs_H8" and r["candidate_method"] == "student"]
    table = [r for r in pairs if r["comparison"] == "template_vs_full_student" and r["candidate_hidden_size"] == 16]
    def contrast(rows):
        if not rows: return "No complete test contrasts yet."
        changes = [100*(r["candidate_over_control_ipc"]-1) for r in rows]
        return "{} positive / {} negative / {} tied; IPC change {:+.4f}% to {:+.4f}%".format(
            sum(v>0 for v in changes),sum(v<0 for v in changes),sum(v==0 for v in changes),min(changes),max(changes))
    lines = ["# Concise paper story: widths and unconditional templates", "",
             "Status: **{}**; new complete formal runs {}/24. This is a post-inspection supplemental mechanism study on602, using retained real seed7 H8/H16 teacher-action models and explicit compatible archived controls.".format(record["status"],record["complete_count"]), "",
             "The question is whether added recurrent width improves the same teacher-action student, and whether its learned decisions are needed after a frozen validation-derived unconditional payload table is available. The independent online NN and the table are actual simulation interventions; request labels alone do not establish a performance cause.", "",
             "| Statement | Evidence status | Measured support / limit |", "|---|---|---|",
             "| H16 uniformly improves over H8 | {} | {}. Same seed/data/training budget within policy; no independent-seed claim. |".format("未支持" if any(r["delta_ipc"]<0 for r in width) else "数据支持（仅本矩阵）" if width else "待验证",contrast(width)),
             "| Learned payload/state is universally necessary for the full-NN benefit | {} | H16 unconditional table vs NN: {}. Failure of one table to match is not proof of stateful novelty. |".format("未支持" if any(r["delta_ipc"]>=0 for r in table) else "待验证",contrast(table)),
             "| Common ceiling gives equal realized traffic | 未支持 | Cap16144/M is inherited from original validation NoPF, while actual allowed requests and completed DRAM bytes are separately measured. |",
             "| Benefits survive limited inference resources | 待验证 | IPC is synchronous MAC0; actual dense/nonlinear counts and FP32+h/c storage are reported. Four-MAC aggregate work/window is a necessary throughput condition, not finite-cost IPC. |",
             "| Three seeds / broader workload generality | 待验证 | One retained seed7,602, two complete25M test ROIs; no retraining,605, online updates or engine redesign. |", "",
             "Protocol: train0–20M, fit templates only on each matching full20–25M validation output; tests25–50M and75–100M, each5M common cache warmup. ROI0 was already observed; the supplemental comparisons are post-inspection. H8 fixed tables use the original H8 validation selection; H16 uses its own modalK and joint per-slot delta/fill. No test fitting, threshold selection or prefix extrapolation.", "",
             "Native SPP keeps prefetch-fill FILTER feedback; its NN observes all raw L2 fill callbacks. Target-L2 and target-LLC LOAD cohort accuracy/timeliness remain separate, with NA for incomplete tracking. Useful/NoPF-miss ratios are not causal miss reduction. Pending/resident-unused is retained and unused is not called pollution.", "",
             "[All paired outcomes](paired_comparisons.csv), [per-run IPC/requests/DRAM/cohorts](per_run.csv), [new24](per_run_new.csv), [audit](audit.json), [actual distributions](action_histograms_full_roi.csv), [phase distributions](action_histograms.csv), [storage and counted costs](storage_cost.csv), [full findings](FINDINGS.md).", "",
             "![Actual frozen NN and unconditional-table routes](method.svg)", ""]
    (out/"PAPER_STORY.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--reference-raw",type=Path,default=BASE/"archive/20261017/raw_v2")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=HERE/"config.json")
    p.add_argument("--partial",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args()
    raw, ref, out = args.raw.resolve(), args.reference_raw.resolve(), args.out.resolve()
    configure(out)
    if out.exists(): raise ValueError("Preserve all prior analysis; choose fresh --out")
    config=read(args.config)
    audit=OLD.Audit()
    audit.check(config.get("protocol")==PROTOCOL and config.get("macs")==0 and config.get("seed")==7
                and config.get("common_cap")==16144, "fixed_supplemental_protocol")
    audit.check(config["validation_records"]==[20000000,25000000]
                and config["test_records"]==[[25000000,50000000],[75000000,100000000]]
                and config["warmup_instructions"]==5000000,"fixed_full_formal_intervals")
    expected=expected_specs(config);by_id={s["run_id"]:s for s in expected}
    matrix=read(raw/"matrix.json")
    supplied=matrix.get("runs",matrix.get("specs",[])) if isinstance(matrix,dict) else matrix
    supplied=[s.get("spec",s) for s in supplied]
    audit.check(len(supplied)==24 and {s["run_id"] for s in supplied}==set(by_id),
                "driver_exact_24_run_matrix")
    for s in supplied:
        if s["run_id"] in by_id:
            audit.check(all(s.get(k)==v for k,v in by_id[s["run_id"]].items()),"driver_matrix_expected_spec",s["run_id"])
    bindings=read(raw/"reference_bindings.json")
    audit.check(isinstance(bindings,dict) and bool(bindings),"explicit_reference_bindings_retained")
    refs, metrics, ref_audit=reference_runs(ref,config,audit)
    binding_check=verify_bindings(raw,ref,bindings,refs,config,audit)
    new_runs={};inventory=[];statuses=Counter()
    for path in sorted((raw/"runs").glob("*/run.json")):
        record=read(path);s=record["spec"];rid=s["run_id"]
        inventory.append(dict(source=str(path),status=record.get("status"),spec=s));statuses[record.get("status","unknown")]+=1
        audit.check(path.parent.name==rid and rid in by_id,"new_run_path_and_registered_id",rid)
        if rid in by_id:
            audit.check(all(s.get(k)==v for k,v in by_id[rid].items()),"exact_new_formal_spec",rid)
        if record.get("status")!="complete": continue
        run=dict(spec=s,summary=read(path.parent/"summary.json"),dir=path.parent,record=record)
        validate_summary(run,config,audit)
        new_runs[rid]=run
    missing=sorted(set(by_id)-set(new_runs))
    hists=[];classified=[];snaps=[];phases=[];callbacks=[];raw_rows=[];new_metrics=[]
    for rid,run in new_runs.items():
        s=run["spec"];base=refs["{}_r{}_none_none_full_cap0".format(s["stage"],s["roi"])]
        compatible(s,base["spec"],config,audit)
        audit.check(run["record"].get("command", [])[1:] == base["record"].get("command", [])[1:]
                    and bool(run["record"].get("command")),
                    "same_recorded_native_knobs_trace_and_ROI_CLI", rid)
        row=annotate(run,OLD.derive(run,base,raw),raw,base)
        new_metrics.append(row);metrics.append(row)
        raw_rows.append(dict(run_id=rid,**OLD.flatten(run["summary"])))
        hists.extend(OLD.profile_rows(run["summary"],s,audit))
        classified.extend(OLD.classification_rows(run["summary"],s,audit))
        seq=OLD.snapshot_rows(run,audit);snaps.extend(seq);phases.extend(OLD.phase_progress(seq,s,config))
        callbacks.append(OLD.verify_callbacks(run,audit))
    selection=verify_selection(read(raw/"selection.json"),new_runs,refs,config,audit) if (raw/"selection.json").exists() else {"available":False}
    if not missing:
        audit.check(selection.get("templates",{}).keys()=={"ampm_h16","spp_h16"},"both_H16_templates_validated")
    cost_check=attach_h16_cost(metrics,raw,audit,complete=not missing)
    pairs=comparisons(metrics,audit)
    status="FAIL" if audit.failures else "INCOMPLETE" if missing or len(new_runs)!=24 else "PASS"
    record=dict(status=status,protocol=PROTOCOL,reference_protocol=REFERENCE_PROTOCOL,
                expected_count=24,complete_count=len(new_runs),new_test_count=sum(r["stage"]=="test" for r in new_metrics),
                new_validation_count=sum(r["stage"]=="validation" for r in new_metrics),
                reference_count=len(refs),missing_or_incomplete_run_ids=missing,run_status_counts=dict(statuses),
                failure_count=len(audit.failures),checks=audit.checks,failures=audit.failures,notes=audit.notes,
                callback_logs_checked=True,plot_status="disabled_by_flag" if args.no_plots else "pending",
                actual_full_ROI_only=True,raw_inputs_modified=False,raw_root=str(raw),reference_raw=str(ref),
                reference_data_rewritten=False,hardware_scope="same configured Sacramento, reused actual binary; historical per-run host telemetry absent")
    out.mkdir(parents=True)
    method_root = EXP.parents[2] / "docs/602_width_templates"
    for extension in ("svg", "png"):
        shutil.copyfile(method_root/("method."+extension),out/("method."+extension))
    save(out/"audit.json",record);save(out/"inventory.json",inventory);save(out/"raw_snapshots.json",{r:run["summary"] for r,run in new_runs.items()})
    save(out/"reference_bindings.json",bindings);save(out/"binding_verification.json",binding_check);save(out/"selection_verification.json",selection);save(out/"callback_verification.json",callbacks)
    save(out/"metric_definitions.json",metric_definitions());save(out/"cost_attachment_verification.json",cost_check)
    if cost_check.get("available"):shutil.copyfile(raw/"cost/cost_fields_h16.json",out/"cost_fields_h16.json")
    shutil.copyfile(ref/"analysis_final/cost_fields.json",out/"cost_fields_h8_original.json")
    shutil.copyfile(ref/"analysis_final/per_run.csv",out/"reference_per_run_original.csv")
    shutil.copyfile(ref/"analysis_final/audit.json",out/"reference_audit_original.json")
    csv_out(out/"per_run_new.csv",new_metrics);csv_out(out/"per_run.csv",metrics);csv_out(out/"raw_flat_new.csv",raw_rows)
    csv_out(out/"route_status.csv",[dict(policy=policy,
        new_complete_runs=sum(r["policy"]==policy for r in new_metrics),
        new_test_runs=sum(r["policy"]==policy and r["stage"]=="test" for r in new_metrics),
        new_validation_runs=sum(r["policy"]==policy and r["stage"]=="validation" for r in new_metrics),
        original_reference_rows=sum(r["policy"]==policy and r["evidence_origin"]=="archived_64_reused" for r in metrics),
        width_scope="retained H16 plus H8 capped template" if policy in ("ampm","spp") else "H8 capped template only" if policy=="streamer" else "reference only; no new Stride run or width claim",
        original_protocol=REFERENCE_PROTOCOL,new_protocol=PROTOCOL) for policy in ("streamer","ampm","spp","stride")])
    csv_out(out/"paired_comparisons.csv",pairs);csv_out(out/"action_histograms.csv",hists)
    aggregate=Counter()
    for item in hists: aggregate[(item["run_id"],item["stream"],item["statistic"],item["value"])]+=item["count"]
    totals=Counter()
    for (rid,stream,stat,value),count in aggregate.items():totals[(rid,stream,stat)]+=count
    csv_out(out/"action_histograms_full_roi.csv",[dict(run_id=rid,stream=stream,statistic=stat,value=value,count=count,
        fraction=ratio(count,totals[(rid,stream,stat)]),scope="all_actual_full_ROI_phases") for (rid,stream,stat,value),count in sorted(aggregate.items())])
    csv_out(out/"classified_requests.csv",classified);csv_out(out/"progress_snapshots.csv",snaps);csv_out(out/"phase_progress.csv",phases)
    csv_out(out/"storage_cost.csv",[{k:v for k,v in r.items() if k in ("result_id","run_id","protocol","evidence_origin","policy","method","mode","stage","roi","seed","budget","hidden_size","cycles","completed_updates","inference_macs","inference_nonlinears") or any(t in k for t in ("model_","recurrent_","storage_","mac4_","global_state","weight_","state_capacity","state_evictions","host_","native_source_cost","cost_record","template_"))} for r in metrics])
    if not args.no_plots and status=="PASS":
        spec=importlib.util.spec_from_file_location("width_template_figures",HERE/"plot_widths.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.plot(out,metrics,pairs,ref/"analysis_final")
        record["plot_status"]="generated";save(out/"audit.json",record)
    elif not args.no_plots:
        record["plot_status"]="not_generated_incomplete_or_failed";save(out/"audit.json",record)
    report(out,record,metrics,pairs)
    paper_story(out,record,metrics,pairs)
    print(json.dumps({k:record[k] for k in ("status","complete_count","expected_count","failure_count","plot_status")}))
    return 1 if audit.failures else 0 if status=="PASS" or args.partial else 2


if __name__=="__main__":
    sys.path.insert(0,str(EXP.parents[2]))
    raise SystemExit(main())
