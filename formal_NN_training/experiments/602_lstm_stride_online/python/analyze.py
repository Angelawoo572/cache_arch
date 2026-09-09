#!/usr/bin/env python3
"""Compact measured tables for the causal 602 online experiment.

Legacy ratios describe counter activity in the stated interval. Cohort outcomes
are separate fields: a request still in flight or resident-unused is censored.
No run is considered finished merely because a directory or metadata exists.
"""
import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORACLE = ROOT / "formal_NN_training/experiments/602_offline_lstm_stride/python/analyze_replay.py"
COUNTERS = ("l2_loads", "l2_load_miss", "pf_requested", "pf_issued", "pf_useful", "pf_late", "pf_filled", "pf_useless", "pf_dropped", "pq_merged_duplicate_proxy")
PROGRESS = ("eligible_callbacks", "admitted_decisions", "completed_decisions", "supervised_decisions", "available_positive_decisions", "positive_actions", "completed_updates", "training_sample_exposures", "published_versions", "state_evictions", "input_drops", "output_drops", "training_drops", "decoder_limit_drops", "numerical_failures")
ALIASES = {
    "instructions": ("instructions", "retired", "retired_instructions", "roi_instructions"),
    "cycles": ("cycles", "cycle", "roi_cycles"),
    "l2_loads": ("l2_loads", "NL", "l2_demands", "demand_accesses", "loads"),
    "l2_load_miss": ("l2_load_miss", "M", "l2_misses", "demand_misses", "load_misses"),
    "pf_requested": ("pf_requested", "P", "requested", "prefetch_requested"),
    "pf_issued": ("pf_issued", "Ipf", "issued", "prefetch_issued"),
    "pf_useful": ("pf_useful", "U", "useful", "prefetch_useful"),
    "pf_late": ("pf_late", "L", "late", "prefetch_late"),
    "pf_filled": ("pf_filled", "filled", "prefetch_filled"),
    "pf_useless": ("pf_useless", "unused_evicted", "prefetch_useless"),
    "pf_dropped": ("pf_dropped", "dropped", "prefetch_dropped"),
    "pq_merged_duplicate_proxy": ("pq_merged_duplicate_proxy", "Q", "pq_merges", "pq_merged"),
    "occupancy_lines": ("occupancy_lines", "valid_lines", "occupancy"),
    "l2_capacity_lines": ("l2_capacity_lines", "capacity_lines"),
    "l2_sets": ("l2_sets", "sets"),
    "l2_ways": ("l2_ways", "ways"),
    "eligible_callbacks": ("eligible_callbacks", "eligible_l2_callbacks", "eligible", "callbacks"),
    "admitted_decisions": ("admitted_decisions", "nn_admitted", "admitted"),
    "completed_decisions": ("completed_decisions", "nn_completed", "completed"),
    "supervised_decisions": ("supervised_decisions", "available_supervised_decisions", "observed_labels", "labels_available"),
    "positive_actions": ("positive_actions", "available_positive_actions", "teacher_action_atoms"),
    "completed_updates": ("completed_updates", "updates", "updates_completed"),
    "training_sample_exposures": ("training_sample_exposures", "sample_exposures", "trained_samples"),
    "published_versions": ("published_versions", "published_version", "model_version"),
    "input_drops": ("input_drops", "inference_dropped"),
    "output_drops": ("output_drops", "output_dropped_addresses"),
    "training_drops": ("training_drops", "training_dropped"),
    "decoder_limit_drops": ("decoder_limit_drops", "decoder_dropped_addresses"),
}


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ratio(numerator, denominator):
    n, d = number(numerator), number(denominator)
    return n / d if n is not None and d not in (None, 0) else None


def metrics(c, baseline=None):
    """Exact legacy formulas, with undefined ratios represented by None/NA."""
    useful, issued, merged = (c.get(k) for k in ("pf_useful", "pf_issued", "pq_merged_duplicate_proxy"))
    late, misses, loads = (c.get(k) for k in ("pf_late", "l2_load_miss", "l2_loads"))
    m0 = baseline.get("l2_load_miss") if baseline else None
    return {
        "ipc": ratio(c.get("instructions"), c.get("cycles")),
        "l2_mpki": ratio(None if misses is None else 1000 * misses, c.get("instructions")),
        "l2_load_miss_rate": ratio(misses, loads),
        "useful_prefetch_coverage": ratio(useful, m0),
        "miss_reduction": ratio(None if m0 is None or misses is None else m0 - misses, m0),
        "raw_accuracy": ratio(useful, issued),
        "selected_accuracy": ratio(useful, None if issued is None or merged is None else issued - merged),
        "legacy_timeliness": ratio(useful, None if useful is None or late is None else useful + late),
        "request_pressure": ratio(c.get("pf_requested"), loads),
    }


def flatten(row):
    merged = dict(row)
    for name in ("counters", "cache", "nn", "training", "progress", "geometry", "occupancy"):
        if isinstance(row.get(name), dict):
            merged.update(row[name])
    return merged


def normalize(row):
    flat = flatten(row)
    normalized = dict(flat)
    for key in set(COUNTERS + PROGRESS + ("instructions", "cycles", "occupancy_lines", "l2_capacity_lines", "l2_sets", "l2_ways")):
        options = ALIASES.get(key, (key,))
        value = next((flat[k] for k in options if k in flat), None)
        if value is not None:
            normalized[key] = number(value)
    return normalized


def read_snapshots(path):
    rows, warnings = [], []
    if not path:
        return rows, warnings
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"invalid JSON line {index} in {path.name}")
            continue
        if raw.get("type", raw.get("event", "snapshot")) in ("snapshot", "sample", "final", "summary", "milestone"):
            row = normalize(raw)
            if row.get("instructions") is not None and row.get("cycles") is not None:
                rows.append(row)
    by_instruction = {r["instructions"]: r for r in rows}
    rows = sorted(by_instruction.values(), key=lambda r: r["instructions"])
    if any(b["cycles"] < a["cycles"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"nonmonotonic snapshot cycles in {path}")
    return rows, warnings


def delta(end, start):
    keys = set(COUNTERS + PROGRESS + ("instructions", "cycles"))
    result = {}
    for key in keys:
        a, b = number(end.get(key)), number(start.get(key))
        result[key] = a - b if a is not None and b is not None else None
        if result[key] is not None and result[key] < 0:
            raise ValueError(f"cumulative counter {key} decreased")
    return result


def windows(rows, width=1_000_000):
    """Use exact retained milestones; never invent events by interpolation.

    Crossing samples can be at most one retirement group after a nominal
    boundary. Actual instruction counts are retained in each interval, and
    baseline cycles are normalized by the observed instruction count.
    """
    if not rows:
        return []
    zero = {k: 0 for k in COUNTERS + PROGRESS + ("instructions", "cycles")}
    origin = next((r for r in rows if r["instructions"] == 0), zero)
    boundaries = [origin]
    target = width
    for row in rows:
        if row["instructions"] >= target:
            if row["instructions"] - target > 1000:
                target = (int(row["instructions"]) // width + 1) * width
                continue
            boundaries.append(row)
            target += width
    result = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        d = delta(end, start)
        d.update({"window": index, "start_instructions": start["instructions"], "end_instructions": end["instructions"], "cumulative_cycles": end["cycles"], "snapshot": end})
        result.append(d)
    return result


def first_sustained(rows, predicate, count=3):
    for index in range(len(rows) - count + 1):
        group = rows[index:index + count]
        if all(predicate(r) for r in group) and all(b["window"] == a["window"] + 1 for a, b in zip(group, group[1:])):
            return group[0], group[-1]
    return None


def occupancy_milestones(snapshots, explicit=None):
    out = []
    explicit = explicit or {}
    for fraction in (0.50, 0.90, 0.95, 0.99, 1.00):
        key = str(int(fraction * 100))
        provided = explicit.get(key, explicit.get(key + "%"))
        source = "exact fill event" if provided else "sampled attainment (upper bound)"
        found = normalize(provided) if isinstance(provided, dict) else next((r for r in snapshots if ratio(r.get("occupancy_lines"), r.get("l2_capacity_lines")) is not None and ratio(r["occupancy_lines"], r["l2_capacity_lines"]) >= fraction), None)
        out.append({"occupancy_percent": int(fraction * 100), "status": "REACHED" if found else ("NOT REACHED" if snapshots or explicit else "UNAVAILABLE"), "instructions": found.get("instructions") if found else None, "cycles": found.get("cycles") if found else None, "measurement": source if found else None})
    return out


def load_oracle():
    spec = importlib.util.spec_from_file_location("original_602_counter_parser", ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else ({} if default is None else default)


def find_file(directory, names):
    return next((directory / name for name in names if (directory / name).is_file()), None)


def arm(config):
    value = str(config.get("arm", config.get("method", config.get("mode", "unknown"))))
    return {"none": "no_pref", "no-prefetch": "no_pref", "no_prefetch": "no_pref", "conventional_stride": "stride", "frozen_i1m": "frozen", "frozen_i20m": "frozen", "online_random": "scratch", "random": "scratch", "random_start": "scratch", "online_scratch": "scratch", "warm_start": "warm", "online_warm": "warm"}.get(value, value)


def identity(config, run_id):
    checkpoint = config.get("checkpoint_tag", config.get("budget", config.get("checkpoint", "")))
    if config.get("arm") in ("frozen_i1m", "frozen_i20m", "online_warm"):
        checkpoint = "i20m" if config["arm"] == "frozen_i20m" else "i1m"
    protocol = config.get("protocol", "UNSPECIFIED")
    skip = config.get("skip_records", config.get("skip_instructions", config.get("skip", 0)))
    warmup = config.get("warmup_instructions", 0)
    length = config.get("simulation_instructions")
    phase = config.get("phase", config.get("stage"))
    if phase is None:
        if protocol == "software_fixture":
            phase = "software_fixture"
        elif 20_000_000 <= skip < 25_000_000 and length is not None and skip + length <= 25_000_000 and not warmup:
            phase = "development"
        elif length == 25_000_000 and ((protocol == "cold" and skip == 25_000_000 and not warmup) or (protocol == "historical" and warmup == 25_000_000)):
            phase = "final"
        else:
            phase = "unspecified"
    return {"run_id": run_id, "arm": arm(config), "protocol": protocol, "phase": phase, "hidden_size": config.get("hidden_size", config.get("hidden", config.get("h"))), "seed": config.get("seed"), "checkpoint": checkpoint, "state_capacity": config.get("state_capacity", config.get("pc_capacity", 0)), "service_case": config.get("service_case", config.get("macs", config.get("service", 0))), "slice_origin_instructions": config.get("slice_origin_instructions", skip + warmup), "simulation_instructions": length}


def match_protocol(a, b):
    return all(str(a.get(k)) == str(b.get(k)) for k in ("protocol", "phase", "slice_origin_instructions", "simulation_instructions"))


def choose_reference(run, runs, which):
    candidates = [r for r in runs if r["complete"] and match_protocol(run["identity"], r["identity"])]
    if which in ("no_pref", "stride"):
        candidates = [r for r in candidates if r["identity"]["arm"] == which]
    else:
        candidates = [r for r in candidates if r["identity"]["arm"] == "frozen" and str(r["identity"]["hidden_size"]) == str(run["identity"]["hidden_size"]) and "i1m" in str(r["identity"]["checkpoint"]) and str(r["identity"]["state_capacity"]) == str(run["identity"]["state_capacity"]) and str(r["identity"]["service_case"]) == str(run["identity"]["service_case"])]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous {which} reference for {run['identity']['run_id']}")
    return candidates[0] if candidates else None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if v is None else json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def inspect_run(directory, oracle):
    cfg_file = find_file(directory, ("run.json", "config.json", "run_config.json"))
    config = read_json(cfg_file) if cfg_file else {}
    ident = identity(config, config.get("run_id", directory.name))
    log = find_file(directory, ("run.log", "champsim.log", "simulator.log"))
    complete = bool(log and re.search(r"Finished CPU\s+0\s+instructions:", log.read_text(errors="replace")))
    warnings, counters = [], {}
    if complete:
        try:
            counters = oracle.parse_log(log)
        except (RuntimeError, ValueError) as exc:
            warnings.append(str(exc)); complete = False
    timeline = find_file(directory, ("snapshots.jsonl", "timeline.jsonl", "observer.jsonl", "stats.jsonl"))
    snapshots, notes = read_snapshots(timeline)
    warnings += notes
    origins = {r.get("trace_origin_records") for r in snapshots if r.get("trace_origin_records") is not None}
    if len(origins) > 1:
        raise ValueError(f"changing trace-slice origin in {directory}")
    ident["observed_trace_origin_records"] = next(iter(origins), None)
    if complete and snapshots:
        for key in COUNTERS + ("instructions", "cycles"):
            if key in snapshots[-1] and key in counters and snapshots[-1][key] != counters[key]:
                warnings.append(f"final snapshot/{key}={snapshots[-1][key]} differs from simulator final={counters[key]}")
                complete = False
    elif complete:
        warnings.append("finished simulator log has no measurement snapshots")
    observer = read_json(directory / "observer_stats.json")
    engine = {}
    if log:
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("online602_final "):
                engine = normalize(json.loads(line[len("online602_final "):]))
    cohorts = []
    if timeline:
        thresholds = observer.setdefault("occupancy_milestones", {})
        for line in timeline.read_text().splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if raw.get("event") == "occupancy_threshold":
                thresholds[str(int(raw["percent"]))] = raw
            elif raw.get("event") == "cohort":
                total = sum(raw.get(k, 0) for k in ("timely", "late", "unused", "redundant_cache", "redundant_inflight", "pending", "resident_unused"))
                if total != raw.get("enqueued"):
                    raise ValueError(f"nonconserved issue cohort in {directory}")
                cohorts.append(raw)
    storage = read_json(directory / "storage.json")
    worker = read_json(directory / "worker_stats.json")
    host = {k: v for k, v in config.items() if k in ("host_seconds", "simulator_wall_seconds", "simulator_peak_rss_kib")}
    host.update({k: v for k, v in worker.items() if k.startswith("worker_host_") or k in ("worker_peak_rss_bytes", "worker_rss_scope")})
    host.update(read_json(directory / "host.json"))
    return {"identity": ident, "config": config, "directory": str(directory), "complete": complete, "counters": counters, "snapshots": snapshots, "windows": windows(snapshots), "observer": observer, "engine": engine, "storage": storage, "worker": worker, "host": host, "cohorts": cohorts, "warnings": warnings}


def storage_categories(run):
    """FP32 logical bounds/observed live allocations, separate from allocator RSS.

    Engine example capacity already covers one worker batch plus pending data.
    Padded tensors and unique autograd-saved activation storage are additional.
    """
    ident, worker = run["identity"], run["worker"]
    engine = {**(run["snapshots"][-1] if run["snapshots"] else {}), **run["engine"]}
    online = ident["arm"] in ("scratch", "warm")
    categories = []
    def add(name, configured, observed, note=""):
        categories.append({"category": name, "configured_max_bytes": configured, "observed_peak_bytes": observed, "configured_max_kib": ratio(configured, 1024), "observed_peak_kib": ratio(observed, 1024), "scope": "required logical predictor/trainer storage", "note": note})
    weight = engine.get("weight_bytes")
    if weight is None:
        return []
    add("active FP32 weights", weight, weight)
    cap = engine.get("state_capacity", ident.get("state_capacity"))
    add("PC tags, valid/replacement metadata and h/c", engine.get("state_bytes_configured") if cap else (None if weight else 0), engine.get("state_bytes_peak"), "UNBOUNDED configured maximum" if not cap and weight else "exact-PC entries; logical packed metadata plus FP32 h/c")
    for name, conf, obs in (("inference scratch", "inference_scratch_bytes", "inference_scratch_bytes"), ("active event metadata", "active_event_metadata_bytes", "active_event_metadata_bytes"), ("decoded delta/address result payload", "decoded_result_bytes_configured", "decoded_result_bytes_peak"), ("input queue", "input_queue_bytes_configured", "input_queue_bytes_peak"), ("output queue", "output_queue_bytes_configured", "output_queue_bytes_peak"), ("shadow Stride / conventional Stride state", "teacher_bytes_configured", "teacher_bytes_configured")):
        note = "fixed arithmetic workspace reservation, not a sampled allocator peak" if name == "inference scratch" else ("fixed active packet reservation" if name == "active event metadata" else "")
        add(name, engine.get(conf), engine.get(obs), note)
    add("teacher label availability queue", engine.get("teacher_output_bytes_configured", engine.get("teacher_queue_bytes_configured")), engine.get("teacher_output_bytes_peak", engine.get("teacher_queue_bytes_peak")), "48-byte reserved hardware packets per occupied entry; current co-simulation availability Label uses 24 bytes, excluding host deque overhead")
    if online:
        example_bytes = ratio(engine.get("training_examples_bytes_configured"), 128)
        examples_peak = ((number(engine.get("peak_training_queue")) or 0) * example_bytes + (number(worker.get("peak_logical_example_payload_bytes")) or 0)) if example_bytes is not None else None
        add("bounded training examples and saved start states", engine.get("training_examples_bytes_configured"), examples_peak, "64 pending plus one batch in worker; summed component peaks are conservative, not guaranteed simultaneous")
        add("padded training input and target tensors", worker.get("configured_max_padded_input_target_tensor_bytes"), worker.get("peak_padded_input_target_tensor_bytes"))
        add("TBPTT autograd-saved activations", worker.get("configured_saved_activation_storage_byte_budget"), worker.get("peak_unique_saved_activation_storage_bytes_excluding_parameters_and_padded_inputs"), "unique underlying saved storages; excludes parameter and padded input/target storage; aliases counted once")
        add("gradient tensors", worker.get("configured_max_gradient_tensor_bytes"), worker.get("peak_gradient_tensor_bytes"))
        moments = worker.get("configured_max_adam_moment_tensor_bytes")
        steps = worker.get("configured_max_adam_step_tensor_bytes")
        add("Adam moments and step tensors", moments + steps if moments is not None and steps is not None else None, worker.get("peak_optimizer_tensor_bytes_including_steps"))
        add("training FP32 weight copy", worker.get("model_parameter_bytes"), worker.get("model_parameter_bytes"))
        add("publication buffer", engine.get("publication_bytes_configured"), engine.get("publication_bytes_configured") if engine.get("published_versions", 0) else 0)
        retained = engine.get("retained_weight_bank_bytes_configured")
        observed_retained = engine.get("retained_weight_bank_bytes_peak")
        if retained is not None and observed_retained is None and not engine.get("publications_during_inference", 0):
            observed_retained = 0
        add("retained old inference weight bank", retained, observed_retained, "separate bank protects an in-flight inference version during publication; configured only for finite online service")
    return categories


def analyze(run_dir, out_dir, historical_no_pref=None):
    oracle = load_oracle()
    historical_baseline = oracle.parse_log(historical_no_pref) if historical_no_pref else None
    if historical_baseline and (historical_baseline["instructions"] != 25_000_003 or historical_baseline["pf_requested"] != 0):
        raise ValueError("historical no-prefetch reference must have original 25,000,003 measured instructions and no requests")
    configs = list(run_dir.glob("*/run.json")) + list(run_dir.glob("runs/*/run.json"))
    directories = sorted({p.parent for p in configs})
    if not directories:
        directories = sorted({p.parent for p in run_dir.glob("*/run.log")})
    if (run_dir / "run.log").is_file():
        directories = [run_dir]
    runs = [inspect_run(d, oracle) for d in directories]
    summaries, all_windows, early_windows, learning, occupancy, all_samples, storage_rows, cohorts, resources = [], [], [], [], [], [], [], [], []
    for run in runs:
        ident, counters = run["identity"], run["counters"]
        no_pref = choose_reference(run, runs, "no_pref")
        stride = choose_reference(run, runs, "stride")
        frozen = choose_reference(run, runs, "frozen")
        baseline = no_pref["counters"] if no_pref else None
        baseline_source = "new matched same-protocol no-prefetch run" if baseline else None
        if not baseline and historical_baseline and ident["protocol"] == "historical" and ident["phase"] == "final" and counters.get("instructions") == historical_baseline["instructions"]:
            baseline = historical_baseline
            baseline_source = "historical same-protocol no-prefetch end counters only"
        summary = {**ident, "status": "COMPLETE" if run["complete"] else "INCOMPLETE", "end_no_pref_reference": baseline_source, **{k: counters.get(k) for k in ("instructions", "cycles") + COUNTERS}, **metrics(counters, baseline)}
        if stride and counters.get("instructions"):
            matched_cycles = stride["counters"]["cycles"] * counters["instructions"] / stride["counters"]["instructions"]
            summary["cycles_saved_vs_stride"] = matched_cycles - counters["cycles"]
        last = {**(run["snapshots"][-1] if run["snapshots"] else {}), **run["engine"]}
        summary.update({k: last.get(k, run["worker"].get(k)) for k in PROGRESS})
        summary["warnings"] = "; ".join(run["warnings"])
        summary.update(run["host"])
        summaries.append(summary)
        for sample in run["snapshots"]:
            all_samples.append({**ident, **{k: v for k, v in sample.items() if not isinstance(v, (dict, list))}})
        for width, destination in ((1_000_000, all_windows), (100_000, early_windows)):
            reference_windows = {name: windows(ref["snapshots"], width) if ref else [] for name, ref in (("no_pref", no_pref), ("stride", stride), ("frozen", frozen))}
            for w in windows(run["snapshots"], width):
                index = w["window"] - 1
                refs = {name: (series[index] if len(series) > index else None) for name, series in reference_windows.items()}
                row = {**ident, **{k: v for k, v in w.items() if k != "snapshot"}, **metrics(w, refs["no_pref"])}
                s, f = refs["stride"], refs["frozen"]
                row["window_cycles_saved_vs_stride"] = s["cycles"] * w["instructions"] / s["instructions"] - w["cycles"] if s else None
                row["cumulative_cycles_saved_vs_stride"] = s["cumulative_cycles"] * w["end_instructions"] / s["end_instructions"] - w["cumulative_cycles"] if s else None
                row["ipc_fraction_of_frozen_i1m"] = ratio(row["ipc"], ratio(f["instructions"], f["cycles"])) if f else None
                row.update({"total_" + k: w["snapshot"].get(k) for k in PROGRESS})
                destination.append(row)
        own_windows = [w for w in all_windows if w["run_id"] == ident["run_id"]]
        for name, predicate in (("positive_cumulative_cycle_saving_vs_stride", lambda r: (number(r.get("cumulative_cycles_saved_vs_stride")) or 0) > 0), ("positive_window_cycle_saving_vs_stride", lambda r: (number(r.get("window_cycles_saved_vs_stride")) or 0) > 0), ("99_percent_frozen_i1m_window_ipc", lambda r: (number(r.get("ipc_fraction_of_frozen_i1m")) or 0) >= .99)):
            found = first_sustained(own_windows, predicate)
            first, confirmed = found if found else ({}, {})
            milestone_fields = ("ipc", "useful_prefetch_coverage", "selected_accuracy", "legacy_timeliness", "request_pressure", "cumulative_cycles_saved_vs_stride", "total_eligible_callbacks", "total_supervised_decisions", "total_positive_actions", "total_completed_updates", "total_training_sample_exposures", "total_published_versions")
            learning.append({**ident, "milestone": name, "status": "REACHED" if found else "NOT REACHED", "first_window_end": first.get("end_instructions"), "confirmed_at_instructions": confirmed.get("end_instructions"), **{k: first.get(k) for k in milestone_fields}, **{"confirmation_" + k: confirmed.get(k) for k in milestone_fields}, "cumulative_cycles_saved_at_confirmation": confirmed.get("cumulative_cycles_saved_vs_stride")})
        for row in occupancy_milestones(run["snapshots"], run["observer"].get("occupancy_milestones")):
            occupancy.append({**ident, **row})
        for row in run["cohorts"]:
            cohorts.append({**ident, **row, "censored": row.get("pending", 0) + row.get("resident_unused", 0)})
        entries = run["storage"].get("categories", [])
        if not entries:
            entries = storage_categories(run)
        if isinstance(entries, dict):
            entries = [{"category": k, **v} for k, v in entries.items()]
        for entry in entries:
            storage_rows.append({**ident, **entry})
        resource_fields = ("training_admitted", "training_dropped", "publications_during_inference", "peak_state_entries", "peak_teacher_queue", "macs_per_cycle", "weight_bandwidth_bytes_per_cycle", "inference_macs", "inference_nonlinears", "training_forward_macs", "training_modeled_macs", "training_modeled_nonlinears", "inference_service_cycles", "training_service_cycles", "publication_cycles", "teacher_calls", "teacher_tag_comparison_bound", "max_tbptt_span", "peak_padded_positions", "peak_inflight_updates", "peak_input_queue", "peak_output_queue", "peak_training_queue", "inference_scratch_traffic_bytes", "training_state_traffic_bytes", "input_queue", "output_queue", "training_queue", "inference_inflight", "update_inflight", "callback_loads", "callback_load_misses", "capacity_bytes", "capacity_lines", "line_bytes", "l2_sets", "l2_ways", "fills_load", "fills_rfo", "fills_prefetch", "fills_writeback", "replacements", "occupancy_peak", "line_lifetimes", "line_lifetime_cycles", "useful_line_lifetimes", "useful_line_lifetime_cycles", "prefetch_useful_lifetimes", "prefetch_fill_to_first_demand_cycles")
        worker_fields = ("peak_effective_same_lifetime_tbptt_span", "peak_padded_projection_positions", "total_padded_projection_positions", "trainer_available_decisions", "trainer_available_positive_decisions", "training_action_exposures", "last_loss", "last_gradient_l2", "last_gate_class_weights", "class_balance_history", "optimizer", "learning_rate", "library_threads")
        resource = {**ident, **{k: last.get(k) for k in resource_fields}, **{k: last.get(k) for k in PROGRESS}, **{k: run["worker"].get(k) for k in worker_fields}, **run["host"]}
        resource.update({"nn_started": last.get("nn_started"), "updates_started": last.get("updates_started"), "mean_inference_macs": ratio(last.get("inference_macs"), last.get("nn_started")), "mean_inference_nonlinears": ratio(last.get("inference_nonlinears"), last.get("nn_started")), "mean_update_macs": ratio(last.get("training_modeled_macs"), last.get("updates_started")), "mean_update_nonlinears": ratio(last.get("training_modeled_nonlinears"), last.get("updates_started"))})
        resources.append(resource)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("windows_1m", all_windows), ("windows_100k", early_windows), ("learning_milestones", learning), ("occupancy_milestones", occupancy), ("snapshots", all_samples), ("storage", storage_rows), ("issue_cohorts", cohorts), ("resources", resources)):
        write_csv(out_dir / (name + ".csv"), rows)
    payload = {"schema_version": 1, "raw_run_dir": str(run_dir), "completed_runs": sum(r["complete"] for r in runs), "total_discovered_runs": len(runs), "summary": summaries, "learning_milestones": learning, "occupancy_milestones": occupancy, "storage": storage_rows, "window_policy": "Three consecutive 1M-retired-instruction endpoints with positive cumulative saving establish recovered startup cost. A separate positive-window streak is descriptive only. Frozen attainment uses window IPC >=99% of same-H frozen i1m for three windows. Report first qualifying endpoint and third-window confirmation. Crossing samples retain actual retirement counts; reference cycles normalized for a differing final retirement group. Interval quality uses legacy counter deltas, not issue-cohort success.", "runs": [{k: v for k, v in r.items() if k not in ("snapshots", "windows", "cohorts")} for r in runs]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Analyzed {payload['completed_runs']}/{len(runs)} finished runs into {out_dir}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--historical-no-pref", type=Path, help="Optional original same-protocol no-prefetch end log; never used for cold or dynamic windows")
    args = parser.parse_args()
    analyze(args.run_dir, args.out_dir, args.historical_no_pref)


if __name__ == "__main__":
    main()
