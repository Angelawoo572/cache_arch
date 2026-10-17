#!/usr/bin/env python3
"""Compact measured tables for the 602 frozen-weight dynamic inference experiment.

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
PROGRESS = ("eligible_callbacks", "admitted_decisions", "completed_decisions", "state_evictions", "input_drops", "output_drops", "decoder_limit_drops", "numerical_failures")
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
    "input_drops": ("input_drops", "inference_dropped"),
    "output_drops": ("output_drops", "output_dropped_addresses"),
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
    for name in ("counters", "cache", "nn", "progress", "geometry", "occupancy"):
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
    return {"none": "no_pref", "no-prefetch": "no_pref", "no_prefetch": "no_pref", "conventional_stride": "stride", "frozen_i1m": "frozen", "frozen_i20m": "frozen"}.get(value, value)


def identity(config, run_id):
    checkpoint = config.get("checkpoint_tag", config.get("budget", config.get("checkpoint", "")))
    if config.get("arm") in ("frozen_i1m", "frozen_i20m"):
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
            # Read original frozen logs without renaming their source records.
            for prefix in ("dynamic602_final ", "online602_final "):
                if line.startswith(prefix):
                    engine = normalize(json.loads(line[len(prefix):]))
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
    host = {k: v for k, v in config.items() if k in ("host_seconds", "simulator_wall_seconds", "simulator_peak_rss_kib")}
    host.update(read_json(directory / "host.json"))
    return {"identity": ident, "directory": str(directory), "complete": complete, "counters": counters, "snapshots": snapshots, "windows": windows(snapshots), "observer": observer, "engine": engine, "host": host, "cohorts": cohorts, "warnings": warnings}


def deployment_fields(row):
    """Keep inference/cache measurements; legacy frozen logs contain unused fields."""
    excluded = ("train", "teacher", "update", "supervised", "positive_actions", "positive_decisions", "publication", "published", "sample_exposures", "tbptt", "padded", "retained_weight", "model_version", "prediction_weight_version")
    return {k: v for k, v in row.items() if not any(term in k for term in excluded)}


def storage_categories(run):
    """Only required FP32 frozen deployment storage; no training allocations."""
    ident = run["identity"]
    engine = {**(run["snapshots"][-1] if run["snapshots"] else {}), **run["engine"]}
    categories = []
    def add(name, configured, observed, note=""):
        categories.append({"category": name, "configured_max_bytes": configured, "observed_peak_bytes": observed, "configured_max_kib": ratio(configured, 1024), "observed_peak_kib": ratio(observed, 1024), "scope": "required logical frozen predictor storage", "note": note})
    if ident["arm"] == "stride":
        add("conventional Stride state", engine.get("stride_bytes_configured", engine.get("teacher_bytes_configured", 2064)), engine.get("stride_bytes_configured", engine.get("teacher_bytes_configured", 2064)), "baseline tracker state; no neural shadow teacher")
        return categories
    if ident["arm"] == "no_pref":
        add("no predictor", 0, 0)
        return categories
    weight = engine.get("weight_bytes")
    if weight is None:
        return []
    add("active FP32 weights", weight, weight)
    cap = engine.get("state_capacity", ident.get("state_capacity"))
    add("PC tags, metadata and h/c", engine.get("state_bytes_configured") if cap else None, engine.get("state_bytes_peak"), "UNBOUNDED configured maximum" if not cap else "exact-PC entries; logical packed metadata plus FP32 h/c")
    for name, conf, obs in (("inference scratch", "inference_scratch_bytes", "inference_scratch_bytes"), ("active event metadata", "active_event_metadata_bytes", "active_event_metadata_bytes"), ("decoded delta/address payload", "decoded_result_bytes_configured", "decoded_result_bytes_peak"), ("input queue", "input_queue_bytes_configured", "input_queue_bytes_peak"), ("output queue", "output_queue_bytes_configured", "output_queue_bytes_peak")):
        note = "fixed workspace reservation, not a sampled allocator peak" if name in ("inference scratch", "active event metadata") else ""
        add(name, engine.get(conf), engine.get(obs), note)
    return categories


def analyze(run_dir, out_dir, historical_no_pref=None):
    oracle = load_oracle()
    historical_baseline = oracle.parse_log(historical_no_pref) if historical_no_pref else None
    if historical_baseline and (historical_baseline["instructions"] != 25_000_003 or historical_baseline["pf_requested"] != 0):
        raise ValueError("historical no-prefetch reference must have original 25,000,003 measured instructions and no requests")
    configs = list(run_dir.glob("*/run.json")) + list(run_dir.glob("runs/*/run.json"))
    directories = sorted({p.parent for p in configs})
    if (run_dir / "run.log").is_file():
        directories = [run_dir]
    # Eligibility is decided from actual run configuration, never directory names.
    accepted = [d for d in directories if arm(read_json(d / "run.json")) in ("no_pref", "stride", "frozen")]
    runs = [inspect_run(d, oracle) for d in accepted]
    summaries, all_windows, early_windows, occupancy, all_samples, storage_rows, cohorts, resources = [], [], [], [], [], [], [], []
    for run in runs:
        ident, counters = run["identity"], run["counters"]
        no_pref, stride = (choose_reference(run, runs, name) for name in ("no_pref", "stride"))
        baseline = no_pref["counters"] if no_pref else None
        source = "same-protocol no-prefetch run" if baseline else None
        if not baseline and historical_baseline and ident["protocol"] == "historical" and ident["phase"] == "final" and counters.get("instructions") == historical_baseline["instructions"]:
            baseline, source = historical_baseline, "historical same-protocol no-prefetch end counters only"
        summary = {**ident, "source_raw_directory": run["directory"], "status": "COMPLETE" if run["complete"] else "INCOMPLETE", "end_no_pref_reference": source, **{k: counters.get(k) for k in ("instructions", "cycles") + COUNTERS}, **metrics(counters, baseline)}
        if stride and counters.get("instructions"):
            summary["cycles_saved_vs_stride"] = stride["counters"]["cycles"] * counters["instructions"] / stride["counters"]["instructions"] - counters["cycles"]
        last = {**(run["snapshots"][-1] if run["snapshots"] else {}), **run["engine"]}
        summary.update({k: last.get(k) for k in PROGRESS})
        summary.update(warnings="; ".join(run["warnings"]), **run["host"])
        summaries.append(summary)
        for sample in run["snapshots"]:
            all_samples.append({**ident, **deployment_fields({k: v for k, v in sample.items() if not isinstance(v, (dict, list))})})
        for width, destination in ((1_000_000, all_windows), (100_000, early_windows)):
            references = {name: windows(ref["snapshots"], width) if ref else [] for name, ref in (("no_pref", no_pref), ("stride", stride))}
            for w in windows(run["snapshots"], width):
                index = w["window"] - 1
                refs = {name: series[index] if len(series) > index else None for name, series in references.items()}
                row = {**ident, **{k: v for k, v in w.items() if k != "snapshot"}, **metrics(w, refs["no_pref"])}
                s = refs["stride"]
                row["window_cycles_saved_vs_stride"] = s["cycles"] * w["instructions"] / s["instructions"] - w["cycles"] if s else None
                row["cumulative_cycles_saved_vs_stride"] = s["cumulative_cycles"] * w["end_instructions"] / s["end_instructions"] - w["cumulative_cycles"] if s else None
                row.update({"total_" + k: w["snapshot"].get(k) for k in PROGRESS})
                destination.append(row)
        occupancy.extend({**ident, **row} for row in occupancy_milestones(run["snapshots"], run["observer"].get("occupancy_milestones")))
        cohorts.extend({**ident, **row, "censored": row.get("pending", 0) + row.get("resident_unused", 0)} for row in run["cohorts"])
        storage_rows.extend({**ident, **entry} for entry in storage_categories(run))
        resources.append({**ident, **deployment_fields(last), **run["host"], "mean_inference_macs": ratio(last.get("inference_macs"), last.get("nn_started")), "mean_inference_nonlinears": ratio(last.get("inference_nonlinears"), last.get("nn_started"))})
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("summary", summaries), ("windows_1m", all_windows), ("windows_100k", early_windows), ("occupancy_milestones", occupancy), ("snapshots", all_samples), ("storage", storage_rows), ("issue_cohorts", cohorts), ("resources", resources)):
        write_csv(out_dir / (name + ".csv"), values)
    totals = []
    for summary in summaries:
        entries = [r for r in storage_rows if r["run_id"] == summary["run_id"]]
        totals.append({**{k: summary[k] for k in identity({}, "") if k in summary}, **{key: sum(row[key] for row in entries) if entries and all(row[key] is not None for row in entries) else None for key in ("configured_max_bytes", "observed_peak_bytes")}, "note": "sum of component peaks; not a simultaneous allocation measurement"})
    write_csv(out_dir / "storage_totals.csv", totals)
    payload = {"schema_version": 2, "scope": "offline-pretrained frozen-weight dynamic inference only", "raw_run_dir": str(run_dir), "completed_runs": sum(r["complete"] for r in runs), "total_discovered_runs": len(runs), "excluded_non_frozen_directories": len(directories)-len(accepted), "summary": summaries, "occupancy_milestones": occupancy, "storage": storage_rows, "window_policy": "Exact retained retirement milestones; legacy counter deltas and separate issue cohorts. No learning milestones.", "runs": [{"identity": r["identity"], "directory": r["directory"], "complete": r["complete"], "counters": r["counters"], "engine": deployment_fields(r["engine"]), "host": r["host"], "warnings": r["warnings"]} for r in runs]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Analyzed {payload['completed_runs']}/{len(runs)} eligible finished runs into {out_dir}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--historical-no-pref", type=Path)
    args = parser.parse_args()
    analyze(args.run_dir, args.out_dir, args.historical_no_pref)


if __name__ == "__main__":
    main()
