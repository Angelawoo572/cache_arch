#!/usr/bin/env python3
"""Independent stdlib checks for reader metrics/pairs/Pareto; never executes project code."""
import argparse
import csv
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "formal_study/results/602_width_templates"


def read_csv(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames or [], rows


def number(value):
    if value is None or str(value).strip().lower() in ("", "na", "n/a", "null", "none"):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite CSV value")
    return value


def boolean(value):
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError("not a boolean: " + str(value))


def relation(nn_ipc, native_ipc, nn_cost, native_cost):
    if nn_ipc == native_ipc and nn_cost == native_cost:
        return "exact_tie"
    if nn_ipc >= native_ipc and nn_cost <= native_cost:
        return "nn_dominates"
    if native_ipc >= nn_ipc and native_cost <= nn_cost:
        return "native_dominates"
    return "tradeoff"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "per_run.csv")
    parser.add_argument("--completion-source", type=Path, default=BASE / "per_run_h16_completion.csv")
    parser.add_argument("--metrics", type=Path, default=BASE / "reader_metrics.csv")
    parser.add_argument("--pairs", type=Path, default=BASE / "reader_pairs.csv")
    parser.add_argument("--pareto", type=Path, default=BASE / "reader_pareto.csv")
    parser.add_argument("--require-pareto", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in (args.source, args.completion_source, args.metrics, args.pairs, args.pareto, args.output):
        if path is not None and path.resolve() != ROOT and ROOT not in path.resolve().parents:
            raise ValueError("all reader files must stay inside cache")
    errors = []
    checks = 0

    def check(condition, context, field, actual=None, expected=None):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(dict(context=context, field=field, actual=actual, expected=expected))

    def check_value(row, name, expected, context):
        check(name in row, context, "required field " + name)
        if name not in row:
            return
        actual = row[name]
        try:
            if expected is None:
                good = number(actual) is None
            elif isinstance(expected, bool):
                good = boolean(actual) is expected
            elif isinstance(expected, (int, float, Fraction)):
                value = number(actual)
                if value is None:
                    good = False
                elif isinstance(expected, int):
                    good = value == expected
                else:
                    good = math.isclose(value, float(expected), rel_tol=2e-12, abs_tol=2e-12)
            else:
                good = actual == expected
        except (ValueError, TypeError):
            good = False
        check(good, context, name, actual, expected if not isinstance(expected, Fraction) else float(expected))

    source_headers, original_rows = read_csv(args.source)
    check(len(original_rows) == 88, "source", "88 original rows", len(original_rows), 88)
    completion_headers, completion_rows = read_csv(args.completion_source)
    check(len(completion_rows) == 30, "completion source", "30 completed validation/test rows", len(completion_rows), 30)
    source_rows = original_rows + completion_rows
    check(len({r["result_id"] for r in source_rows}) == len(source_rows), "source", "unique provenance identities across rounds")
    selected = [r for r in source_rows
                if r["stage"] == "test" and r["mode"] == "full"
                and r["method"] in ("native", "student", "stride_student", "none")
                and r["budget"] in ("0", "16144")]
    check(len(selected) == 50, "source", "eligible full-comparison rows", len(selected), 50)
    expected = {}
    for r in selected:
        rid = r["result_id"] or r["protocol"] + "::" + r["run_id"]
        h = int(r["hidden_size"] or (r["raw__h"] if r["method"] == "stride_student" else "0"))
        arm = "nopf" if r["method"] == "none" else "native" if r["method"] == "native" else "nn"
        ins, cycles = int(r["instructions"]), int(r["cycles"])
        ipc = Fraction(ins, cycles)
        allowed = int(r["raw__request_gate_allowed"])
        requests = Fraction(allowed * 1000000, ins)
        dram = int(r["raw__dram_total_bus_bytes"])
        check(r["status"] == "complete" and r["macs"] == "0" and r["page_policy"] == "cross_page",
              rid, "complete/full MAC0 shared-page eligibility")
        check(r["warmup"] == "5000000" and r["build_id"] == "build_1792211655617683046",
              rid, "same archived actual build and warmup")
        check(int(r["seed"]) == 7 and int(r["roi"]) in (0, 1) and
              int(r["skip"]) == (20000000 if int(r["roi"]) == 0 else 70000000) and
              abs(ins - 25000000) <= 4, rid, "same seed and full two-ROI contract")
        check(dram == int(r["raw__dram_read_bus_bytes"]) + int(r["raw__dram_write_bus_bytes"]),
              rid, "DRAM read/write byte sum")
        check(dram == int(r["raw__line_bytes"]) *
              (int(r["raw__dram_read_bus_completed_transactions"]) +
               int(r["raw__dram_write_bus_completed_transactions"])),
              rid, "actual completed DRAM transactions")
        for level, miss_field, baseline_field in (
                ("l2", "raw__M", "no_pf_L2_LOAD_misses"),
                ("llc", "raw__LLC_LOAD_misses", "no_pf_LLC_LOAD_misses")):
            baseline = int(r[baseline_field])
            expected_reduction = (baseline - int(r[miss_field])) / baseline if baseline else None
            actual_reduction = number(r["net_" + level + "_load_miss_reduction"])
            check(expected_reduction is None and actual_reduction is None or
                  expected_reduction is not None and actual_reduction is not None and
                  math.isclose(actual_reduction, expected_reduction, rel_tol=2e-12, abs_tol=2e-12),
                  rid, "source net " + level + " miss reduction")
        check(int(r["l2_load_misses"]) == int(r["raw__M"]), rid, "L2 LOAD miss source count")
        check(allowed == int(r["raw__request_cache_enqueued"]) +
              int(r["raw__request_cache_merged"]) + int(r["raw__request_cache_rejected"]),
              rid, "allowed cache-result partition")
        for field, value in (("ipc", ipc), ("requests_per_million_instructions", requests)):
            check(math.isclose(float(r[field]), float(value), rel_tol=2e-12),
                  rid, "source formula " + field)
        if h == 16:
            completion = r["protocol"] == "four602_all_h16_20261017_v1"
            check(r["protocol"] == ("four602_all_h16_20261017_v1" if r["policy"] in ("stride", "streamer") else "four602_width_templates_20261017_v1"),
                  rid, "correct independent H16 round")
            check(r["evidence_origin"] == ("new_H16_completion_run" if completion else "new_supplemental_run") and
                  r["baseline_reuse_is_explicit"] == "True" and
                  r["no_pf_protocol"] == "four602_frozen_live_20261017_v2",
                  rid, "explicit H16 reference reuse")
        else:
            check(r["evidence_origin"] == "archived_64_reused" and
                  r["protocol"] == "four602_frozen_live_20261017_v2",
                  rid, "original H8/native/NoPF provenance")
        e = {key: r[key] for key in
             ("result_id", "run_id", "protocol", "evidence_origin", "source_raw", "source_run", "policy")}
        e["result_id"] = rid
        e.update(arm=arm, hidden_size=h, roi=int(r["roi"]), budget=int(r["budget"]), seed=int(r["seed"]),
                 instructions=ins, cycles=cycles, ipc=ipc,
                 ipc_speedup_vs_NoPF=float(r["ipc_speedup_vs_NoPF"]),
                 requests_per_million_instructions=requests,
                 allowed_requests=allowed, dram_bus_bytes=dram,
                 l2_load_misses=int(r["l2_load_misses"]),
                 l2_load_mpki=Fraction(1000 * int(r["l2_load_misses"]), ins),
                 net_l2_load_miss_reduction=float(r["net_l2_load_miss_reduction"]),
                 net_llc_load_miss_reduction=float(r["net_llc_load_miss_reduction"]))
        for field in ("inference_macs", "inference_nonlinears", "completed_updates", "model_weight_bytes", "recurrent_hc_bytes"):
            e[field] = int(r[field]) if r.get(field) not in (None, "") else None
        for level, baseline_name in (("l2", "no_pf_L2_LOAD_misses"), ("llc", "no_pf_LLC_LOAD_misses")):
            n = int(r[level + "_target_enqueued"])
            timely = int(r[level + "_target_timely_load"])
            late = int(r[level + "_target_late_load"])
            source_tracking_available = boolean(r[level + "_target_metrics_available"])
            # Reader availability means applicability, not the raw tracking flag:
            # NoPF has no prefetch cohort quality to evaluate.
            available = source_tracking_available and arm != "nopf"
            useful = timely + late
            check(useful == int(r[level + "_target_LOAD_useful"]), rid, "source " + level + " LOAD useful partition")
            baseline = int(r[baseline_name])
            e.update({level + "_enqueued": n, level + "_timely_load": timely,
                      level + "_late_load": late, level + "_useful_load": useful,
                      level + "_available": available,
                      level + "_accuracy": Fraction(useful, n) if n and available else None,
                      level + "_coverage_proxy": Fraction(useful, baseline) if baseline and available else None,
                      level + "_timeliness": Fraction(timely, useful) if useful and available else None})
        check(rid not in expected, "source", "unique result_id", rid)
        expected[rid] = e
    natives = {(e["policy"], e["roi"], e["budget"]): e
               for e in expected.values() if e["arm"] == "native"}
    nopf = {e["roi"]: e for e in expected.values() if e["arm"] == "nopf"}
    for e in expected.values():
        control = nopf[e["roi"]]
        check(math.isclose(e["ipc_speedup_vs_NoPF"], float(e["ipc"] / control["ipc"]), rel_tol=2e-12),
              e["result_id"], "NoPF IPC ratio")
        l2_misses = control["l2_load_misses"]
        check(math.isclose(e["net_l2_load_miss_reduction"],
                           (l2_misses - e["l2_load_misses"]) / l2_misses, rel_tol=2e-12, abs_tol=2e-12),
              e["result_id"], "net L2 miss reduction")
    identity_counts = Counter("nopf" if e["arm"] == "nopf" else "H16" if e["hidden_size"] == 16 else "H8_native_or_NN"
                              for e in expected.values())
    check(identity_counts == {"H8_native_or_NN": 32, "H16": 16, "nopf": 2},
          "source", "core partitions", dict(identity_counts))
    model_counts = Counter((e["policy"], e["hidden_size"])
                           for e in expected.values() if e["arm"] == "nn")
    check(model_counts == {(policy, h): 4 for policy in ("stride", "streamer", "ampm", "spp") for h in (8, 16)},
          "source", "all eight actual models have two ROIs and two budgets", dict((str(k), v) for k, v in model_counts.items()))

    metrics_headers, metrics_rows = read_csv(args.metrics)
    check(len(metrics_rows) == 50, "reader_metrics", "row count", len(metrics_rows), 50)
    seen = set()
    for row in metrics_rows:
        rid = row.get("result_id", "")
        check(rid in expected and rid not in seen, "reader_metrics", "unique eligible result_id", rid)
        seen.add(rid)
        if rid in expected:
            for name, value in expected[rid].items():
                check_value(row, name, value, "reader_metrics:" + rid)
    check(seen == set(expected), "reader_metrics", "exact 50 identities")

    pairs_expected = {}
    for e in expected.values():
        if e["arm"] != "nn":
            continue
        native = natives[(e["policy"], e["roi"], e["budget"])]
        key = (e["policy"], e["hidden_size"], e["roi"], e["budget"])
        pair = {k: e[k] for k in ("policy", "hidden_size", "roi", "budget")}
        pair.update(native_result_id=native["result_id"], nn_result_id=e["result_id"],
                    native_ipc=native["ipc"], nn_ipc=e["ipc"],
                    ipc_gain_pct=100 * (e["ipc"] / native["ipc"] - 1),
                    native_requests_per_million=native["requests_per_million_instructions"],
                    nn_requests_per_million=e["requests_per_million_instructions"],
                    requests_change_pct=100 * (e["requests_per_million_instructions"] /
                                               native["requests_per_million_instructions"] - 1),
                    native_dram_bus_bytes=native["dram_bus_bytes"], nn_dram_bus_bytes=e["dram_bus_bytes"],
                    dram_change_pct=100 * (Fraction(e["dram_bus_bytes"], native["dram_bus_bytes"]) - 1),
                    paired_requests_relation=relation(e["ipc"], native["ipc"],
                                                     e["requests_per_million_instructions"],
                                                     native["requests_per_million_instructions"]),
                    paired_dram_relation=relation(e["ipc"], native["ipc"],
                                                 e["dram_bus_bytes"], native["dram_bus_bytes"]))
        for prefix, values in (("native_", native), ("nn_", e)):
            for level in ("l2", "llc"):
                for field in ("enqueued", "accuracy", "coverage_proxy", "timeliness"):
                    pair[prefix + level + "_" + field] = values[level + "_" + field]
        pairs_expected[key] = pair
    check(len(pairs_expected) == 32, "source", "32 own-native pairs")
    pair_headers, pair_rows = read_csv(args.pairs)
    check(len(pair_rows) == 32, "reader_pairs", "row count", len(pair_rows), 32)
    pair_seen = set()
    for row in pair_rows:
        try:
            key = (row["policy"], int(row["hidden_size"]), int(row["roi"]), int(row["budget"]))
        except (KeyError, ValueError):
            check(False, "reader_pairs", "pair identity columns");continue
        check(key in pairs_expected and key not in pair_seen, "reader_pairs", "unique eligible pair", key)
        pair_seen.add(key)
        if key in pairs_expected:
            for name, value in pairs_expected[key].items():
                check_value(row, name, value, "reader_pairs:" + str(key))
    check(pair_seen == set(pairs_expected), "reader_pairs", "exact 16 H8 + 16 H16 pairs")

    pareto_report = dict(status="not_available", expected_rows=176)
    if args.pareto.exists():
        pareto_headers, pareto_rows = read_csv(args.pareto)
        pareto_expected = {}
        for candidate_set in ("H8_only", "H8_H16"):
            for roi in (0, 1):
                for budget in (0, 16144):
                    group = [e for e in expected.values()
                             if e["roi"] == roi and (e["arm"] == "nopf" or e["budget"] == budget)
                             and (candidate_set != "H8_only" or e["hidden_size"] != 16)]
                    check(len(group) == (9 if candidate_set == "H8_only" else 13),
                          candidate_set, "per-ROI/cap candidate count")
                    for cost_axis in ("requests_per_million_instructions", "dram_bus_bytes"):
                        for e in group:
                            dominated = any(a[cost_axis] <= e[cost_axis] and a["ipc"] >= e["ipc"] and
                                            (a[cost_axis] < e[cost_axis] or a["ipc"] > e["ipc"])
                                            for a in group)
                            key = (candidate_set, roi, budget, cost_axis, e["result_id"])
                            pareto_expected[key] = dict(x_cost=e[cost_axis], ipc=e["ipc"],
                                                       is_pareto=not dominated, actual_run_budget=e["budget"])
        check(len(pareto_rows) == 176, "reader_pareto", "row count", len(pareto_rows), 176)
        seen_pareto = set()
        for row in pareto_rows:
            try:
                cap = int(row.get("budget", row.get("cap", "")))
                key = (row["candidate_set"], int(row["roi"]), cap, row["cost_axis"], row["result_id"])
            except (KeyError, ValueError):
                check(False, "reader_pareto", "identity columns");continue
            check(key in pareto_expected and key not in seen_pareto,
                  "reader_pareto", "unique eligible candidate", key)
            seen_pareto.add(key)
            if key in pareto_expected:
                for name, value in pareto_expected[key].items():
                    check_value(row, name, value, "reader_pareto:" + str(key))
        check(seen_pareto == set(pareto_expected), "reader_pareto", "exact candidate sets")
        pareto_report = dict(status="checked", rows=len(pareto_rows),
                             exact_nondominated=sum(v["is_pareto"] for v in pareto_expected.values()))
    elif args.require_pareto:
        check(False, "reader_pareto", "required output exists")

    null_counts = Counter()
    for e in expected.values():
        for level in ("l2", "llc"):
            for name in ("accuracy", "coverage_proxy", "timeliness"):
                if e[level + "_" + name] is None:
                    null_counts[level + "_" + name] += 1
    capped = [p for p in pairs_expected.values() if p["budget"] == 16144]
    capped_summary = {}
    for scope, values in (("H8", [p for p in capped if p["hidden_size"] == 8]),
                          ("H16", [p for p in capped if p["hidden_size"] == 16]),
                          ("H8_plus_H16", capped)):
        capped_summary[scope] = dict(
            pairs=len(values),
            IPC_wins=sum(p["ipc_gain_pct"] > 0 for p in values),
            IPC_ties=sum(p["ipc_gain_pct"] == 0 for p in values),
            IPC_losses=sum(p["ipc_gain_pct"] < 0 for p in values),
            DRAM_increases=sum(p["nn_dram_bus_bytes"] > p["native_dram_bus_bytes"] for p in values),
            DRAM_ties=sum(p["nn_dram_bus_bytes"] == p["native_dram_bus_bytes"] for p in values),
            DRAM_decreases=sum(p["nn_dram_bus_bytes"] < p["native_dram_bus_bytes"] for p in values))
    all_pairs = list(pairs_expected.values())
    def comparison_counts(values):
        return dict(pairs=len(values),
                    IPC_wins=sum(p["ipc_gain_pct"] > 0 for p in values),
                    IPC_ties=sum(p["ipc_gain_pct"] == 0 for p in values),
                    IPC_losses=sum(p["ipc_gain_pct"] < 0 for p in values),
                    DRAM_increases=sum(p["nn_dram_bus_bytes"] > p["native_dram_bus_bytes"] for p in values),
                    DRAM_ties=sum(p["nn_dram_bus_bytes"] == p["native_dram_bus_bytes"] for p in values),
                    DRAM_decreases=sum(p["nn_dram_bus_bytes"] < p["native_dram_bus_bytes"] for p in values))
    headline = dict(
        H8_capped=comparison_counts([p for p in all_pairs if p["hidden_size"] == 8 and p["budget"] == 16144]),
        H8_uncapped=comparison_counts([p for p in all_pairs if p["hidden_size"] == 8 and p["budget"] == 0]),
        H16_capped=comparison_counts([p for p in all_pairs if p["hidden_size"] == 16 and p["budget"] == 16144]),
        H16_uncapped=comparison_counts([p for p in all_pairs if p["hidden_size"] == 16 and p["budget"] == 0]),
        H16_all=comparison_counts([p for p in all_pairs if p["hidden_size"] == 16]))
    h16_lower_dram = [dict(policy=p["policy"], roi=p["roi"], budget=p["budget"],
                          ipc_gain_pct=float(p["ipc_gain_pct"]),
                          delta_dram_bytes=p["nn_dram_bus_bytes"] - p["native_dram_bus_bytes"],
                          dram_change_pct=float(p["dram_change_pct"]))
                      for p in all_pairs if p["hidden_size"] == 16 and p["nn_dram_bus_bytes"] < p["native_dram_bus_bytes"]]
    headline["H16_lower_DRAM_pairs"] = h16_lower_dram
    check(headline["H8_capped"]["pairs"] == 8 and headline["H8_capped"]["IPC_wins"] == 6 and
          headline["H8_capped"]["IPC_ties"] == 0, "root headline", "H8 capped 6/8 IPC wins")
    check(headline["H8_uncapped"]["pairs"] == 8 and headline["H8_uncapped"]["IPC_wins"] == 3 and
          headline["H8_uncapped"]["IPC_ties"] == 0, "root headline", "H8 uncapped 3/8 IPC wins")
    check(headline["H16_all"]["pairs"] == 16 and headline["H16_capped"]["pairs"] == 8 and
          headline["H16_uncapped"]["pairs"] == 8, "root headline", "complete H16 outcomes; no win assumed")
    report = dict(status="PASS" if not errors else "FAIL", checks=checks, errors=errors,
                  source_rows=len(source_rows), original_source_rows=len(original_rows),
                  completion_source_rows=len(completion_rows), reader_metric_rows=len(metrics_rows),
                  reader_pair_rows=len(pair_rows), identity_partitions=dict(identity_counts),
                  expected_NA_counts=dict(null_counts), pareto=pareto_report,
                  capped_own_native_comparisons=capped_summary, root_headline_checks=headline,
                  NoPF_reader_rule="available=False means prefetch quality not applicable; all ratios NA. "
                                   "This deliberately differs from the raw trivially complete empty-cohort tracking flag; "
                                   "NoPF IPC/requests/DRAM remain valid Pareto points.",
                  scope="Read-only CSV/formula/provenance check, strict raw IPC/cost Pareto and paired relations; "
                        "no project code/model imported, no simulation/training. Pareto is conditional on each "
                        "ROI/cap candidate set and MAC0 protocol; it is not a hardware or causal ranking.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
