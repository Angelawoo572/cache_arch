#!/usr/bin/env python3
"""Parse v3.9 current-run replay logs and compare them to all normal baselines."""
from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "pythia_stats", str(SCRIPT_DIR / "01_parse_prefetch_behavior_audit.py")
)
pythia_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pythia_stats)

LOADED_RE = re.compile(r"\[list_replayer\] loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+PC-line-occ triggers")
FINAL_RE = re.compile(
    r"\[list_replayer\] emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses\s+"
    r"\((\d+)\s+matched PC-line-occ triggers;\s+(\d+)\s+loaded trigger keys;\s+key=pc_line_occ\)"
)
NO_PREF_NAMES = {"no_pref", "none", "nopref"}


def to_float(value, default=0.0):
    try:
        return default if value is None or value == "" else float(value)
    except Exception:
        return default


def div(a, b):
    return float(a) / float(b) if float(b) else 0.0


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_meta(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def replayer_stats(log):
    out = {
        "list_replayer_instantiated": 0,
        "list_loaded_entries": 0,
        "list_loaded_trigger_keys": 0,
        "replayer_emitted_candidates": 0,
        "replayer_runtime_l2_loads": 0,
        "replayer_matched_trigger_keys": 0,
        "replayer_final_loaded_trigger_keys": 0,
    }
    if not log.is_file():
        return out
    with log.open(errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if "adding L2C_PREFETCHER: list_replayer" in line:
                out["list_replayer_instantiated"] = 1
            match = LOADED_RE.search(line)
            if match:
                out["list_loaded_entries"] = int(match.group(1))
                out["list_loaded_trigger_keys"] = int(match.group(2))
            match = FINAL_RE.search(line)
            if match:
                out["replayer_emitted_candidates"] = int(match.group(1))
                out["replayer_runtime_l2_loads"] = int(match.group(2))
                out["replayer_matched_trigger_keys"] = int(match.group(3))
                out["replayer_final_loaded_trigger_keys"] = int(match.group(4))
    return out


def baseline_index(rows):
    by_trace = defaultdict(list)
    no_pref = {}
    best_normal = {}
    for row in rows:
        trace = (row.get("trace") or "").strip()
        pf = (row.get("prefetcher") or "").strip()
        if not trace or not pf or int(to_float(row.get("run_failed"))) != 0:
            continue
        by_trace[trace].append(row)
        if pf in NO_PREF_NAMES:
            no_pref[trace] = row
        elif trace not in best_normal or to_float(row.get("ipc")) > to_float(best_normal[trace].get("ipc")):
            best_normal[trace] = row
    for trace in by_trace:
        by_trace[trace].sort(key=lambda row: (-to_float(row.get("ipc")), row.get("prefetcher", "")))
    return by_trace, no_pref, best_normal


def plan_rows(path):
    rows = read_rows(path)
    required = {"tag", "trace", "source_rel"}
    seen = set()
    for number, row in enumerate(rows, start=2):
        missing = sorted(key for key in required if not (row.get(key) or "").strip())
        if missing:
            raise ValueError("plan row {} missing {}".format(number, missing))
        tag = row["tag"].strip()
        if tag in seen:
            raise ValueError("duplicate plan tag {}".format(tag))
        seen.add(tag)
    if not rows:
        raise ValueError("empty replay plan")
    return rows


def one_result(plan, args, baseline_by_trace, normal_no_pref, best_normal):
    tag = plan["tag"].strip()
    trace = plan["trace"].strip()
    log = args.log_root / (tag + ".list_replayer.log")
    meta = load_meta(args.replay_input_root / (tag + ".pc_line_occ.csv.meta.json"))
    base = pythia_stats.summarize_one(trace, "v3_9_" + tag, log, nodup=True)
    replay = replayer_stats(log)
    same = pythia_stats.summarize_one(
        trace, "same_binary_no_pref", args.same_binary_log_root / (trace + ".same_binary_no_pref.log"), nodup=True
    )
    same_ipc = to_float(same.get("ipc"))

    row = dict(plan)
    row.update(base)
    row.update(replay)
    row.update({
        "tag": tag,
        "trace": trace,
        "replay_ipc": to_float(base.get("ipc")),
        "same_binary_no_pref_ipc": same_ipc,
        "same_binary_no_pref_run_failed": int(to_float(same.get("run_failed"))),
        "delta_ipc_vs_same_binary_no_pref": to_float(base.get("ipc")) - same_ipc,
        "speedup_vs_same_binary_no_pref": div(to_float(base.get("ipc")), same_ipc),
        "keyed_entries": int(meta.get("entries", 0)),
        "keyed_unique_trigger_keys": int(meta.get("unique_trigger_keys", 0)),
        "keyed_unmatched_rows": int(meta.get("unmatched_rows", -1)),
        "keyed_dropped_invalid_address": int(meta.get("dropped_invalid_address", -1)),
    })
    row["keyed_trigger_coverage"] = div(replay["replayer_matched_trigger_keys"], row["keyed_unique_trigger_keys"])
    row["replay_transport_ok"] = int(
        replay["list_replayer_instantiated"] == 1
        and replay["list_loaded_entries"] == row["keyed_entries"]
        and replay["list_loaded_trigger_keys"] == row["keyed_unique_trigger_keys"]
        and replay["replayer_final_loaded_trigger_keys"] == row["keyed_unique_trigger_keys"]
        and row["keyed_unmatched_rows"] == 0
        and not int(to_float(base.get("run_failed")))
    )

    normal_zero = normal_no_pref.get(trace, {})
    normal_best = best_normal.get(trace, {})
    reference_no_pref_ipc = to_float(normal_zero.get("ipc"))
    row.update({
        "reference_no_pref_ipc": reference_no_pref_ipc,
        "reference_best_normal": normal_best.get("prefetcher", ""),
        "reference_best_normal_ipc": to_float(normal_best.get("ipc")),
        "delta_ipc_vs_reference_best_normal": to_float(base.get("ipc")) - to_float(normal_best.get("ipc")),
        "speedup_vs_reference_best_normal": div(to_float(base.get("ipc")), to_float(normal_best.get("ipc"))),
        "reference_vs_replayer_no_pref_ipc_delta": reference_no_pref_ipc - same_ipc,
        "reference_vs_replayer_no_pref_within_tolerance": int(
            not int(to_float(same.get("run_failed")))
            and abs(reference_no_pref_ipc - same_ipc) <= args.no_pref_ipc_tolerance
        ),
        "reference_baseline_rows": len(baseline_by_trace.get(trace, [])),
    })
    return row


def comparisons(candidates, baseline_by_trace):
    out = []
    for candidate in candidates:
        if int(to_float(candidate.get("run_failed"))) or not int(to_float(candidate.get("replay_transport_ok"))):
            continue
        for normal in baseline_by_trace.get(candidate["trace"], []):
            out.append({
                "trace": candidate["trace"],
                "tag": candidate["tag"],
                "candidate_role": candidate.get("candidate_role", ""),
                "replay_kind": candidate.get("replay_kind", ""),
                "requested_recipe": candidate.get("requested_recipe", ""),
                "final_recipe": candidate.get("final_recipe", ""),
                "candidate_model_family": candidate.get("model_family", ""),
                "candidate_ipc": to_float(candidate.get("replay_ipc")),
                "candidate_delta_vs_same_binary_no_pref": to_float(candidate.get("delta_ipc_vs_same_binary_no_pref")),
                "candidate_speedup_vs_same_binary_no_pref": to_float(candidate.get("speedup_vs_same_binary_no_pref")),
                "candidate_accuracy": to_float(candidate.get("selected_accuracy")),
                "candidate_timeliness": to_float(candidate.get("timeliness")),
                "candidate_pf_issued": int(to_float(candidate.get("pf_issued"))),
                "candidate_pf_useful": int(to_float(candidate.get("pf_useful"))),
                "candidate_pf_useless": int(to_float(candidate.get("pf_useless"))),
                "candidate_pf_late": int(to_float(candidate.get("pf_late"))),
                "normal_prefetcher": normal.get("prefetcher", ""),
                "normal_ipc": to_float(normal.get("ipc")),
                "normal_accuracy": to_float(normal.get("selected_accuracy") or normal.get("accuracy")),
                "normal_timeliness": to_float(normal.get("timeliness")),
                "normal_pf_issued": int(to_float(normal.get("pf_issued"))),
                "normal_pf_useful": int(to_float(normal.get("pf_useful"))),
                "normal_pf_useless": int(to_float(normal.get("pf_useless"))),
                "normal_pf_late": int(to_float(normal.get("pf_late"))),
                "delta_ipc_candidate_minus_normal": to_float(candidate.get("replay_ipc")) - to_float(normal.get("ipc")),
                "speedup_candidate_over_normal": div(to_float(candidate.get("replay_ipc")), to_float(normal.get("ipc"))),
                "candidate_beats_normal": int(to_float(candidate.get("replay_ipc")) > to_float(normal.get("ipc"))),
                "reference_vs_replayer_no_pref_within_tolerance": candidate.get("reference_vs_replayer_no_pref_within_tolerance", 0),
            })
    return out


def choose_winners(candidates):
    by_trace = defaultdict(list)
    for row in candidates:
        if int(to_float(row.get("run_failed"))) == 0 and int(to_float(row.get("replay_transport_ok"))) == 1:
            by_trace[row["trace"]].append(row)
    winners = []
    for trace, rows in sorted(by_trace.items()):
        ordered = sorted(rows, key=lambda row: (-to_float(row.get("replay_ipc")), row.get("tag", "")))
        winner = dict(ordered[0])
        winner["winner_rank"] = 1
        winner["replayed_candidate_count"] = len(ordered)
        winner["runner_up_tag"] = ordered[1].get("tag", "") if len(ordered) > 1 else ""
        winner["runner_up_ipc"] = to_float(ordered[1].get("replay_ipc")) if len(ordered) > 1 else 0.0
        winner["ipc_margin_vs_runner_up"] = to_float(winner.get("replay_ipc")) - winner["runner_up_ipc"] if len(ordered) > 1 else 0.0
        winner["performance_gate_status"] = (
            "performance_pass"
            if to_float(winner.get("replay_ipc")) > to_float(winner.get("reference_best_normal_ipc"))
            else "performance_abstain"
        )
        winners.append(winner)
    return winners


def write_markdown(path, candidates, winners, baselines):
    by_trace_candidates = defaultdict(list)
    for row in candidates:
        by_trace_candidates[row["trace"]].append(row)
    by_trace_winner = {row["trace"]: row for row in winners}
    lines = [
        "# v3.9 replay comparison",
        "",
        "Candidate IPC is from keyed ListReplayer replay. Normal-prefetcher IPC is a reference from the supplied baseline summary. The no-pref IPC tolerance field checks whether the two binaries agree closely on their no-pref controls.",
        "",
    ]
    for trace in sorted(by_trace_candidates):
        lines.append("## {}".format(trace))
        winner = by_trace_winner.get(trace)
        if winner:
            lines.append(
                "Winner: `{}` — IPC {:.6f}; same-binary no-pref {:.6f}; best normal reference `{}` {:.6f}; status `{}`.".format(
                    winner.get("tag", ""),
                    to_float(winner.get("replay_ipc")),
                    to_float(winner.get("same_binary_no_pref_ipc")),
                    winner.get("reference_best_normal", ""),
                    to_float(winner.get("reference_best_normal_ipc")),
                    winner.get("performance_gate_status", ""),
                )
            )
        else:
            lines.append("No valid replay candidate completed.")
        lines.append("")
        lines.append("Current-run candidates:")
        for row in sorted(by_trace_candidates[trace], key=lambda item: -to_float(item.get("replay_ipc"))):
            lines.append(
                "- `{}`: IPC {:.6f}; Δ vs same-binary no-pref {:+.6f}; accuracy {:.4f}; timeliness {:.4f}; issued {}; transport_ok={}.".format(
                    row.get("tag", ""),
                    to_float(row.get("replay_ipc")),
                    to_float(row.get("delta_ipc_vs_same_binary_no_pref")),
                    to_float(row.get("selected_accuracy")),
                    to_float(row.get("timeliness")),
                    int(to_float(row.get("pf_issued"))),
                    int(to_float(row.get("replay_transport_ok"))),
                )
            )
        lines.append("")
        lines.append("Normal reference ranking:")
        for normal in baselines.get(trace, []):
            lines.append(
                "- `{}`: IPC {:.6f}; accuracy {:.4f}; timeliness {:.4f}; issued {}.".format(
                    normal.get("prefetcher", ""),
                    to_float(normal.get("ipc")),
                    to_float(normal.get("selected_accuracy") or normal.get("accuracy")),
                    to_float(normal.get("timeliness")),
                    int(to_float(normal.get("pf_issued"))),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--replay-input-root", required=True, type=Path)
    ap.add_argument("--same-binary-log-root", required=True, type=Path)
    ap.add_argument("--baseline-summary", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--no-pref-ipc-tolerance", type=float, default=0.002)
    args = ap.parse_args()

    plan = plan_rows(args.plan)
    baseline_rows = read_rows(args.baseline_summary)
    baseline_by_trace, normal_no_pref, best_normal = baseline_index(baseline_rows)
    candidates = [one_result(row, args, baseline_by_trace, normal_no_pref, best_normal) for row in plan]
    candidates.sort(key=lambda row: (row["trace"], row["tag"]))
    winners = choose_winners(candidates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "v3_9_replay_results.csv", candidates)
    write_rows(args.out_dir / "v3_9_final_winners.csv", winners)
    write_rows(args.out_dir / "v3_9_vs_all_normal_prefetchers.csv", comparisons(candidates, baseline_by_trace))
    write_rows(args.out_dir / "v3_9_final_winners_vs_all_normal_prefetchers.csv", comparisons(winners, baseline_by_trace))
    write_markdown(args.out_dir / "v3_9_comparison.md", candidates, winners, baseline_by_trace)

    expected = set(row["trace"] for row in plan)
    actual = set(row["trace"] for row in winners)
    missing = sorted(expected - actual)
    if missing:
        raise SystemExit("no valid replay winner for traces: {}".format(missing))

    for name in [
        "v3_9_replay_results.csv",
        "v3_9_final_winners.csv",
        "v3_9_vs_all_normal_prefetchers.csv",
        "v3_9_final_winners_vs_all_normal_prefetchers.csv",
        "v3_9_comparison.md",
    ]:
        print("[write] {}".format(args.out_dir / name))


if __name__ == "__main__":
    main()
