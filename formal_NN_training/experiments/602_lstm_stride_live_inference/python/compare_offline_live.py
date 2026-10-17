#!/usr/bin/env python3
"""Generate the offline-primary sufficiency analysis and live appendix."""

import argparse
import csv
import json
from pathlib import Path

from offline_sufficiency import (  # re-exported for policy regression tests
    KEY_BUDGETS,
    build_analysis,
    conclusions_for_hidden,
)


def load(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def tex_escape(value):
    if value is None:
        return "NA"
    return (
        str(value).replace("\\", r"\textbackslash{}")
        .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    )


def fmt(value, digits=4):
    return "NA" if value is None else ("{:,.%df}" % digits).format(value)


def pct(value, digits=2):
    return "NA" if value is None else ("{:.%df}\\%%" % digits).format(
        100.0 * value
    )


def pp(value, digits=2):
    return "NA" if value is None else ("{:+.%df}" % digits).format(value)


def integer(value):
    return "NA" if value is None else "{:,}".format(int(round(value)))


def plateau_label(row):
    if row.get("stable_0_5_percent_ipc_plateau_start"):
        return "start"
    if row.get("stable_0_5_percent_ipc_plateau_member"):
        return "yes"
    return "no"


def write_rows_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for field in fields:
                value = row.get(field)
                encoded[field] = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list)) else value
                )
            writer.writerow(encoded)


def write_conclusions_csv(path, analysis):
    rows = []
    for hidden in (8, 16):
        for key, value in sorted(
            analysis["conclusions"]["h{}".format(hidden)].items()
        ):
            rows.append({
                "hidden_size": hidden,
                "conclusion": key,
                "value": (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list)) else value
                ),
            })
    for key, value in sorted(analysis["analysis_policy"].items()):
        rows.append({
            "hidden_size": "all",
            "conclusion": "analysis_policy." + key,
            "value": value,
        })
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("hidden_size", "conclusion", "value")
        )
        writer.writeheader()
        writer.writerows(rows)


def key_rows(points, hidden):
    return [
        row for row in points
        if row.get("hidden_size") == hidden
        and row.get("budget_tag") in KEY_BUDGETS
    ]


def write_key_budget_table(path, hidden, points):
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{longtable}{@{}lrrrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Budget & Decisions & $K>0$ & $\sum K$ & Student act & Req./load "
            r"& Coverage & L2 miss & IPC & $\Delta$IPC & Plateau\\"
        ),
        r"\midrule",
        r"\endhead",
    ]
    for row in key_rows(points, hidden):
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\".format(
                tex_escape(row["budget_tag"]),
                integer(row.get("decision_rows")),
                integer(row.get("positive_count_rows")),
                integer(row.get("action_atoms")),
                pct(row.get("student_heldout_act_rate")),
                fmt(row.get("requests_per_l2_load"), 3),
                pct(row.get("coverage")),
                pct(row.get("l2_load_miss_rate")),
                fmt(row.get("ipc"), 5),
                pct(row.get("relative_ipc_difference_vs_same_h_20m"), 3),
                plateau_label(row),
            )
        )
    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        (
            r"\footnotesize $\Delta$IPC is signed and always compares h%d-$N$ "
            r"with h%d-i20m. Decisions, $K>0$, and $\sum K$ are teacher-label "
            r"statistics; student act is a held-out student behavior."
        ) % (hidden, hidden),
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def write_behavior_differences(path, points):
    lines = []
    for hidden in (8, 16):
        lines.extend([
            r"\subsection*{h%d auxiliary differences from h%d-i20m}" % (
                hidden, hidden
            ),
            r"\begin{center}\scriptsize",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            (
                r"Budget & $\Delta$coverage (pp) & $\Delta$miss (pp) & "
                r"$\Delta$req./load & $\Delta$act (pp) & "
                r"$\Delta$timeliness (pp) & $\Delta$replay actions\\"
            ),
            r"\midrule",
        ])
        for row in key_rows(points, hidden):
            lines.append(
                "{} & {} & {} & {} & {} & {} & {}\\\\".format(
                    tex_escape(row["budget_tag"]),
                    pp(row.get(
                        "coverage_percentage_point_difference_vs_same_h_20m"
                    )),
                    pp(row.get(
                        "l2_miss_rate_percentage_point_difference_vs_same_h_20m"
                    )),
                    pp(row.get(
                        "request_pressure_difference_vs_same_h_20m"
                    ), 3),
                    pp(row.get(
                        "student_act_rate_percentage_point_difference_vs_same_h_20m"
                    )),
                    pp(row.get(
                        "timeliness_percentage_point_difference_vs_same_h_20m"
                    )),
                    integer(row.get(
                        "replay_action_count_difference_vs_same_h_20m"
                    )),
                )
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ])
    lines.append(
        "These diagnostics are not silently combined with the IPC criterion. "
        "System-performance equivalence concerns IPC/cache outcomes; no "
        "policy-equivalence claim is made unless act rate, request pressure, "
        "and replay-action behavior are also explicitly close."
    )
    Path(path).write_text("\n".join(lines) + "\n")


def best_aggressive(conclusion):
    rows = conclusion.get(
        "offline_high_performing_small_budget_candidates"
    ) or []
    return max(
        rows,
        key=lambda row: row.get(
            "relative_ipc_uplift_vs_same_h_20m", float("-inf")
        ),
    ) if rows else None


def write_conclusions_tex(path, analysis):
    lines = []
    for hidden in (8, 16):
        item = analysis["conclusions"]["h{}".format(hidden)]
        plateau = item.get("offline_stable_plateau")
        aggressive = best_aggressive(item)
        lines.extend([
            r"\subsection*{h%d seed-7 conclusion}" % hidden,
            r"\begin{itemize}",
            (
                r"\item Same-hidden-size reference: h%d-i20m, offline IPC %s."
                % (hidden, fmt(item.get("offline_20m_ipc"), 5))
            ),
            (
                r"\item Minimum reaching at least 99/99.5/99.9\%% of 20M "
                r"IPC: %s / %s / %s."
            ) % (
                tex_escape(item.get(
                    "smallest_budget_reaching_at_least_99_percent_of_20m_offline_ipc"
                )),
                tex_escape(item.get(
                    "smallest_budget_reaching_at_least_99_5_percent_of_20m_offline_ipc"
                )),
                tex_escape(item.get(
                    "smallest_budget_reaching_at_least_99_9_percent_of_20m_offline_ipc"
                )),
            ),
            (
                r"\item Minimum within $\pm$1.0/$\pm$0.5/$\pm$0.1\%% of "
                r"20M IPC: %s / %s / %s."
            ) % (
                tex_escape(item.get(
                    "smallest_within_1.0_percent_20m_offline_ipc"
                )),
                tex_escape(item.get(
                    "smallest_within_0.5_percent_20m_offline_ipc"
                )),
                tex_escape(item.get(
                    "smallest_within_0.1_percent_20m_offline_ipc"
                )),
            ),
        ])
        if plateau:
            lines.append(
                r"\item Stable 0.5\%% offline IPC plateau starts at %s; the "
                r"maximum same-H 20M IPC deviation over the observed suffix "
                r"is %s."
                % (
                    tex_escape(plateau["budget_tag"]),
                    pct(plateau["max_ipc_relative_deviation"], 3),
                )
            )
        else:
            lines.append(
                r"\item No pre-20M stable 0.5\%% offline IPC plateau is observed."
            )
        if aggressive:
            signals = aggressive.get(
                "aggressive_signals_vs_same_h_20m"
            ) or []
            signal_labels = {
                "coverage": "coverage",
                "requests_per_l2_load": "request pressure",
                "student_heldout_act_rate": "student act rate",
            }
            lines.append(
                r"\item Aggressive high-performing candidate: %s, with %s "
                r"IPC uplift over same-H 20M. Its differing %s prevent a "
                r"policy-equivalence label."
                % (
                    tex_escape(aggressive["budget_tag"]),
                    pct(aggressive[
                        "relative_ipc_uplift_vs_same_h_20m"
                    ], 3),
                    tex_escape(", ".join(
                        signal_labels.get(signal, signal)
                        for signal in signals
                    ) if signals else "reported behavior metrics"),
                )
            )
        lines.extend([r"\end{itemize}", ""])
    lines.extend([
        r"\paragraph{Scope.} These conclusions apply only to "
        r"602.gcc\_s-734B, seed 7, and the original offline keyed-replay "
        r"protocol. Multi-seed evidence is required before a broader claim.",
        r"\paragraph{Training-work limitation.} Epochs remain fixed. Larger "
        r"prefixes therefore receive more optimizer steps, so this experiment "
        r"answers how much trace and training work are needed under the current "
        r"recipe; it does not isolate unique-data diversity from update count.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def write_live_tex(path, live_points):
    lines = [
        (
            "Live measurements are secondary functional deployment validation "
            "with zero modeled NN inference latency. Host wall-clock "
            "nanoseconds are not simulated CPU latency."
        ),
        "",
    ]
    if not live_points:
        lines.append(
            r"\textbf{No qualifying live rows are present.} Missing points "
            r"are neither interpolated nor used in the offline conclusion."
        )
    else:
        lines.extend([
            r"\begin{center}\scriptsize",
            r"\begin{tabular}{@{}llrrrr@{}}",
            r"\toprule",
            (
                r"Model & Budget & Offline$_N$ IPC & Live$_N$ IPC & "
                r"Live$_N-$offline$_N$ & Live $\Delta$ vs same-H live 20M\\"
            ),
            r"\midrule",
        ])
        for row in live_points:
            lines.append(
                "h{} & {} & {} & {} & {} & {}\\\\".format(
                    row["hidden_size"], tex_escape(row["budget_tag"]),
                    fmt(row.get("same_checkpoint_offline_ipc"), 5),
                    fmt(row.get("ipc"), 5),
                    fmt(row.get("live_minus_same_checkpoint_offline_ipc"), 5),
                    pct(row.get(
                        "relative_live_ipc_difference_vs_same_h_live_20m"
                    ), 3),
                )
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                "The first difference tests same-checkpoint implementation "
                "parity (live-$N$ versus offline-$N$). The final column tests "
                "live training-size sufficiency (live-$N$ versus same-H "
                "live-i20m)."
            ),
        ])
    Path(path).write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    offline_data = load(args.run_dir / "offline_sweep_results.json")
    if offline_data is None:
        raise RuntimeError("offline_sweep_results.json is required")
    live_data = load(
        args.run_dir / "live_sweep_results.json",
        {"schema_version": 1, "points": [], "references": {}},
    )
    offline = offline_data.get("points", [])
    live = [
        row for row in live_data.get("points", [])
        if row.get("state_mode") == "parity"
        and row.get("live_run_mode", "live") == "live"
    ]
    analysis = build_analysis(offline, live)
    analysis["references"] = {
        "offline": offline_data.get("references", {}),
        "live": live_data.get("references", {}),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "offline_sufficiency.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    write_rows_csv(
        args.run_dir / "offline_sufficiency.csv",
        analysis["offline_points"],
    )
    if analysis["live_secondary"]["points"]:
        write_rows_csv(
            args.run_dir / "live_validation_comparisons.csv",
            analysis["live_secondary"]["points"],
        )
    (args.run_dir / "live_validation_comparisons.json").write_text(
        json.dumps(analysis["live_secondary"], indent=2, sort_keys=True)
        + "\n"
    )

    report_dir = args.run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    conclusions_payload = dict(analysis["conclusions"])
    conclusions_payload.update({
        "schema_version": analysis["schema_version"],
        "scope": analysis["scope"],
        "primary_question": analysis["primary_question"],
        "comparison_rule": analysis["comparison_rule"],
        "analysis_policy": analysis["analysis_policy"],
        "live_secondary_point_count": len(
            analysis["live_secondary"]["points"]
        ),
    })
    (report_dir / "conclusions.json").write_text(
        json.dumps(conclusions_payload, indent=2, sort_keys=True) + "\n"
    )
    write_conclusions_csv(report_dir / "conclusions.csv", analysis)
    write_key_budget_table(
        report_dir / "generated_h8_key_budgets.tex", 8,
        analysis["offline_points"],
    )
    write_key_budget_table(
        report_dir / "generated_h16_key_budgets.tex", 16,
        analysis["offline_points"],
    )
    write_behavior_differences(
        report_dir / "generated_behavior_differences.tex",
        analysis["offline_points"],
    )
    write_conclusions_tex(
        report_dir / "generated_conclusions.tex", analysis
    )
    write_live_tex(
        report_dir / "generated_live_validation.tex",
        analysis["live_secondary"]["points"],
    )
    print("[ok] offline-primary same-H sufficiency analysis generated")
    print("[scope] {}".format(analysis["scope"]))


if __name__ == "__main__":
    main()
