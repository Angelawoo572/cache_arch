#!/usr/bin/env python3
"""Generate the three primary report figures as PGFPlots TeX fragments.

This analysis-only script uses the Python standard library.  It reads the
already aggregated JSON files and never imports matplotlib, launches ChampSim,
or changes a raw result.  Overleaf performs the final figure rendering.
"""

import argparse
import json
import math
from pathlib import Path


COLORS = {8: "hEight", 16: "hSixteen"}
MARKS = {8: "*", 16: "square*"}


def load_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rows_for(rows, hidden, field):
    return sorted(
        (
            row for row in rows
            if row.get("hidden_size") == hidden and finite(row.get(field))
        ),
        key=lambda row: row["instruction_budget"],
    )


def coordinates(rows, hidden, field, scale=1.0):
    selected = rows_for(rows, hidden, field)
    return " ".join(
        "({:.12g},{:.12g})".format(
            float(row["instruction_budget"]), scale * float(row[field])
        )
        for row in selected
    )


def one_coordinate(instruction_budget, value):
    if not finite(instruction_budget) or not finite(value):
        return ""
    return "({:.12g},{:.12g})".format(
        float(instruction_budget), float(value)
    )


def best_aggressive(conclusion, preferred_tag=None):
    candidates = conclusion.get(
        "offline_high_performing_small_budget_candidates"
    ) or []
    if preferred_tag:
        preferred = [
            row for row in candidates
            if row.get("budget_tag") == preferred_tag
        ]
        if preferred:
            return preferred[0]
    return max(
        candidates,
        key=lambda row: row.get(
            "relative_ipc_uplift_vs_same_h_20m", float("-inf")
        ),
    ) if candidates else None


def reference_ipc(analysis, name):
    return (
        analysis.get("references", {}).get("offline", {}).get(name) or {}
    ).get("ipc")


def line_coordinates(left, right, value):
    return "({:.12g},{:.12g}) ({:.12g},{:.12g})".format(
        left, value, right, value
    )


def write_figure_one(path, analysis):
    rows = analysis["offline_points"]
    valid = [
        row for row in rows
        if finite(row.get("instruction_budget")) and finite(row.get("ipc"))
    ]
    if not valid:
        raise RuntimeError("offline_sufficiency.json has no measured IPC rows")
    left = min(float(row["instruction_budget"]) for row in valid)
    right = max(float(row["instruction_budget"]) for row in valid)
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.98\linewidth,height=0.56\linewidth,",
        r"  xmode=log,log basis x=10,",
        r"  xlabel={trace-start retired training instructions},",
        r"  ylabel={offline keyed-replay IPC},",
        r"  grid=both,minor grid style={gray!10},major grid style={gray!22},",
        r"  legend style={font=\scriptsize,at={(0.5,-0.22)},anchor=north,legend columns=3},",
        r"  tick label style={font=\small},label style={font=\small},",
        r"  enlargelimits=false]",
    ]
    for hidden in (8, 16):
        conclusion = analysis["conclusions"]["h{}".format(hidden)]
        reference = conclusion.get("offline_20m_ipc")
        if not finite(reference):
            continue
        low = float(reference) * 0.995
        high = float(reference) * 1.005
        color = COLORS[hidden]
        lines.extend([
            r"\addplot[name path=h{}low,draw=none] coordinates {{{}}};".format(
                hidden, line_coordinates(left, right, low)
            ),
            r"\addplot[name path=h{}high,draw=none] coordinates {{{}}};".format(
                hidden, line_coordinates(left, right, high)
            ),
            r"\addplot[{},fill opacity=0.10,draw=none] fill between[of=h{}low and h{}high];".format(
                color, hidden, hidden
            ),
            r"\addplot[{},densely dashed,line width=0.8pt] coordinates {{{}}};".format(
                color, line_coordinates(left, right, float(reference))
            ),
            r"\addlegendentry{{h{} i20m $\pm0.5\%$}}".format(hidden),
        ])
    for name, style, color, label in (
        ("no_pref", "densely dotted", "black!65", "no prefetch"),
        ("offline_stride", "dashdotted", "strideGreen", "offline Stride"),
    ):
        value = reference_ipc(analysis, name)
        if finite(value):
            lines.extend([
                r"\addplot[{}, {},line width=1pt] coordinates {{{}}};".format(
                    color, style,
                    line_coordinates(left, right, float(value)),
                ),
                r"\addlegendentry{{{}}}".format(label),
            ])
    for hidden in (8, 16):
        color = COLORS[hidden]
        lines.extend([
            r"\addplot[{},mark={},line width=1.25pt,mark size=2.2pt] coordinates {{{}}};".format(
                color, MARKS[hidden], coordinates(rows, hidden, "ipc")
            ),
            r"\addlegendentry{{h{} measured}}".format(hidden),
        ])
        conclusion = analysis["conclusions"]["h{}".format(hidden)]
        plateau = conclusion.get("offline_stable_plateau")
        if plateau:
            matching = [
                row for row in rows
                if row.get("hidden_size") == hidden
                and row.get("instruction_budget")
                == plateau.get("instruction_budget")
            ]
            if matching and finite(matching[0].get("ipc")):
                lines.extend([
                    r"\addplot[only marks,{},mark=triangle*,mark size=4pt] coordinates {{{}}};".format(
                        color, one_coordinate(
                            plateau["instruction_budget"], matching[0]["ipc"]
                        )
                    ),
                    r"\addlegendentry{{h{} stable start: {}}}".format(
                        hidden, plateau["budget_tag"]
                    ),
                ])
        aggressive = best_aggressive(
            conclusion, "i250k" if hidden == 8 else "i100k"
        )
        if aggressive:
            lines.extend([
                r"\addplot[only marks,{},mark=star,mark size=5pt,mark options={{solid,draw=black}}] coordinates {{{}}};".format(
                    color, one_coordinate(
                        aggressive["instruction_budget"], aggressive["ipc"]
                    )
                ),
                r"\addlegendentry{{h{} aggressive: {}}}".format(
                    hidden, aggressive["budget_tag"]
                ),
            ])
    lines.extend([
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Primary offline keyed-replay IPC. Each curve is compared with its own 20M reference. Stars are aggressive high-performing candidates, not policy-equivalent points.}",
        r"\label{fig:offline-ipc}",
        r"\end{figure}",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def panel(lines, rows, field, scale, ylabel, show_x=False, legend=False,
          ipc_band=False):
    options = [r"ylabel={%s}" % ylabel]
    if not show_x:
        options.append(r"xticklabels=\empty")
    else:
        options.append(r"xlabel={trace-start retired training instructions}")
    lines.append(r"\nextgroupplot[{}]".format(",".join(options)))
    valid = [
        row for row in rows
        if finite(row.get("instruction_budget")) and finite(row.get(field))
    ]
    left = min(float(row["instruction_budget"]) for row in valid)
    right = max(float(row["instruction_budget"]) for row in valid)
    if ipc_band:
        if valid:
            lines.extend([
                r"\addplot[name path=ipcLow,draw=none] coordinates {{{}}};".format(
                    line_coordinates(left, right, -0.5)
                ),
                r"\addplot[name path=ipcHigh,draw=none] coordinates {{{}}};".format(
                    line_coordinates(left, right, 0.5)
                ),
                r"\addplot[strideGreen,fill opacity=0.08,draw=none] fill between[of=ipcLow and ipcHigh];",
            ])
    lines.append(
        r"\addplot[black!55,densely dotted] coordinates {{{}}};".format(
            line_coordinates(left, right, 0.0)
        )
    )
    for hidden in (8, 16):
        lines.append(
            r"\addplot[{},mark={},line width=1.1pt,mark size=1.9pt] coordinates {{{}}};".format(
                COLORS[hidden], MARKS[hidden],
                coordinates(rows, hidden, field, scale),
            )
        )
        if legend:
            lines.append(r"\addlegendentry{{h{}}}".format(hidden))


def write_figure_two(path, analysis):
    rows = analysis["offline_points"]
    panels = (
        ("relative_ipc_difference_vs_same_h_20m", 100.0,
         r"$\Delta$IPC (\%)", True),
        ("coverage_percentage_point_difference_vs_same_h_20m", 1.0,
         r"$\Delta$coverage (pp)", False),
        ("l2_miss_rate_percentage_point_difference_vs_same_h_20m", 1.0,
         r"$\Delta$L2 miss (pp)", False),
        ("request_pressure_difference_vs_same_h_20m", 1.0,
         r"$\Delta$requests/load", False),
        ("student_act_rate_percentage_point_difference_vs_same_h_20m", 1.0,
         r"$\Delta$student act (pp)", False),
    )
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=1 by 5,vertical sep=0.42cm},",
        r"  width=0.96\linewidth,height=0.205\linewidth,",
        r"  xmode=log,log basis x=10,",
        r"  grid=both,minor grid style={gray!10},major grid style={gray!22},",
        r"  tick label style={font=\scriptsize},label style={font=\scriptsize},",
        r"  legend style={font=\scriptsize,legend columns=2,at={(0.99,0.97)},anchor=north east}]",
    ]
    for index, (field, scale, ylabel, ipc_band) in enumerate(panels):
        panel(
            lines, rows, field, scale, ylabel,
            show_x=index == len(panels) - 1,
            legend=index == 0,
            ipc_band=ipc_band,
        )
    lines.extend([
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{Signed differences from the corresponding same-hidden-size 20M model. Auxiliary cache and policy/traffic dimensions are reported separately from the IPC criterion.}",
        r"\label{fig:same-h-differences}",
        r"\end{figure}",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def write_figure_three(path, summary):
    rows = sorted(
        summary["points"], key=lambda row: row["instruction_budget"]
    )
    panels = (
        ("decision_rows", "decision rows"),
        ("positive_count_rows", r"teacher $K>0$ rows"),
        ("action_atoms", r"teacher action atoms $=\sum K$"),
    )
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 1,horizontal sep=1.15cm},",
        r"  width=0.315\linewidth,height=0.29\linewidth,",
        r"  xmode=log,log basis x=10,",
        r"  xlabel={training instructions},",
        r"  grid=both,minor grid style={gray!10},major grid style={gray!22},",
        r"  tick label style={font=\scriptsize},label style={font=\scriptsize},",
        r"  scaled y ticks=true]",
    ]
    for field, ylabel in panels:
        values = " ".join(
            "({:.12g},{:.12g})".format(
                float(row["instruction_budget"]), float(row[field])
            )
            for row in rows if finite(row.get(field))
        )
        lines.extend([
            r"\nextgroupplot[ylabel={%s}]" % ylabel,
            r"\addplot[strideGreen,mark=*,line width=1.1pt,mark size=1.8pt] coordinates {%s};" % values,
        ])
    lines.extend([
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{Training supervision versus prefix. All three quantities are teacher-label statistics, not student act rates.}",
        r"\label{fig:training-supervision}",
        r"\end{figure}",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    analysis_path = args.run_dir / "offline_sufficiency.json"
    summary_path = args.run_dir / "training_prefix_data_summary.json"
    if not analysis_path.is_file():
        raise RuntimeError("missing analysis: {}".format(analysis_path))
    if not summary_path.is_file():
        raise RuntimeError("missing supervision summary: {}".format(
            summary_path
        ))
    analysis = load_json(analysis_path)
    summary = load_json(summary_path)
    out = args.out_dir or args.run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    outputs = (
        out / "generated_figure1_offline_ipc.tex",
        out / "generated_figure2_same_h_differences.tex",
        out / "generated_figure3_training_supervision.tex",
    )
    write_figure_one(outputs[0], analysis)
    write_figure_two(outputs[1], analysis)
    write_figure_three(outputs[2], summary)
    for output in outputs:
        print("[ok] wrote {}".format(output))
    print("[scope] TeX/PGFPlots only; no matplotlib or local PDF required")


if __name__ == "__main__":
    main()
