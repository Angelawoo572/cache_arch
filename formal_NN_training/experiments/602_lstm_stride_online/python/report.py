#!/usr/bin/env python3
"""Generate the scientific LaTeX report and small plotted inputs from runs.

This program never starts simulation/training and never fills missing results
with historical bars. Run pdflatex on the emitted source and render for QA.
"""
import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path

from analyze import COUNTERS, PROGRESS, load_oracle, metrics, number, write_csv


def tex(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def fmt(value, digits=3, percent=False):
    v = number(value)
    if v is None:
        return "NA"
    if percent:
        return f"{100*v:.{digits}f}\\%"
    return f"{v:,.{digits}f}"


def observed_range(values, scale=1, digits=0):
    available = [number(v) / scale for v in values if number(v) is not None]
    if not available:
        return "NA"
    lo, hi = min(available), max(available)
    return fmt(lo, digits) if lo == hi else fmt(lo, digits) + "--" + fmt(hi, digits)


def rows(path):
    return list(csv.DictReader(path.open())) if path.is_file() and path.stat().st_size else []


def label(row):
    a = row["arm"]
    if a == "no_pref":
        return "No prefetch"
    if a == "stride":
        return "Stride"
    prefix = f"h{row.get('hidden_size')} "
    if a == "frozen":
        result = prefix + "frozen " + str(row.get("checkpoint", ""))
    else:
        result = prefix + ("scratch" if a == "scratch" else "warm i1m") + f" s{row.get('seed')}"
    if str(row.get("state_capacity")) == "64":
        result += f" /64 /MAC{row.get('service_case')}"
    return result


def functional(row):
    return row.get("protocol") == "cold" and row.get("phase") == "final" and str(row.get("state_capacity", 0)) in ("0", "unbounded", "None") and str(row.get("service_case", 0)) in ("0", "zero", "None")


def storage_label(row):
    result = re.sub(r" s\d+", "", label(row))
    return result.replace("frozen " + str(row.get("checkpoint", "")), "frozen") if row["arm"] == "frozen" else result


def history(root):
    oracle = load_oracle()
    baseline_path = root / "references/logs/no_pref.log"
    baseline = oracle.parse_log(baseline_path) if baseline_path.is_file() else None
    result = []
    for h in (8, 16):
        for budget in ("i1m", "i20m"):
            point = root / f"points/h{h}/{budget}/seed7"
            meta_path = point / "point_metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            for mode, relative in (("offline keyed replay", "offline/replay.log"), ("frozen live parity", "live/parity/live/run.log")):
                log = point / relative
                if not log.is_file() or "Finished CPU" not in log.read_text(errors="replace"):
                    continue
                stats = oracle.parse_log(log)
                result.append({"hidden_size": h, "budget": budget, "seed": 7, "mode": mode, **{k: stats.get(k) for k in ("instructions", "cycles") + COUNTERS}, **metrics(stats, baseline), **{k: meta.get(k) for k in ("decision_rows", "positive_count_rows", "action_atoms", "epochs", "optimizer_steps", "learning_rate", "parameter_count", "weight_bytes")}, "source": f"points/h{h}/{budget}/seed7/{relative}"})
    return result


def historical_compatibility(root, data):
    """Compare every original parser field; tolerance does not mean equality."""
    oracle = load_oracle()
    comparisons = []
    for run in data.get("runs", []):
        ident = run["identity"]
        if not run.get("complete") or ident.get("protocol") != "historical":
            continue
        h, budget = ident["hidden_size"], ident["checkpoint"]
        old_log = root / f"points/h{h}/{budget}/seed7/live/parity/live/run.log"
        if not old_log.is_file():
            continue
        old, new = oracle.parse_log(old_log), run["counters"]
        for key, value in old.items():
            comparisons.append({"run_id": ident["run_id"], "hidden_size": h, "checkpoint": budget, "field": key, "old_value": value, "new_value": new.get(key), "difference": new[key] - value if new.get(key) is not None else None, "exact_match": new.get(key) == value})
    return comparisons


def table(headers, body, alignment=None, size="small", long=False):
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    environment = "longtable" if long else "tabular"
    start = ["\\begin{center}", f"\\{size}", f"\\begin{{{environment}}}{{{alignment}}}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    if long:
        start += ["\\endhead"]
    for row in body:
        start.append(" & ".join(str(v) for v in row) + r" \\")
    return "\n".join(start + ["\\bottomrule", f"\\end{{{environment}}}", "\\end{center}"])


COLORS = {"no_pref": "#444444", "stride": "#777777", "frozen": "#0072B2", "scratch": "#D55E00", "warm": "#009E73"}
PLOT_DIR = None
PLOT_NUMBER = 0


def style(row):
    dash = "solid"
    if row.get("checkpoint") == "i20m":
        dash = "dashed"
    if row.get("arm") == "scratch":
        dash = {"7": "solid", "17": "dashed", "27": "densely dotted"}.get(str(row.get("seed")), "solid")
    return f"color={COLORS.get(row['arm'], 'violet')},{dash},line width=0.8pt"


def plot(selected, file_by_run, metric, ylabel, title, width=r"0.92\linewidth", height="6cm", legend=True, xmax=None, markers=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global PLOT_NUMBER
    available = [r for r in selected if r["run_id"] in file_by_run]
    if not available:
        return r"\emph{No completed measurement series available for this plot.}"
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    for row in available:
        series = rows(PLOT_DIR / file_by_run[row["run_id"]])
        xs = [number(p["x"]) for p in series]
        ys = [number(p.get(metric)) for p in series]
        linestyle = "--" if row.get("checkpoint") == "i20m" else "-"
        if row["arm"] == "scratch":
            linestyle = {"7": "-", "17": "--", "27": ":"}.get(str(row.get("seed")), "-")
        if str(row.get("state_capacity")) == "64":
            linestyle = {"0": "-", "4": ":", "16": "--"}.get(str(row.get("service_case")), "-")
        ax.plot(xs, [float("nan") if y is None else y for y in ys],
                color=COLORS.get(row["arm"], "#CC79A7"), linestyle=linestyle,
                linewidth=1.25, label=label(row))
    ax.set(title=title, xlabel="Retired instructions since 25M trace boundary (millions)", ylabel=ylabel)
    ax.grid(alpha=.22, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if "saved" in metric:
        ax.axhline(0, color="#333333", linewidth=.65)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    if metric == "occupancy":
        ax.set_ylim(0, 103)
    if xmax is not None:
        ax.set_xlim(0, xmax)
        visible = [number(p.get(metric)) for r in available for p in rows(PLOT_DIR / file_by_run[r["run_id"]]) if (number(p.get("x")) or 0) <= xmax]
        visible = [v for v in visible if v is not None]
        if visible and metric != "occupancy":
            lo, hi = min(visible), max(visible)
            pad = max((hi-lo)*.08, abs(hi)*.03, .001)
            ax.set_ylim(lo-pad, hi+pad)
    for x, note in markers:
        ax.axvline(x, color="#555555", linestyle=":", linewidth=.7, alpha=.7)
        ax.text(x, .98, note, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=6.5)
    if legend:
        ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2))
    PLOT_NUMBER += 1
    filename = f"figures/figure_{PLOT_NUMBER:02d}_{metric}.pdf"
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
    fig.savefig(PLOT_DIR / filename.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"\\includegraphics[width={width}]{{{filename}}}"


def figure(title, contents, caption):
    return "\n".join([r"\begin{figure}[p]", r"\centering", contents, r"\caption{" + tex(caption) + "}", r"\end{figure}"])


def finite_summary_plot(hardware, storage, stride):
    """Measured timing/storage sensitivity, with absent arms left absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global PLOT_NUMBER
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), layout="constrained")
    points = []
    for run in hardware:
        entries = [s for s in storage if s["run_id"] == run["run_id"]]
        sizes = [number(s.get("configured_max_bytes")) for s in entries]
        size = sum(sizes)/1024 if sizes and all(v is not None for v in sizes) else None
        eligible = number(run.get("eligible_callbacks"))
        points.append({"run_id": run["run_id"], "arm": run["arm"], "service_case": run["service_case"], "ipc": number(run.get("ipc")), "admitted_fraction": number(run.get("admitted_decisions"))/eligible if eligible else None, "completed_fraction": number(run.get("completed_decisions"))/eligible if eligible else None, "updates": number(run.get("completed_updates")), "configured_kib": size})
    for arm_name, title in (("frozen", "Frozen i1m"), ("scratch", "Random-start s7")):
        group = sorted([p for p in points if p["arm"] == arm_name], key=lambda p: int(p["service_case"]))
        if not group:
            continue
        x = [{0: 0, 4: 1, 16: 2}[int(p["service_case"])] for p in group]
        color = COLORS[arm_name]
        axes[0, 0].plot(x, [p["ipc"] for p in group], "o-", color=color, label=title)
        axes[0, 1].plot(x, [p["completed_fraction"] for p in group], "o-", color=color, label=title + " completed")
        axes[0, 1].plot(x, [p["admitted_fraction"] for p in group], "x:", color=color, label=title + " admitted")
        if arm_name == "scratch":
            axes[1, 0].plot(x, [p["updates"] for p in group], "o-", color=color, label=title)
        for point in group:
            if point["configured_kib"] is not None:
                axes[1, 1].scatter(point["configured_kib"], point["ipc"], color=color, marker={0:"o", 4:"s", 16:"^"}[int(point["service_case"])], s=35)
                offset = {0: (4, 4), 4: (4, 4), 16: (4, 18)}[int(point["service_case"])]
                axes[1, 1].annotate("zero" if int(point["service_case"]) == 0 else f"{point['service_case']} MAC", (point["configured_kib"], point["ipc"]), xytext=offset, textcoords="offset points", fontsize=6.5)
    if stride and number(stride.get("ipc")) is not None:
        for ax in (axes[0, 0], axes[1, 1]):
            ax.axhline(number(stride["ipc"]), color=COLORS["stride"], linestyle="--", linewidth=.8, label="Stride")
    for ax, title, ylabel in ((axes[0, 0], "End-to-end performance", "IPC"), (axes[0, 1], "Inference admission and completion", "Fraction of eligible callbacks"), (axes[1, 0], "Online optimization completed", "Completed updates")):
        ax.set(title=title, xlabel="Inference/training MACs per cycle", ylabel=ylabel, xticks=[0, 1, 2], xticklabels=["zero cost", "4", "16"])
    axes[1, 1].set(title="Configured storage versus performance", xlabel="Total logical configured storage (KiB)", ylabel="IPC", xscale="log")
    for ax in axes.flat:
        ax.grid(alpha=.2, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    axes[0, 1].legend(fontsize=6, frameon=False)
    PLOT_NUMBER += 1
    filename = f"figures/figure_{PLOT_NUMBER:02d}_finite_sensitivity.pdf"
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
    fig.savefig(PLOT_DIR / filename.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_csv(PLOT_DIR / "figures/finite_sensitivity.csv", points)
    return f"\\includegraphics[width=0.98\\linewidth]{{{filename}}}"


def generate(results_dir, out_dir, historical_root=None, recipe_path=None):
    global PLOT_DIR, PLOT_NUMBER
    PLOT_DIR, PLOT_NUMBER = out_dir, 0
    data = json.loads((results_dir / "results.json").read_text())
    summaries = data["summary"]
    complete = [r for r in summaries if r["status"] == "COMPLETE"]
    final = [r for r in complete if r["protocol"] in ("cold", "historical") and r.get("phase") == "final"]
    primary = [r for r in final if functional(r)]
    historic = history(historical_root) if historical_root else rows(results_dir / "historical.csv")
    if historic:
        write_csv(results_dir / "historical.csv", historic)
    compatibility = historical_compatibility(historical_root, data) if historical_root else rows(results_dir / "historical_compatibility.csv")
    if compatibility:
        write_csv(results_dir / "historical_compatibility.csv", compatibility)
    recipe = json.loads(recipe_path.read_text()) if recipe_path and recipe_path.is_file() else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    winrows, early_rows, sample_rows = rows(results_dir / "windows_1m.csv"), rows(results_dir / "windows_100k.csv"), rows(results_dir / "snapshots.csv")
    window_files, early_files, occupancy_files = {}, {}, {}
    for index, run in enumerate(final):
        selected = [r for r in winrows if r["run_id"] == run["run_id"]]
        if selected:
            points = []
            for row in selected:
                points.append({"x": number(row["end_instructions"]) / 1e6, **{k: number(row.get(k)) for k in ("ipc", "useful_prefetch_coverage", "selected_accuracy", "legacy_timeliness", "request_pressure", "cumulative_cycles_saved_vs_stride", "window_cycles_saved_vs_stride", "total_eligible_callbacks", "total_completed_updates")}})
            relative = f"figures/windows_{index:02d}.csv"
            # nan is a missing/censored ordinate, never a zero result.
            write_csv(out_dir / relative, [{k: "nan" if v is None else v for k, v in p.items()} for p in points])
            window_files[run["run_id"]] = relative
        selected = [r for r in early_rows if r["run_id"] == run["run_id"] and (number(r.get("end_instructions")) or 0) <= 4_000_000]
        if selected:
            relative = f"figures/early_{index:02d}.csv"
            write_csv(out_dir / relative, [{"x": number(row["end_instructions"]) / 1e6, **{k: number(row.get(k)) for k in ("ipc", "useful_prefetch_coverage", "selected_accuracy", "legacy_timeliness", "request_pressure", "cumulative_cycles_saved_vs_stride", "window_cycles_saved_vs_stride", "total_eligible_callbacks", "total_completed_updates")}} for row in selected])
            early_files[run["run_id"]] = relative
        selected = [r for r in sample_rows if r["run_id"] == run["run_id"]]
        if selected:
            points = []
            for row in selected:
                cap = number(row.get("l2_capacity_lines", row.get("capacity_lines")))
                occ = number(row.get("occupancy_lines"))
                replacements = number(row.get("replacements"))
                pending, resident = number(row.get("cohort_pending")), number(row.get("cohort_resident_unused"))
                points.append({"x": number(row["instructions"]) / 1e6, "occupancy": 100 * occ / cap if occ is not None and cap else "nan", "replacement_turnovers": replacements / cap if replacements is not None and cap else "nan", "updates": number(row.get("completed_updates")) or 0, "censored_requests": pending + resident if pending is not None and resident is not None else "nan"})
            relative = f"figures/cache_{index:02d}.csv"
            write_csv(out_dir / relative, points)
            occupancy_files[run["run_id"]] = relative

    preamble = r"""\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,longtable,array,graphicx,pdflscape}
\usepackage{amsmath,xcolor}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Causal Online Learning for a Tiny Neural Stride Prefetcher},pdfauthor={602 Stride online experiment}}
\definecolor{teal}{RGB}{0,128,128}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{4pt}
\title{How Quickly Does a Tiny Neural Prefetcher Become Useful?\\\large Causal Online Learning for 602.gcc\_s-734B}
\author{602 Stride online experiment}
\date{Measured simulator/CPU co-simulation}
\begin{document}
\maketitle
"""
    text = [preamble, r"\section{Measured answer and scope}"]
    text.append(f"This report contains {len(primary)} completed primary cold-start runs and {len(final) - len(primary)} completed compatibility or finite-resource runs. The analysis discovered {data['total_discovered_runs']} run directories; {data['completed_runs']} had a finished simulator log with the required raw counters. Software fixtures and development pilots are excluded from research comparisons.")
    online = [r for r in primary if r["arm"] in ("scratch", "warm")]
    if online:
        wins = [r for r in online if (number(r.get("cycles_saved_vs_stride")) or 0) > 0]
        text.append(f"Among {len(online)} completed functional online arms, {len(wins)} saved total measured cycles relative to the matched conventional Stride run. These are per-run observations on one trace slice, not statistical equivalence or a hardware implementation claim.")
        for h in (8, 16):
            for kind, description in (("scratch", "random-start learning"), ("warm", "i1m warm-start adaptation")):
                group = [r for r in online if str(r["hidden_size"]) == str(h) and r["arm"] == kind]
                if not group:
                    continue
                ids = {r["run_id"] for r in group}
                seed_list = ", ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"]))
                relevant = [m for m in data["learning_milestones"] if m["run_id"] in ids]
                for key, what in (("positive_cumulative_cycle_saving_vs_stride", "recovered startup cost versus Stride"), ("99_percent_frozen_i1m_window_ipc", "attained 99\\% of same-H frozen-i1m window IPC")):
                    observations = [m for m in relevant if m["milestone"] == key and m["status"] == "REACHED"]
                    if observations:
                        text.append(f"For h{h} {description} (completed seeds {seed_list}), {len(observations)}/{len(group)} runs {what}. The first qualifying endpoint was {observed_range([m['first_window_end'] for m in observations], 1e6, 1)}M retired instructions from the slice boundary, confirmed at {observed_range([m['confirmed_at_instructions'] for m in observations], 1e6, 1)}M. At first attainment the learner had observed {observed_range([m.get('total_eligible_callbacks') for m in observations])} eligible callbacks and completed {observed_range([m.get('total_completed_updates') for m in observations])} updates.")
                    else:
                        text.append(f"For h{h} {description} (completed seeds {seed_list}), the predeclared milestone ``{what}'' was NOT REACHED.")
                text.append(f"Over all 25M measured instructions, these runs saved {observed_range([r.get('cycles_saved_vs_stride') for r in group], 1e6, 3)} million cycles versus Stride. Their useful-prefetch coverage was {observed_range([r.get('useful_prefetch_coverage') for r in group], .01, 2)}\\%, selected accuracy {observed_range([r.get('selected_accuracy') for r in group], .01, 2)}\\%, and request pressure {observed_range([r.get('request_pressure') for r in group], 1, 3)} requests per L2 LOAD. These dimensions are reported separately; close IPC does not imply equal action behavior.")
                stride_ref = next((r for r in primary if r["arm"] == "stride"), None)
                frozen_ref = next((r for r in primary if r["arm"] == "frozen" and str(r["hidden_size"]) == str(h) and r["checkpoint"] == "i1m"), None)
                if stride_ref and number(stride_ref.get("request_pressure")):
                    pressures = [number(r["request_pressure"])/number(stride_ref["request_pressure"]) for r in group if number(r.get("request_pressure")) is not None]
                    saved_percent = [number(r["cycles_saved_vs_stride"])/number(stride_ref["cycles"]) for r in group if number(r.get("cycles_saved_vs_stride")) is not None]
                    text.append(f"This is {observed_range(saved_percent, .01, 2)}\\% fewer cycles with {observed_range(pressures, 1, 2)} times Stride's requested traffic. The matched Stride coverage/selected accuracy are {fmt(stride_ref.get('useful_prefetch_coverage'), 2, percent=True)}/{fmt(stride_ref.get('selected_accuracy'), 2, percent=True)}. The measured trade-off is useful misses covered versus request pressure; lower precision alone is not a cache-pollution measurement.")
                if frozen_ref and number(frozen_ref.get("ipc")):
                    fractions = [number(r["ipc"])/number(frozen_ref["ipc"]) for r in group if number(r.get("ipc")) is not None]
                    text.append(f"Their end-to-end IPC is {observed_range(fractions, .01, 3)}\\% of the matched same-H frozen-i1m IPC. This end-to-end ratio includes startup and is separate from the three-window attainment milestone.")
                if h == 16:
                    final_windows = []
                    for run in group:
                        series = [w for w in winrows if w["run_id"] == run["run_id"]]
                        if series:
                            final_windows.append(series[-1])
                    if final_windows:
                        text.append(f"The h16 {description} result is not permanent frozen-model attainment: in the final 1M-instruction window, IPC was {observed_range([w.get('ipc_fraction_of_frozen_i1m') for w in final_windows], .01, 2)}\\% of same-H frozen-i1m, coverage was {observed_range([w.get('useful_prefetch_coverage') for w in final_windows], .01, 2)}\\%, and request pressure had fallen to {observed_range([w.get('request_pressure') for w in final_windows], 1, 3)}. The curves show late regression after the initial qualifying three-window streak. With unbounded state and zero modeled service delay, this is not a finite-table or engine-congestion loss. It accompanies a changing learned action policy while optimization continues; the imitation objective does not optimize cache usefulness or IPC. The completed measurements are retained without tuning the recipe to remove this unsuccessful aspect of the capacity check.")
    else:
        text.append(r"\textbf{No completed primary online measurement is available. No conclusion about learning usefulness is established by this report.}")
    finite_answer = [r for r in final if r["protocol"] == "cold" and str(r.get("state_capacity")) == "64"]
    for case in (4, 16):
        group = [r for r in finite_answer if str(r.get("service_case")) == str(case)]
        if not group:
            continue
        completed_fractions = [number(r["completed_decisions"])/number(r["eligible_callbacks"]) for r in group if number(r.get("eligible_callbacks"))]
        winners = sum((number(r.get("cycles_saved_vs_stride")) or 0) > 0 for r in group)
        scratch = next((r for r in group if r["arm"] == "scratch"), None)
        text.append(f"With the same 64-entry table and the predeclared {case}-MAC/cycle service model, {winners}/{len(group)} completed frozen/scratch arms beat Stride end to end. Their IPC was {observed_range([r.get('ipc') for r in group], 1, 6)}, useful coverage {observed_range([r.get('useful_prefetch_coverage') for r in group], .01, 3)}\\%, and only {observed_range(completed_fractions, .01, 2)}\\% of eligible NN decisions completed. " + (f"The scratch learner completed {fmt(scratch.get('completed_updates'), 0)} updates, with {fmt(scratch.get('input_drops'), 0)} inference input drops and {fmt(scratch.get('training_drops'), 0)} training admission drops. " if scratch else "") + "The algorithmic zero-cost usefulness therefore does not establish a timing-costed hardware win. Frozen and online service measurements use identical queue/table limits; the conventional Stride comparator retains its original callback-time requests.")
    if len(primary) < 14:
        text.append(r"\textbf{The requested 14-run functional suite is incomplete.} Available rows are reported individually; missing runs are not treated as successful or interpolated.")
    text += [r"\section{From offline imitation to online updates}", r"The summer experiment trained the compact PC-keyed LSTM from Stride supervision and replayed its direct signed-delta actions. The next experiment moved the same frozen model into C++ live ChampSim inference: weights remained fixed while recurrent state changed. The present experiment adds progressive weight updates from labels of already observed events. Frozen inference, random-start learning, and i1m warm-start adaptation are different conditions. The i1m warm start includes offline training and is never learning from zero observations.", r"PC and aligned cache-line address are the only neural prediction features. The learned hurdle determines whether to emit, the learned positive count determines $K$, and the GRU decoder generates direct signed deltas with free-running feedback. Teacher actions supervise loss only. The shadow Stride does not issue prefetches in an NN arm. Its current-line first action is preserved, rather than changing Stride to a different textbook policy."]
    text += [r"\subsection{Causality and recipe}", r"Each eligible demand is predicted with available recurrent state and a coherent weight version before its label may enter training. Batches are chronological; same-PC grouping is limited to already observed events inside a batch, using saved detached starting states. State lifetimes distinguish eviction and reallocation. Publication carries detached h/c forward. A fresh optimizer belongs to each independent run. The teacher and outcomes never enter the feature encoder."]
    text.append(r"The locked recipe takes one Adam step at learning rate 0.002 per 64 newly admitted chronological decisions, one pass per batch, with one training-library thread. The authoritative hurdle/count/direct-delta loss and free-running decoder recurrence are reused. Hurdle weights remain $(1,1)$ until both classes have appeared in training-available labels; thereafter they are $(N/(2N_-),N/(2N_+))$ using only cumulative received labels, including the current completed batch. Labels dropped before training admission do not supply balancing statistics. Empty, single-class and short batches are valid; the final incomplete pending batch is not flushed after the simulation boundary. TBPTT spans the actual same-lifetime subsequence inside a batch, at most 64 steps, not 256 same-PC steps or 64 steps for every PC.")
    text += [r"\subsection{Software validation}", r"Correctness fixtures cover frozen Python/C++ parity, future-prefix independence, real gradients and parameter changes, single-class/empty/short batches, state eviction lifetimes, delayed completion without another demand callback, coherent publication, occupancy accounting, and timely/late/unused/unresolved lifecycle outcomes. These checks validate software behavior and are not research measurements. A saved-activation measurement hook was corrected to retain detached aliases instead of gradient-history cycles; a controlled 500-update check produced byte-identical final weights. Final online measurements were rerun with the corrected hook and unchanged learning recipe."]
    if recipe:
        command_fields = {"source", "simulator", "simulator_base", "trace", "checkpoints", "python", "raw"}
        simple = [(tex(k), tex(v)) for k, v in recipe.items() if k not in command_fields and isinstance(v, (str, int, float, bool))]
        if simple:
            text.append(table(["Locked recipe field", "Value"], simple, "p{0.40\\linewidth}p{0.50\\linewidth}", "small"))
        for key in ("causal_loop", "training_schedule", "service_assumptions", "storage_assumptions", "limitations"):
            if recipe.get(key):
                text.append(tex(recipe[key]))
    else:
        text.append(r"The exact execution recipe is recorded in the experiment's \texttt{design.md}, run configuration, and per-run raw counters. This generated report does not infer unspecified resource costs.")
    text += [r"\subsection{Two starting conditions}", r"Historical compatibility uses 25M no-prefetch cache warmup plus 25M measurement and empty parity predictor state at the boundary. Its NN service cost is zero and state routing unbounded. The conventional live Stride reference from earlier work trained and issued during warmup; its lifecycle counts have a different scope and cannot replace the matched comparator.", r"The primary protocol is \emph{cold start at the held-out trace-slice boundary}, corresponding to the original 25M--50M instruction region. Skipped records are not simulated and do not warm caches. Frozen arms load only trained weights; caches, recurrent histories, teacher, and online optimizer otherwise begin fresh. This is not program startup. All cold arms execute the same selected target instruction records; callback order and timing may change with the closed-loop prefetcher.", r"Retired target instructions from the slice origin are the horizontal axis. Eligible L2 callbacks, admitted/completed decisions, available labels/actions, updates, training exposures, and published versions are separate counters. Read-ahead trace records do not count as predictor observations. The original retirement-group warmup actually ends after 25,000,004 retired instructions, and compatibility measurement retires 25,000,003 instructions. These small overshoots are preserved and recorded separately from the nominal 25M protocol. Cold runs skip exactly 25,000,000 records and retire exactly 25,000,000 selected instructions."]
    text += [r"\section{End-to-end measurements}", r"IPC and cycles include startup. Prefetch quality below uses legacy counter meanings and matched no-prefetch denominators. Raw counts are retained in the committed CSV tables. NA denotes an undefined ratio or missing observation."]
    if any(r.get("end_no_pref_reference") == "historical same-protocol no-prefetch end counters only" for r in final):
        text.append(r"Historical compatibility coverage and miss reduction reuse the completed original same-protocol no-prefetch \emph{end counters}: 25,000,003 measured instructions and 203,131 L2 LOAD misses. This is a reference, not another new run. It is never used for cold-start comparisons or dynamic windows. The earlier conventional live Stride warmup reference is not reused for cycle-saving comparisons.")
    if final:
        run_config = {r["identity"]["run_id"]: r["config"] for r in data.get("runs", [])}
        configuration_rows = []
        for run in final:
            cfg = run_config.get(run["run_id"], {})
            entries = "NA" if run["arm"] == "no_pref" else ("64 Stride" if run["arm"] == "stride" else ("unbounded" if not number(run.get("state_capacity")) else str(run["state_capacity"])))
            initialization = run.get("checkpoint") or ("random" if run["arm"] == "scratch" else "NA")
            configuration_rows.append([tex(label(run)), tex(run["protocol"]), fmt(run.get("observed_trace_origin_records"), 0), fmt(number(cfg.get("warmup_instructions"))/1e6 if number(cfg.get("warmup_instructions")) is not None else None, 0), tex(initialization), tex(run.get("seed", "NA")), tex(entries), tex(run.get("service_case"))])
        text.append(table(["Run", "Protocol", "Origin records", "Warm M", "Weights", "Seed", "State", "MACs/cycle"], configuration_rows, "llrrlllr", "scriptsize", long=True))
    if final:
        text.append(table(["Run", "IPC", "Cycles (M)", "Saved vs Stride (M)", "L2 MPKI"], [[tex(label(r) + (" historical" if r["protocol"] == "historical" else "")), fmt(r.get("ipc"), 5), fmt(number(r.get("cycles")) / 1e6 if number(r.get("cycles")) is not None else None), fmt(number(r.get("cycles_saved_vs_stride")) / 1e6 if number(r.get("cycles_saved_vs_stride")) is not None else None), fmt(r.get("l2_mpki"))] for r in final], "lrrrr", long=True))
        text.append(table(["Run", "Coverage", "Miss reduction", "Raw accuracy", "Selected accuracy", "Timeliness", "$P/N_L$"], [[tex(label(r)), fmt(r.get("useful_prefetch_coverage")), fmt(r.get("miss_reduction")), fmt(r.get("raw_accuracy")), fmt(r.get("selected_accuracy")), fmt(r.get("legacy_timeliness")), fmt(r.get("request_pressure"))] for r in final], "lrrrrrr", "scriptsize", long=True))
    resources = rows(results_dir / "resources.csv")
    resource_by_run = {r["run_id"]: r for r in resources}
    if online:
        text.append(r"\subsection{What the online arms actually observed and processed}")
        text.append(table(["Run", "Eligible", "NN admitted", "NN completed", "Labels", "Positive labels"], [[tex(label(r))] + [fmt(r.get(k), 0) for k in ("eligible_callbacks", "admitted_decisions", "completed_decisions", "supervised_decisions", "available_positive_decisions")] for r in online], "lrrrrr", "scriptsize", long=True))
        text.append(table(["Run", "Eligible", "Labels", "Positive actions", "Updates", "Exposures", "Versions"], [[tex(label(r))] + [fmt(r.get(k), 0) for k in ("eligible_callbacks", "supervised_decisions", "positive_actions", "completed_updates", "training_sample_exposures", "published_versions")] for r in online], "lrrrrrr", "scriptsize", long=True))
        for run in online:
            r = resource_by_run.get(run["run_id"], {})
            series = [w for w in winrows if w["run_id"] == run["run_id"]]
            last_window = series[-1] if series else {}
            text.append(tex(label(run)) + ": " + f"{fmt(run.get('cycles_saved_vs_stride'), 0)} cumulative cycles saved; final full-window saving {fmt(last_window.get('window_cycles_saved_vs_stride'), 0)} cycles. " + f"The maximum observed same-lifetime TBPTT span was {fmt(r.get('max_tbptt_span'), 0)} events, with {fmt(r.get('state_evictions'), 0)} state evictions, {fmt(r.get('input_drops'), 0)} inference input drops and {fmt(r.get('output_drops'), 0)} output drops. " + f"Available supervision contained {fmt(run.get('positive_actions'), 0)} action atoms, while {fmt(run.get('training_sample_exposures'), 0)} decisions were exposed to completed updates.")
            text.append(f"The largest observed padded projection batch had {fmt(r.get('peak_padded_projection_positions'), 0)} positions (actual same-lifetime span {fmt(r.get('peak_effective_same_lifetime_tbptt_span'), 0)}, versus a configured global batch of 64). Final completed-batch loss was {fmt(r.get('last_loss'), 5)} and gradient L2 norm {fmt(r.get('last_gradient_l2'), 5)}. These are optimization diagnostics on received training labels, not useful coverage or a quality reward. " + f"Final causal hurdle weights were {tex(r.get('last_gate_class_weights', 'NA'))}.")
    text += [r"\subsection{Preserved metric definitions}", r"With $M_0$ matched no-prefetch L2 LOAD misses, $M$ method misses, $N_L$ L2 LOADs, $P$ requests, $I_{pf}$ issued requests, $Q$ PQ merges, $U$ useful prefetches, and $L$ late prefetches:", r"\[\mathrm{coverage}=U/M_0,\quad\mathrm{miss\ reduction}=(M_0-M)/M_0,\quad\mathrm{raw\ accuracy}=U/I_{pf},\]", r"\[\mathrm{selected\ accuracy}=U/(I_{pf}-Q),\quad\mathrm{legacy\ timeliness}=U/(U+L),\quad\mathrm{pressure}=P/N_L.\]", r"The original cache's useful-bit accounting can include a first RFO demand, while $N_L$ and $M$ retain their LOAD scope. Original printed LOAD misses increment at real fills; accepted lookup/callback miss counts are additional fields, not substitutes for $M$. Merged demands can therefore make callback and printed-fill counts differ. These legacy definitions are preserved. Neither $1-\mathrm{accuracy}$ nor unused eviction is called direct cache pollution. Counter deltas in windows measure activity within those windows; an issue and its later usefulness may cross a window boundary, so a window accuracy can be undefined even when a useful event occurred."]
    for h in (8, 16):
        selected = [r for r in primary if r["arm"] in ("stride", "no_pref") or str(r.get("hidden_size")) == str(h)]
        if any(str(r.get("hidden_size")) == str(h) and r["arm"] == "scratch" for r in selected):
            text.append(figure("Early learning", plot(selected, early_files, "ipc", "IPC per 100k retired instructions", f"h{h}: early cold-start learning", xmax=4), "Early measurements use 100k-instruction intervals. They describe the startup trajectory; the predeclared learning milestones continue to use three consecutive 1M windows."))
            text.append(figure("Early cumulative saving", plot(selected, early_files, "cumulative_cycles_saved_vs_stride", "Cumulative cycles saved vs Stride", f"h{h}: recovery of startup cost", xmax=1), "The first million instructions retains the initially negative region. A single positive sampled endpoint is descriptive and does not replace the three-window stability rule."))
            scratch_ids = {r["run_id"] for r in selected if r["arm"] == "scratch"}
            early_summary = []
            for boundary in (100_000, 500_000, 1_000_000):
                points = [r for r in early_rows if r["run_id"] in scratch_ids and number(r.get("end_instructions")) == boundary]
                if points:
                    early_summary.append(f"at {boundary/1e6:g}M: {observed_range([r.get('cumulative_cycles_saved_vs_stride') for r in points])} saved cycles, {observed_range([r.get('total_eligible_callbacks') for r in points])} callbacks, and {observed_range([r.get('total_completed_updates') for r in points])} completed updates")
            if early_summary:
                text.append(f"The observed h{h} scratch startup samples were " + "; ".join(early_summary) + ". These sampled values do not identify an exact event-level crossing time.")
        for metric, y, title in (("ipc", "Window IPC", f"h{h}: online learning and frozen references"), ("useful_prefetch_coverage", "Useful-prefetch coverage", f"h{h}: coverage in 1M instruction windows"), ("selected_accuracy", "Selected accuracy", f"h{h}: selected accuracy in 1M instruction windows"), ("legacy_timeliness", "Legacy timeliness", f"h{h}: legacy timeliness in 1M instruction windows"), ("request_pressure", "Requests per L2 LOAD", f"h{h}: request pressure in 1M instruction windows")):
            text.append(figure(title, plot(selected, window_files, metric, y, title), "Cold-start functional measurements. Each point is a 1M-retired-instruction window. Scratch seeds 7, 17, and 27 are separate runs; pretrained checkpoints have only seed 7. Missing or undefined ordinates are not interpolated."))
        text.append(figure("Cumulative saving", plot(selected, window_files, "cumulative_cycles_saved_vs_stride", "Cumulative cycles saved vs Stride", f"h{h}: retaining startup cost"), "Positive cumulative saving means the run has recovered its startup cost relative to the matched cold-start Stride run. A good late window alone does not establish an end-to-end improvement."))
    text += [r"\clearpage\section{Observed learning milestones}", r"The descriptive stability rule is three consecutive 1M-retired-instruction endpoints with positive cumulative cycle savings versus Stride. The separate frozen target requires window IPC at least 99\% of same-H frozen-i1m IPC in three consecutive 1M windows. Tables give the first qualifying window endpoint and the confirmation endpoint. This rule is not an online controller or a statistical equivalence test. Positive window streaks are retained separately in CSV; they do not imply startup has been recovered."]
    milestones = [m for m in data["learning_milestones"] if m["run_id"] in {r["run_id"] for r in online} and m["milestone"] != "positive_window_cycle_saving_vs_stride"]
    if milestones:
        text.append(table(["Run", "Milestone", "First M", "Confirm M", "Updates first/conf.", "Callbacks first/conf."], [[tex(label(m)), "Saved cycles" if m["milestone"].startswith("positive_cumulative") else "99\\% frozen", fmt(number(m.get("first_window_end")) / 1e6 if number(m.get("first_window_end")) is not None else None, 1) if m["status"] == "REACHED" else "NOT REACHED", fmt(number(m.get("confirmed_at_instructions")) / 1e6 if number(m.get("confirmed_at_instructions")) is not None else None, 1), fmt(m.get("total_completed_updates"), 0) + "/" + fmt(m.get("confirmation_total_completed_updates"), 0), fmt(m.get("total_eligible_callbacks"), 0) + "/" + fmt(m.get("confirmation_total_eligible_callbacks"), 0)] for m in milestones], "llrrrr", "scriptsize", long=True))
        text.append(table(["Run", "Milestone", "Coverage", "Selected accuracy", "Timeliness", "$P/N_L$"], [[tex(label(m)), "Saved cycles" if m["milestone"].startswith("positive_cumulative") else "99\\% frozen"] + [fmt(m.get(k), 4) for k in ("useful_prefetch_coverage", "selected_accuracy", "legacy_timeliness", "request_pressure")] for m in milestones], "llrrrr", "scriptsize", long=True))
        text.append(r"The corresponding coverage, selected accuracy, legacy timeliness, request pressure, supervised decisions, exposures, and published versions are in \texttt{learning\_milestones.csv}; they remain separate quantities rather than a composite score.")
    text += [r"\section{Cache occupancy, turnover, and lifecycle outcomes}", r"Occupancy counts simultaneously valid L2 lines. A valid-to-valid replacement changes the resident line without increasing fullness. Demand LOAD, RFO, prefetch, and writeback fills are separate, alongside replacement count and useful-line lifetimes. Fullness is not proof of a warmed working set or a deadline after which prefetching is useless."]
    geometries = {(r.get("l2_sets"), r.get("l2_ways"), r.get("line_bytes"), r.get("capacity_lines"), r.get("capacity_bytes")) for r in resources if number(r.get("capacity_bytes"))}
    for sets, ways, linebytes, capacity, bytecapacity in sorted(geometries):
        text.append(f"The active L2 geometry is {fmt(sets, 0)} sets $\\times$ {fmt(ways, 0)} ways $\\times$ {fmt(linebytes, 0)} bytes per line: {fmt(capacity, 0)} lines, {fmt(bytecapacity, 0)} bytes ({fmt(number(bytecapacity)/1024, 0)} KiB).")
    selected = [r for r in primary if r["arm"] in ("no_pref", "stride") or (str(r.get("hidden_size")) == "8" and str(r.get("seed")) == "7" and r.get("checkpoint") != "i20m")]
    text.append(figure("Occupancy", plot(selected, occupancy_files, "occupancy", "Simultaneously valid L2 lines (%)", "L2 filling during early learning", xmax=1), "The trace-slice origin is 25M target instructions; this plot expands its first million instructions. Exact fill-event first-attainment records, rather than cumulative fill counts, determine the occupancy milestones."))
    text.append(figure("Turnover", plot(selected, occupancy_files, "replacement_turnovers", "Replacements / L2 capacity lines", "L2 turnover alongside learning"), "Turnover is cumulative valid-to-valid replacements normalized by actual L2 line capacity. It is distinct from simultaneous occupancy and from useful-prefetch coverage."))
    text.append(figure("Censored lifecycle outcomes", plot(selected, occupancy_files, "censored_requests", "Pending or resident-unused prefetches", "Unresolved outcomes at each observation boundary"), "These outcomes remain censored at the displayed boundary. Their later outcomes do not retroactively redefine a window's observed state; final issue-cohort data are provided separately."))
    occ = [o for o in data["occupancy_milestones"] if o["run_id"] in {r["run_id"] for r in selected}]
    if occ:
        text.append(table(["Run", "Occupancy", "Instructions", "Cycles"], [[tex(label(o)), str(o["occupancy_percent"]) + r"\%", fmt(o.get("instructions"), 0) if o["status"] == "REACHED" else "NOT REACHED", fmt(o.get("cycles"), 0)] for o in occ], "lrrr", long=True))
    text.append(r"The supplementary \texttt{issue\_cohorts.csv} tracks actual unique PQ enqueues by issue-instruction cohort. Terminal outcomes are timely first demand, late first demand, unused eviction, redundant cache hit, or redundant in-flight request. Pending and resident-unused requests remain censored. These exclusive cohort outcomes are separate from legacy issued/merged counters and are not relabeled as the legacy accuracy or timeliness metrics.")
    if selected:
        text.append(table(["Run", "LOAD fills", "RFO fills", "PF fills", "WB fills", "Replacements", "Useful lifetimes"], [[tex(label(r))] + [fmt(resource_by_run.get(r["run_id"], {}).get(k), 0) for k in ("fills_load", "fills_rfo", "fills_prefetch", "fills_writeback", "replacements", "useful_line_lifetimes")] for r in selected], "lrrrrrr", "scriptsize", long=True))
        lifetime_rows = []
        for r in selected:
            resource = resource_by_run.get(r["run_id"], {})
            averages = []
            for total, count in (("line_lifetime_cycles", "line_lifetimes"), ("useful_line_lifetime_cycles", "useful_line_lifetimes"), ("prefetch_fill_to_first_demand_cycles", "prefetch_useful_lifetimes")):
                n, d = number(resource.get(total)), number(resource.get(count))
                averages.append(fmt(n/d if n is not None and d else None, 1))
            lifetime_rows.append([tex(label(r))] + averages)
        text.append(table(["Run", "Mean closed line lifetime", "Mean useful closed lifetime", "Mean PF fill to first use"], lifetime_rows, "lrrr", "scriptsize", long=True))
        text.append(r"Lifetime means are simulated cycles, not fill counts. Closed resident lifetimes end on eviction or invalidation; the useful subset served a demand or RFO at fill or during residence. Prefetch fill-to-first-use measures the first demand on a prefetched line. Still-resident line lifetimes are censored rather than assigned an end time; historical lines whose birth predates measurement have no fabricated start time.")
    cache_at_learning = []
    for milestone in milestones:
        if milestone["status"] != "REACHED":
            continue
        for boundary, kind in ((milestone["first_window_end"], "first"), (milestone["confirmed_at_instructions"], "confirm")):
            point = next((r for r in sample_rows if r["run_id"] == milestone["run_id"] and number(r.get("instructions")) == boundary), None)
            if point:
                capacity = number(point.get("capacity_lines"))
                cache_at_learning.append([tex(label(milestone)), ("Saving " if milestone["milestone"].startswith("positive_cumulative") else "99\\% frozen ") + kind, fmt(boundary/1e6, 1), fmt(number(point.get("occupancy_lines"))/capacity if capacity else None, 1, percent=True), fmt(point.get("replacements"), 0), fmt(number(point.get("replacements"))/capacity if capacity else None, 2), fmt(point.get("completed_updates"), 0)])
    if cache_at_learning:
        text.append(table(["Run", "Learning endpoint", "M instr.", "Occupancy", "Replacements", "Turnovers", "Updates"], cache_at_learning, "llrrrrr", "scriptsize", long=True))
        text.append(r"The table places cache turnover beside each first and confirmed learning endpoint. A full cache continues replacing lines while learning and useful prefetching continue. Occupancy is neither a working-set warmness test nor a learning deadline.")
    text += [r"\clearpage\section{Storage and timing sensitivity}", r"Unbounded state and zero modeled latency are algorithmic controls. Finite arms use the same exact-PC table and queue limits for frozen and online inference. Equal entry counts do not imply equal byte cost. FP32 weights are 7,632 bytes (7.453 KiB) for h8 and 20,880 bytes (20.391 KiB) for h16. A bare h/c pair requires $8H$ bytes per entry before tags and replacement metadata; a 64-entry table therefore uses 4,096 or 8,192 h/c bytes. No quantized-model measurement is claimed.", r"The finite-resource sensitivity cases are architectural assumptions, not measured silicon latency, area, power, frequency, or PPA. Inference work depends on the actual model and decoded $K$. Training, active/training weight copies, optimizer moments, publication/copy work, queues, and the shadow teacher must be counted. Host CPU time is not converted into simulated stall cycles. Simulator/CPU co-simulation is not completed on-chip training hardware."]
    text += [r"\subsection{Predeclared architectural cost assumptions}", r"There are separate serial inference and training engines. The two finite cases provide $b\in\{4,16\}$ MACs/cycle with matched weight bandwidth $4b\in\{16,64\}$ bytes/cycle. Each engine has one nonlinear unit at four cycles per operation. Dense, nonlinear, and control phases serialize conservatively. The matched weight-read bandwidth supplies one FP32 weight operand per MAC concurrently within the dense phase; it is not a second serialized full-weight-read charge. The additional bandwidth term below counts scratch/state traffic. The effective inference initiation interval equals its service occupancy because there is one inference in flight; the input queue has 16 entries, output queue 32 addresses, and output issue bandwidth one request/cycle. The learned positive count is not clamped to Stride's degree: the separate 32-address decoder resource bound drops excess output explicitly.", r"Let $E$ denote the learned positive hurdle and $K$ the actually decoded count after explicit resource limits. Inference work is", r"\[F_{\rm infer}=128H+8H^2+2H+EH+K(3H^2+4H),\qquad Z_{\rm infer}=6H+E+K(3H+1).\]", r"Its service time is $\lceil F_{\rm infer}/b\rceil+4Z_{\rm infer}+8+\lceil S/4\rceil+4K+\lceil T_{\rm infer}/(4b)\rceil$, where $S$ is live state entries and $T_{\rm infer}=4(128+4H)+24K$ bytes accounts for scratch/state/output traffic. This is a sensitivity model, not a measured pipeline.", r"For each 64-decision batch, $R$ is actual padded projection positions, $n_+$ the number of positive decisions, and $A$ the teacher action count. $P_\theta=11H^2+150H+4$ is parameter count. The actual forward MAC estimate and conservative backward/optimizer costs are", r"\[F_{\rm train}=128HR+64(8H^2+2H)+n_+H+A(3H^2+4H),\]", r"\[Z_{\rm train}=RH+320H+3AH+192+n_++2P_\theta.\]", r"Training costs $\lceil3F_{\rm train}/b\rceil+12Z_{\rm train}+512+12P_\theta+\lceil T_{\rm train}/(4b)\rceil$ cycles, with $T_{\rm train}=32P_\theta+4(128R+128H+AH)$ bytes. The backward MAC bound is twice forward, so total modeled forward/backward work is three times forward; it is not a profiler measurement of PyTorch kernels. Adam/state movement and parameter copying are counted separately. Publication costs $\lceil4P_\theta/(4b)\rceil+8$ cycles. One update may be in flight and 64 additional decisions pending; a full pending queue drops training admission explicitly.", r"The shadow teacher assumes a separate 24-stage, initiation-interval-one pipeline: four tag comparators in each of 16 stages (64 comparator lanes across the pipeline), then eight control stages, with distributed stage-local tag reads, ideal tracker-state forwarding and a 64-entry label queue. The active L2 accepts at most one read per cycle, so its actual eligible callbacks respect this input rate. A single unpipelined four-comparator unit would not provide this throughput. Teacher actions become training-available at modeled completion. Parameter publication is coherent, and h/c state becomes available on scheduled inference completion. Pending work is serviced at scheduled simulator cycles even without a new demand callback. The conventional Stride comparator retains its original callback-time requests with no added modeled lookup delay. The finite sensitivity adds service costs to the neural inference and online update engines; it does not retrofit a Stride timing model. The zero-cost controls set modeled service/publication delays to zero while retaining actual CPU co-simulation work."]
    storage = [s for s in data.get("storage", []) if s["run_id"] in {r["run_id"] for r in final}]
    unique_storage = {}
    for entry in storage:
        key = (entry.get("arm"), entry.get("hidden_size"), entry.get("state_capacity"), entry.get("service_case"), entry.get("category"))
        if key not in unique_storage:
            unique_storage[key] = dict(entry)
        else:
            for field in ("observed_peak_bytes", "max_bytes", "configured_max_bytes"):
                vals = [number(unique_storage[key].get(field)), number(entry.get(field))]
                unique_storage[key][field] = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
    if unique_storage:
        training_names = {"shadow Stride / conventional Stride state", "teacher label availability queue", "bounded training examples and saved start states", "padded training input and target tensors", "TBPTT autograd-saved activations", "gradient tensors", "Adam moments and step tensors", "training FP32 weight copy", "publication buffer", "retained old inference weight bank"}
        subtotal_rows, compact_totals = [], []
        subtotal_keys = set()
        for run in final:
            if run["arm"] not in ("frozen", "scratch") or (run["arm"] == "frozen" and run.get("checkpoint") != "i1m") or run["protocol"] != "cold" or str(run.get("seed")) != "7":
                continue
            key = (run["arm"], run["hidden_size"], run["state_capacity"], run["service_case"])
            if key in subtotal_keys:
                continue
            subtotal_keys.add(key)
            entries = [s for s in storage if s["run_id"] == run["run_id"]]
            groups = {name: [s for s in entries if (s["category"] in training_names) == is_training] for name, is_training in (("inference", False), ("training", True))}
            values = {}
            for name, group in groups.items():
                conf = [number(s.get("configured_max_bytes")) for s in group]
                obs = [number(s.get("observed_peak_bytes")) for s in group]
                values[name + "_configured_bytes"] = sum(conf) if all(v is not None for v in conf) else None
                values[name + "_peak_envelope_bytes"] = sum(obs) if all(v is not None for v in obs) else None
            compact_totals.append({**{k: run[k] for k in ("run_id", "arm", "hidden_size", "seed", "state_capacity", "service_case")}, **values})
            subtotal_rows.append([tex(label(run))] + [fmt(values[k]/1024 if values[k] is not None else None, 2) for k in ("inference_configured_bytes", "training_configured_bytes", "inference_peak_envelope_bytes", "training_peak_envelope_bytes")])
        write_csv(results_dir / "storage_totals.csv", compact_totals)
        text.append(table(["Design", "Inference max KiB", "Learning max KiB", "Inference peaks KiB", "Learning peaks KiB"], subtotal_rows, "lrrrr", "scriptsize", long=True))
        text.append(r"Inference includes active weights, PC state, arithmetic scratch, active event/result payloads, and input/output queues. The learning subtotal adds the shadow teacher and its queue, examples, padded tensors, TBPTT activations, gradients, optimizer, training weights, publication buffer, and retained version bank. The zero-service unbounded rows have no finite inference maximum; measured occupied-state bytes remain in their peak envelope. The 64-entry rows provide the finite configured total. These are FP32 logical storage costs, not host RSS.")
        text.append(r"The category table groups runs with the same arm, hidden size, capacity and service case; its observed column takes the maximum across completed seeds/checkpoint budgets within that design. The following per-run totals keep runs separate.")
        text.append(table(["Design", "Storage category", "Max (B)", "Peak (B)"], [[tex(storage_label(s)), tex(s.get("category", "")), fmt(s.get("configured_max_bytes", s.get("max_bytes")), 0), fmt(s.get("observed_peak_bytes"), 0)] for s in unique_storage.values()], "p{0.24\\linewidth}p{0.43\\linewidth}rr", "scriptsize", long=True))
        totals = []
        for run in final:
            entries = [s for s in storage if s["run_id"] == run["run_id"]]
            if not entries:
                continue
            configured = [number(s.get("configured_max_bytes")) for s in entries]
            observed = [number(s.get("observed_peak_bytes")) for s in entries]
            maximum = sum(configured) if all(v is not None for v in configured) else None
            peak_sum = sum(observed) if all(v is not None for v in observed) else None
            l2 = number(resource_by_run.get(run["run_id"], {}).get("capacity_bytes"))
            totals.append([tex(label(run)), fmt(maximum / 1024 if maximum is not None else None, 2), fmt(peak_sum / 1024 if peak_sum is not None else None, 2), fmt(maximum / l2 if maximum is not None and l2 else None, 2)])
        text.append(table(["Run", "Configured max KiB", "Peaks/reservations KiB", "Max / L2 bytes"], totals, "lrrr", "scriptsize", long=True))
        text.append(r"A sum of component peaks and fixed workspace reservations is a conservative envelope; their maxima need not occur simultaneously. Arithmetic scratch and active packet metadata are fixed reservations, not sampled allocator peaks. A configured total is NA if any component is unbounded or unreported. Per-category bytes/KiB and scope notes are retained in \texttt{storage.csv}.")
    else:
        text.append(r"\textbf{No category storage measurements were supplied to this report.} Missing categories are NA, not zero. The full required breakdown includes weights; PC tags/valid/replacement/h/c; inference scratch; queues; shadow teacher; examples and TBPTT activations; gradients; moments; training/active copies; and publication buffers.")
    text.append(r"Required predictor storage excludes measurement-only logging and host allocator/RSS overhead. Dedicated predictor SRAM is compared with the actual L2 capacity as a size ratio; it is not assumed to occupy L2 or directly pollute the cache. Configured bounds and observed peaks are reported separately.")
    hardware = [r for r in final if r["protocol"] == "cold" and str(r.get("state_capacity")) == "64"]
    if hardware:
        stride_reference = next((r for r in primary if r["arm"] == "stride"), None)
        text.append(figure("Finite sensitivity", finite_summary_plot(hardware, storage, stride_reference), "Measured h8 seed-7 sensitivity with the same 64-entry state table and bounded queues. Zero cost isolates finite capacity; 4 and 16 MACs/cycle add the predeclared inference, training and publication service model. Log storage axis compares configured logical FP32 storage, including the online trainer. Service costs are architectural assumptions, not measured silicon."))
        text.append(figure("Finite cumulative saving", plot(hardware + ([stride_reference] if stride_reference else []), window_files, "cumulative_cycles_saved_vs_stride", "Cumulative cycles saved vs Stride", "Finite resources: retaining startup and queue losses"), "Each finite arm uses its own live closed-loop callback order. Solid, dotted and dashed lines denote zero cost, 4 MACs/cycle and 16 MACs/cycle respectively; color distinguishes frozen and scratch. The matched conventional Stride reference is shared only within the cold-start protocol."))
        text.append(table(["Finite h8 design", "MACs/cycle", "IPC", "Cycles (M)", "Saved vs Stride (M)"], [[tex(label(r)), tex(r.get("service_case")), fmt(r.get("ipc"), 5), fmt(number(r.get("cycles")) / 1e6), fmt(number(r.get("cycles_saved_vs_stride")) / 1e6 if number(r.get("cycles_saved_vs_stride")) is not None else None)] for r in hardware], "lrrrr", long=True))
        text.append(table(["Finite design", "Eligible", "Completed NN", "Input drops", "Training drops", "Updates", "State evictions"], [[tex(label(r))] + [fmt(r.get(k), 0) for k in ("eligible_callbacks", "completed_decisions", "input_drops", "training_drops", "completed_updates", "state_evictions")] for r in hardware], "lrrrrrr", "scriptsize", long=True))
        finite_ids = {r["run_id"] for r in hardware if r["arm"] == "scratch"}
        finite_milestones = [m for m in data["learning_milestones"] if m["run_id"] in finite_ids and m["milestone"] != "positive_window_cycle_saving_vs_stride"]
        text.append(table(["Finite online design", "Milestone", "First M", "Confirm M", "Updates at first"], [[tex(label(m)), "Saved cycles" if m["milestone"].startswith("positive_cumulative") else "99\\% finite frozen", fmt(number(m.get("first_window_end"))/1e6 if number(m.get("first_window_end")) is not None else None, 1) if m["status"] == "REACHED" else "NOT REACHED", fmt(number(m.get("confirmed_at_instructions"))/1e6 if number(m.get("confirmed_at_instructions")) is not None else None, 1), fmt(m.get("total_completed_updates"), 0)] for m in finite_milestones], "llrrr", "scriptsize", long=True))
        for r in hardware:
            rr = resource_by_run.get(r["run_id"], {})
            functional_reference = next((p for p in primary if p["arm"] == r["arm"] and str(p.get("hidden_size")) == str(r.get("hidden_size")) and str(p.get("seed")) == str(r.get("seed")) and (r["arm"] != "frozen" or p.get("checkpoint") == r.get("checkpoint"))), None)
            zero_reference = next((p for p in hardware if p["arm"] == r["arm"] and str(p.get("service_case")) == "0"), None)
            comparison = functional_reference if str(r.get("service_case")) == "0" else zero_reference
            comparison_label = "unbounded zero-cost control" if str(r.get("service_case")) == "0" else "64-entry zero-cost control"
            if comparison:
                text.append(tex(label(r)) + f": IPC was {fmt(number(r.get('ipc'))/number(comparison.get('ipc')), 2, percent=True)} of its {comparison_label}, with {fmt(number(r.get('cycles'))-number(comparison.get('cycles')), 0)} additional end-to-end cycles. " + f"Completed inference covered {fmt(number(r.get('completed_decisions'))/number(r.get('eligible_callbacks')) if number(r.get('eligible_callbacks')) else None, 2, percent=True)} of eligible callbacks, with {fmt(r.get('input_drops'), 0)} input drops and {fmt(r.get('training_drops'), 0)} training drops. " + f"It finished with {fmt(rr.get('input_queue'), 0)} queued inputs, {fmt(rr.get('inference_inflight'), 0)} inference job, {fmt(rr.get('training_queue'), 0)} pending training decisions, and {fmt(rr.get('update_inflight'), 0)} update in flight. This separates capacity loss from modeled service congestion.")
        text.append(table(["Finite design", "MAC/infer.", "Nonlinear/infer.", "MAC/update", "Nonlinear/update", "Weight B/cycle"], [[tex(label(r))] + [fmt(resource_by_run.get(r["run_id"], {}).get(k), 1) for k in ("mean_inference_macs", "mean_inference_nonlinears", "mean_update_macs", "mean_update_nonlinears", "weight_bandwidth_bytes_per_cycle")] for r in hardware], "lrrrrr", "scriptsize", long=True))
        timed = [r for r in hardware if str(r.get("service_case")) in ("4", "16")]
        cohort_rows = rows(results_dir / "issue_cohorts.csv")
        redundant_fractions = []
        for run in timed:
            cohorts = [c for c in cohort_rows if c["run_id"] == run["run_id"]]
            enqueued = sum(number(c.get("enqueued")) or 0 for c in cohorts)
            redundant = sum(number(c.get("redundant_cache")) or 0 for c in cohorts)
            if enqueued:
                redundant_fractions.append(redundant/enqueued)
        if redundant_fractions:
            text.append(f"Across the completed timed arms, {observed_range(redundant_fractions, .01, 2)}\\% of actual unique PQ enqueues resolved as redundant cache hits: the target line was already resident when the request was serviced. These are not late prefetches under the legacy $L$ counter. Consequently legacy timeliness remained {observed_range([r.get('legacy_timeliness') for r in timed], .01, 2)}\\% while useful coverage was only {observed_range([r.get('useful_prefetch_coverage') for r in timed], .01, 2)}\\%. The ratio $U/(U+L)$ excludes redundant-cache outcomes; a high value cannot establish useful coverage or a timing-costed win. The issue-cohort outcomes expose this distinction without redefining legacy timeliness.")
        text.append(r"Per-inference and per-update averages divide work by started jobs. MAC and service-cycle counters charge a full operation at its start, including a final operation that may extend beyond the retirement boundary; these totals are not measured engine utilization. Simulator completed updates and sample exposures advance at modeled training completion, while publication occurs later. A worker may finish an additional host batch during shutdown, so worker diagnostics do not replace the simulator completion/version counters. Raw MAC, nonlinear, state-traffic and publication-cycle totals, queue peaks, drops, and unfinished jobs are in \texttt{resources.csv}.")
    host_runs = [r for r in final if r["arm"] in ("scratch", "warm")]
    if host_runs:
        text.append(r"\subsection{Measured host co-simulation costs}")
        text.append(table(["Run", "Simulator wall s", "Simulator RSS MiB", "Worker CPU s", "Worker RSS MiB"], [[tex(label(r)), fmt(r.get("simulator_wall_seconds"), 2), fmt(number(r.get("simulator_peak_rss_kib"))/1024 if number(r.get("simulator_peak_rss_kib")) is not None else None, 2), fmt(r.get("worker_host_cpu_seconds"), 2), fmt(number(r.get("worker_peak_rss_bytes"))/1048576 if number(r.get("worker_peak_rss_bytes")) is not None else None, 2)] for r in host_runs], "lrrrr", "scriptsize", long=True))
        text.append(r"RSS includes whole host processes, libraries, allocators and IPC and is not predictor SRAM. Simulator wall time includes waits for the persistent CPU worker. Neither host wall time nor CPU time becomes simulated CPU stalls; congestion is accounted through the declared simulated engines and memory requests.")
    text += [r"\clearpage\section{Historical measurements: context, not replacements}", r"The prior 1M plateau refers to the offline trace-start pretraining budget under a fixed 12-epoch Adam 0.002 recipe, with same-H comparisons to the i20m model. It does not mean an online learner became useful after one million observations. i1m supplied 16,172 eligible training decisions, 2,486 positive decisions and 4,875 action atoms; i20m supplied 325,389, 52,059 and 102,140. Offline updates numbered 228 and 4,404 respectively. Whole-trace PC grouping used there is not a causal online schedule when shared weights change.", r"Actual completed seed-7 live parity logs exist for all 28 exported prior points. From i1m through i20m, the maximum same-H relative IPC deviations were 0.21747\% for h8 and 0.07318\% for h16, meeting the earlier $\pm0.5\%$ descriptive plateau rule. Request pressure and act rates still varied by roughly 13--15\%; policy equivalence was not established. These warm-cache historical results are kept separate from new cold-start IPC."]
    if compatibility:
        ids = sorted({r["run_id"] for r in compatibility})
        exact = lambda value: value is True or value == "True"
        matched = sum(exact(r["exact_match"]) for r in compatibility)
        text.append(f"The new historical compatibility runs match {matched} of {len(compatibility)} original-parser fields exactly across {len(ids)} same-checkpoint comparisons. These comparisons include raw instruction/cycle/cache/prefetch counts and legacy derived ratios. Exact system-counter agreement does not by itself prove every internal floating-point value is identical; Python/C++ software parity is validated separately.")
        comparison_rows = []
        for name in ids:
            rs = [r for r in compatibility if r["run_id"] == name]
            indexed = {r["field"]: r for r in rs}
            comparison_rows.append([f"h{rs[0]['hidden_size']} {tex(rs[0]['checkpoint'])}", str(sum(exact(r["exact_match"]) for r in rs)) + "/" + str(len(rs)), fmt(indexed.get("instructions", {}).get("new_value"), 0), fmt(indexed.get("cycles", {}).get("new_value"), 0), fmt(indexed.get("cycles", {}).get("difference"), 0)])
        text.append(table(["New compatibility model", "Fields exact", "Instructions", "Cycles", "Cycle difference"], comparison_rows, "lrrrr", "scriptsize"))
    if historic:
        text.append(table(["Historical model", "Transport", "IPC", "Cycles", "Coverage", "Pressure"], [[f"h{h['hidden_size']} {tex(h['budget'])}", tex(h["mode"]), fmt(h.get("ipc"), 5), fmt(h.get("cycles"), 0), fmt(h.get("useful_prefetch_coverage")), fmt(h.get("request_pressure"))] for h in historic], "llrrrr", "scriptsize", long=True))
    else:
        text.append(r"Historical contextual counts above were established from the completed source run tree; no historical log table was supplied to this report invocation. New measurements never reuse those values as cold-start results.")
    text += [r"\section{Limitations and interpretation}", r"The experiment studies one held-out slice of one trace, two compact model capacities, three random-start seeds, and one pretrained seed. The recipe is fixed using the development gap before final evaluation. Progressive updates from already passed final events are allowed; later final measurements do not tune hyperparameters. Functional correctness with weak learning is a valid outcome.", r"Data scarcity is assessed using supervised observations and positive actions; optimization using actual updates/exposures and model behavior; state churn using evictions; traffic using requests, drops and cache outcomes; timing using queue/service/publication counters. A late-window improvement without cumulative recovery is reported as such. No composite score hides trade-offs, and absent milestones remain NOT REACHED.", r"The compact committed tables are \path{summary.csv}, \path{windows_1m.csv}, \path{windows_100k.csv}, \path{learning_milestones.csv}, \path{occupancy_milestones.csv}, \path{issue_cohorts.csv}, \path{storage.csv}, and figure inputs. Raw traces, checkpoints, logs, and training buffers remain outside tracked source. Build, test, pilot, run, status, resume, and report commands are in \texttt{command.md}.", r"\end{document}"]
    destination = out_dir / "602_stride_online_learning.tex"
    destination.write_text("\n\n".join(text) + "\n")
    print(f"Wrote measured report source {destination}; compile and inspect its rendered PDF before delivery.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    generate(args.results_dir, args.out_dir, args.historical_root, args.recipe)


if __name__ == "__main__":
    main()
