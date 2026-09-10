#!/usr/bin/env python3
"""Generate the scientific LaTeX report and small plotted inputs from runs.

This program never starts simulation or changes weights and never fills missing results
with historical bars. Run pdflatex on the emitted source and render for QA.
"""
import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path

from analyze import COUNTERS, PROGRESS, load_oracle, metrics, normalize, number, write_csv


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
    result = prefix + "frozen " + str(row.get("checkpoint", ""))
    if str(row.get("state_capacity")) == "64":
        result += f" /64 /MAC{row.get('service_case')}"
    return result


def functional(row):
    return row.get("protocol") == "cold" and row.get("phase") == "final" and str(row.get("state_capacity", 0)) in ("0", "unbounded", "None") and str(row.get("service_case", 0)) in ("0", "zero", "None")


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


COLORS = {"no_pref": "#777777", "stride": "#222222"}


def plot_style(run):
    if run["arm"] != "frozen":
        return COLORS[run["arm"]], ":" if run["arm"] == "no_pref" else "-"
    color = "#0072B2" if int(run["hidden_size"]) == 8 else "#D55E00"
    dash = "--" if run["checkpoint"] == "i20m" else "-"
    if int(run["state_capacity"]) == 64:
        color = {0: "#0072B2", 4: "#CC79A7", 16: "#009E73"}[int(run["service_case"])]
    return color, dash


def panel_plot(out_dir, name, selected, source_rows, panels, xkey="end_instructions", xmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 7.1), layout="constrained")
    inputs = []
    for run in selected:
        series = [r for r in source_rows if r["run_id"] == run["run_id"]]
        if xmax is not None:
            series = [r for r in series if number(r.get(xkey)) is not None and number(r[xkey]) <= xmax*1e6]
        inputs.extend(series)
        color, dash = plot_style(run)
        xs = [number(r[xkey])/1e6 for r in series]
        for ax, (key, title, scale) in zip(axes.flat, panels):
            ys = [number(r.get(key)) for r in series]
            ax.plot(xs, [y*scale if y is not None else float("nan") for y in ys], color=color, linestyle=dash, linewidth=1.3, label=label(run))
            ax.set(title=title, xlabel="Retired instructions from slice boundary (M)")
            ax.grid(alpha=.2);ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.xaxis.label.set_fontsize(8)
            if "saved" in key: ax.axhline(0,color="#555555",linewidth=.6)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=8,frameon=False)
    figure = "figures/"+name+".pdf"
    fig.savefig(out_dir/figure,bbox_inches="tight")
    fig.savefig(out_dir/figure.replace(".pdf",".png"),dpi=140,bbox_inches="tight")
    plt.close(fig)
    write_csv(out_dir/("figures/"+name+".csv"),inputs)
    return "\\begin{center}\\includegraphics[width=0.98\\linewidth]{"+figure+"}\\end{center}"


def generate(results_dir, out_dir, historical_root=None, recipe_path=None):
    data=json.loads((results_dir/"results.json").read_text())
    if any(r["arm"] not in ("no_pref","stride","frozen") for r in data["summary"]):
        raise ValueError("report accepts only actual frozen or baseline runs")
    final=[r for r in data["summary"] if r["status"]=="COMPLETE" and r["phase"]=="final"]
    primary=[r for r in final if functional(r)]
    cold=[r for r in final if r["protocol"]=="cold"]
    hardware=[r for r in cold if int(r["state_capacity"])==64]
    hist=[r for r in final if r["protocol"]=="historical"]
    historic=history(historical_root) if historical_root else rows(results_dir/"historical.csv")
    compatibility=historical_compatibility(historical_root,data) if historical_root else rows(results_dir/"historical_compatibility.csv")
    if historic:write_csv(results_dir/"historical.csv",historic)
    if compatibility:write_csv(results_dir/"historical_compatibility.csv",compatibility)
    out_dir.mkdir(parents=True,exist_ok=True);(out_dir/"figures").mkdir(exist_ok=True)
    wins=rows(results_dir/"windows_1m.csv");samples=rows(results_dir/"snapshots.csv")
    resources=rows(results_dir/"resources.csv");res={r["run_id"]:r for r in resources}
    storage=rows(results_dir/"storage.csv");totals=rows(results_dir/"storage_totals.csv")
    stride=next(r for r in primary if r["arm"]=="stride")
    h8=next(r for r in primary if r["arm"]=="frozen" and int(r["hidden_size"])==8 and r["checkpoint"]=="i1m")
    title=r"""\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,longtable,array,graphicx,amsmath}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Offline Training and Online Inference for a Tiny Stride Prefetcher}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{4pt}
\title{Offline Training and Online Inference\\\large A Tiny Neural Stride Prefetcher on 602.gcc\_s-734B}
\author{602 Stride dynamic inference experiment}
\date{Measured ChampSim execution; frozen deployment scope}
\begin{document}
\maketitle
\section{Question and measured answer}
How does an offline-trained tiny neural prefetcher behave during a causal, live cache simulation when its weights remain fixed, but its recurrent history and predictions evolve? What inference storage, queueing and service costs accompany its usefulness?
"""
    text=[title]
    text.append(f"This report reuses {len(final)} completed compatible runs: {len(primary)} cold functional runs, {len(hardware)} frozen h8 finite-resource runs, and {len(hist)} historical frozen compatibility runs. Original run identifiers and raw source paths are retained in the committed result tables. These are existing frozen measurements, not relabeled weight-changing runs and not new reruns of the full matrix.")
    text.append(f"With zero modeled service cost and unbounded PC routing, h8-i1m reaches IPC {fmt(h8['ipc'],6)} versus Stride's {fmt(stride['ipc'],6)}. It saves {fmt(h8['cycles_saved_vs_stride'],0)} cycles over the complete 25M-instruction cold slice, with useful-prefetch coverage {fmt(h8['useful_prefetch_coverage'],2,True)}, selected accuracy {fmt(h8['selected_accuracy'],2,True)}, legacy timeliness {fmt(h8['legacy_timeliness'],2,True)}, and {fmt(h8['request_pressure'],3)} requests per L2 LOAD.")
    for run in hardware:
        if int(run['service_case']):
            text.append(f"The preserved frozen {run['service_case']}-MAC/cycle case has IPC {fmt(run['ipc'],6)} and saves {fmt(run['cycles_saved_vs_stride'],0)} cycles versus Stride (negative means a loss). Only {fmt(number(run['completed_decisions'])/number(run['eligible_callbacks']),2,True)} of eligible callbacks complete inference.")
    text.append(r"\textbf{Scope.} The existing seed-7 i1m/i20m checkpoint is loaded once. Runtime weights remain fixed while h/c and predictions evolve. Offline training memory is excluded from inference deployment.")
    text.append(r"\section{Progression and evidence}")
    text.append(r"Summer work implemented compact PC-keyed LSTM imitation of conventional Stride, with an offline encoder, learned hurdle, positive-count head and signed-delta decoder. The frozen-live experiment moved the same forward model into ChampSim, preserving causal live callbacks. This branch reuses that implementation and preserves the summer code, offline/Colab workflow, checkpoints and frozen raw results.")
    text.append(r"The regression checks the authoritative Python/C++ frozen forward, changes in h/c and decoded predictions over observed accesses, unchanged parameter values, absence of a training process or update path, finite table eviction, delayed completion without a new demand callback, and the existing cache/metric fixtures. Software checks validate the refactor; the measured tables below retain their original frozen-run provenance.")
    text.append(r"A fresh paired Sacramento regression executed the 100,000-record development slice after skipping 20M records, using h8-i1m, 64 entries and the 4-MAC service case. The refactored binary and preserved frozen binary matched all 30 original parser fields: 325,790 cycles, IPC 0.306946, 1,633 L2 LOADs, 410 requested prefetches, 2 useful and 0 late. Six snapshot/cohort rows preserved cache and service counters, apart from explicit zero-field initialization and the documented event/queue layout reduction. Host elapsed time was 2.96 s versus 3.07 s; it is not a simulated timing gain. This is software regression evidence, not an additional final scientific row.")
    text.append(r"\clearpage\section{Causal inference and matched conditions}")
    text.append(r"Only the 64-bit PC and 64-bit aligned byte address enter the neural encoder. A 128-to-H projection and LSTM update h/c for the exact PC. The learned hurdle selects emission; the positive-count head determines K; a GRU decoder emits direct signed cache-line deltas with free-running predicted feedback. Conventional Stride is a separate baseline, retaining its current-line first requests [current, current + stride], tracker behavior and page restrictions. It is not a runtime neural teacher.")
    text.append(r"The serial inference engine consumes a bounded 16-event input FIFO. A decision uses the last completed h/c for its exact PC. State and outputs become visible at scheduled completion, so queued accesses cannot see a future recurrent state. A finite table has 64 exact-PC entries with LRU replacement; an unbounded control isolates capacity effects. Frozen weights need one active FP32 copy. There is no parameter publication operation.")
    text.append(r"Historical compatibility preserves the original 25M no-prefetch cache warmup plus 25M measurement, zero modeled neural cost, unbounded PC routing and empty predictor state at the measurement boundary. The actual warmup retirement group ends after 25,000,004 instructions, followed by 25,000,003 measured instructions. The old conventional live Stride run issued throughout warmup and is not used as its matched cycle-saving reference.")
    text.append(r"The primary protocol is \emph{cold start at the held-out trace-slice boundary}. It skips exactly 25,000,000 opaque records using the actual compiled reader types, without simulating them or warming caches. It executes exactly the next 25,000,000 selected instructions with fresh caches and recurrent histories, loading only pretrained weights. This is not a program-startup experiment. All methods use the same selected records and cache configuration; actual callback order/timing may differ in closed-loop live execution.")
    text.append(r"Retired target instructions from that boundary are the horizontal axis. Eligible L2 callbacks, admitted decisions and completed decisions are distinct from trace read-ahead and retirement. Snapshots at 0, 1k, 10k and every 100k retired instructions retain startup costs. Curves use exact one-million-instruction counter intervals; cumulative cycle saving always compares matched cold references.")
    text.append(r"\subsection{Assumed inference service}")
    text.append(r"The functional control has zero modeled cost. Sensitivity cases use 4 or 16 MACs/cycle with matched 16 or 64-byte/cycle weight and scratch bandwidth, one nonlinear unit taking four cycles per operation, and serialized dense/nonlinear/memory/control phases. Initiation interval equals data-dependent service time. For hidden size H and actual decoded count K, the forward work is")
    text.append(r"\[W_{MAC}=128H+8H^2+2H+eH+K(3H^2+4H),\quad W_{NL}=6H+e+K(3H+1),\]")
    text.append(r"where e is the learned emission decision. Control costs 8 + ceil(entries/4) + 4K cycles; additional scratch traffic is 4(128+4H)+24K bytes. The cold decoder resource permits 32 outputs, with excess and numerical failures counted. A 32-address output FIFO releases at most one request/cycle in finite cases through the original request path and duplicate/drop behavior. An unconditional per-core-cycle hook services pending work even when no demand callback occurs. Host execution time is never converted to CPU stall cycles. These are architectural sensitivity assumptions, not measured silicon timing, area, power or synthesized frequency.")
    text.append(r"\clearpage\section{Cold end-to-end measurements}")
    text.append(r"All rows execute 25,000,000 instructions and observe 404,271 L2 LOAD callbacks. H8/H16 i1m/i20m use the existing seed-7 pretrained weights; they do not represent independent pretraining seeds. State0 denotes the unbounded algorithmic control; state64 is finite LRU. Zero denotes zero modeled neural service cost.")
    text.append(table(["Method","IPC","Cycles","Saved vs Stride","L2 MPKI"],[[tex(label(r)),fmt(r['ipc'],6),fmt(r['cycles'],0),fmt(r.get('cycles_saved_vs_stride'),0),fmt(r['l2_mpki'],3)] for r in cold],"lrrrr","scriptsize"))
    text.append(table(["Method","Coverage","Miss reduction","Raw acc.","Selected acc.","Timeliness","P/NL"],[[tex(label(r))]+[fmt(r.get(k),4) for k in ('useful_prefetch_coverage','miss_reduction','raw_accuracy','selected_accuracy','legacy_timeliness','request_pressure')] for r in cold],"lrrrrrr","scriptsize"))
    text.append(r"\subsection{Raw counter meanings}")
    text.append(r"Let M0 be same-protocol no-prefetch L2 LOAD misses, M method misses, NL L2 LOADs, P requested prefetches, Ipf issued prefetches, Q PQ-merged requests, U useful and L late. The original parser and formulas are preserved:")
    text.append(r"\[\text{coverage}=U/M0,\quad\text{miss reduction}=(M0-M)/M0,\quad\text{raw accuracy}=U/Ipf,\]\[\text{selected accuracy}=U/(Ipf-Q),\quad\text{legacy timeliness}=U/(U+L),\quad\text{pressure}=P/NL.\]")
    text.append(r"Undefined ratios are NA. Legacy useful may include a first RFO; NL and M retain the original LOAD counter scope. Coverage is not imitation accuracy. Unused eviction and one minus accuracy are not direct cache pollution measurements.")
    text.append(table(["Method","M","P","Ipf","Q","U","L"],[[tex(label(r))]+[fmt(r.get(k),0) for k in ('l2_load_miss','pf_requested','pf_issued','pq_merged_duplicate_proxy','pf_useful','pf_late')] for r in cold],"lrrrrrr","scriptsize"))
    text.append(r"\clearpage\section{Recorded runtime observations}")
    text.append(r"The following h8-i1m zero-cost rows use already recorded cumulative snapshots at 100k, 1M and 25M retired instructions after the 25M trace-slice boundary. Coverage uses NoPF misses at the same recorded instruction count. They are descriptive sample points, not minimum-history requirements, optimized observation budgets or runtime triggers.")
    progress_rows=[]
    no_pref_id=next(r['run_id'] for r in primary if r['arm']=='no_pref')
    for count in (100_000,1_000_000,25_000_000):
        sample=normalize(next(r for r in samples if r['run_id']==h8['run_id'] and number(r['instructions'])==count))
        baseline=normalize(next(r for r in samples if r['run_id']==no_pref_id and number(r['instructions'])==count))
        progress_rows.append({**sample,**metrics(sample,baseline)})
    text.append(table(["Retired instructions","Eligible L2 callbacks","NN completed","Cumulative IPC","Valid L2 lines"],[[fmt(r['instructions'],0),fmt(r['eligible_callbacks'],0),fmt(r['completed_decisions'],0),fmt(r['ipc'],6),fmt(r['occupancy_lines'],0)+" / 4,096"] for r in progress_rows],"rrrrr","small"))
    text.append(table(["Retired instructions","Coverage","Raw accuracy","Selected accuracy","Timeliness","P/NL"],[[fmt(r['instructions'],0)]+[fmt(r[k],4) for k in ('useful_prefetch_coverage','raw_accuracy','selected_accuracy','legacy_timeliness','request_pressure')] for r in progress_rows],"rrrrrr","small"))
    text.append(r"The PC's h/c is carried continuously through successive completed inference events. Snapshot collection and the plotted counter intervals do not reset, truncate or otherwise change that history. No fixed-length history window W is introduced. Weights remain unchanged. The L2 callback count is the actual observation count; at the 1M snapshot it is 16,196, while the legacy completed LOAD counter is 16,195 because callback observation and cache completion have different timing.")
    text.append(r"\clearpage\section{Dynamic performance with fixed weights}")
    text.append(panel_plot(out_dir,'functional_progress',primary,wins,[("ipc","Window IPC",1),("cumulative_cycles_saved_vs_stride","Cumulative cycles saved vs Stride (M)",1e-6),("useful_prefetch_coverage","Window useful-prefetch coverage",1),("request_pressure","Window requested prefetches / L2 LOAD",1)]))
    text.append(r"These are live fixed-weight runs. Changing window behavior reflects incoming accesses, evolving h/c, cache state and realized timing; it does not indicate parameter learning. The cumulative curve retains all startup costs. The h8 and h16 i1m/i20m curves are separate checkpoints, with solid i1m and dashed i20m traces.")
    text.append(r"\clearpage\section{Quality and traffic over execution}")
    text.append(panel_plot(out_dir,'functional_quality',primary,wins,[("raw_accuracy","Window raw accuracy",1),("selected_accuracy","Window selected accuracy",1),("legacy_timeliness","Window legacy timeliness",1),("l2_mpki","Window L2 MPKI",1)]))
    text.append(r"A request can cross a window boundary before its first useful demand. These ratios use legacy counter activity within each interval, so a denominator can be zero even when a useful event occurs. NA remains missing, not zero or failure. Separate issue-cohort outcomes retain pending and resident-unused requests as censored. The committed issue-cohort table contains timely, late, unused, redundant and censored counts rather than replacing a legacy metric under the same name.")
    for row in samples:
        cap=number(row.get('capacity_lines')) or number(row.get('l2_capacity_lines'))
        row['occupancy_percent']=100*number(row['occupancy_lines'])/cap if cap else None
        row['replacement_turnovers']=number(row['replacements'])/cap if cap else None
    text.append(r"\clearpage\section{L2 occupancy and turnover}")
    text.append(panel_plot(out_dir,'cache_progress',[r for r in primary if r['arm']!='frozen' or (int(r['hidden_size'])==8 and r['checkpoint']=='i1m')],samples,[("occupancy_percent","Simultaneous valid-line occupancy (%)",1),("replacements","Valid-to-valid replacements",1),("fills_load","Demand LOAD fills",1),("fills_prefetch","Prefetch fills",1)],xkey='instructions',xmax=1))
    text.append(r"The active L2 has 512 sets, 8 ways and 64-byte lines: 4,096 lines or 256 KiB. Occupancy counts simultaneous valid lines, verified against a scan at samples. Invalid-to-valid fill raises occupancy; valid-to-valid replacement does not. The first million instructions show filling alongside ongoing turnover, not a deadline for prefetch usefulness. Fullness does not establish a warmed working set.")
    text.append(r"\clearpage\subsection{Exact occupancy attainment and fill outcomes}")
    occ=data['occupancy_milestones']
    text.append(table(["Method",r"50\% ins.",r"90\% ins.",r"95\% ins.",r"99\% ins.",r"100\% ins.",r"100\% cycles"],[[tex(label(r))]+[fmt(next((o['instructions'] for o in occ if o['run_id']==r['run_id'] and o['occupancy_percent']==pct),None),0) for pct in (50,90,95,99,100)]+[fmt(next((o['cycles'] for o in occ if o['run_id']==r['run_id'] and o['occupancy_percent']==100),None),0)] for r in cold],"lrrrrrr","scriptsize"))
    text.append(table(["Method","LOAD fills","RFO fills","PF fills","WB fills","Replacements"],[[tex(label(r))]+[fmt(res[r['run_id']].get(k),0) for k in ('fills_load','fills_rfo','fills_prefetch','fills_writeback','replacements')] for r in cold],"lrrrrr","scriptsize"))
    text.append(r"All listed cold runs reached full occupancy; exact first-attainment instruction/cycle counters are in occupancy\_milestones.csv. LOAD, RFO, prefetch and writeback fills retain distinct meanings. Useful line lifetimes and fill-to-first-demand timing are in resources.csv. Request lifecycle observations are measurement-only and are not neural features.")
    cohorts=rows(results_dir/'issue_cohorts.csv')
    grouped=[]
    for run in cold:
        own=[c for c in cohorts if c['run_id']==run['run_id']]
        grouped.append([tex(label(run))]+[fmt(sum(number(c.get(k)) or 0 for c in own),0) for k in ('enqueued','timely','late','unused','redundant_cache','redundant_inflight','censored')])
    text.append(table(["Method","Enqueued","Timely","Late","Unused","Cache red.","Flight red.","Censored"],grouped,"lrrrrrrr","scriptsize"))
    text.append(r"Cohorts use unique enqueue and first-demand lifecycle accounting. They are supplementary and need not equal the legacy useful/late counters. Pending/in-flight and resident-unused outcomes remain censored at the final boundary rather than being assigned failure.")
    text.append(r"\clearpage\section{Finite-service negative results}")
    text.append(panel_plot(out_dir,'finite_progress',[stride]+hardware,wins,[("ipc","Window IPC",1),("cumulative_cycles_saved_vs_stride","Cumulative cycles saved vs Stride (M)",1e-6),("useful_prefetch_coverage","Window useful-prefetch coverage",1),("request_pressure","Window requested prefetches / L2 LOAD",1)]))
    text.append(r"The frozen 4/16-MAC cases remain below matched Stride end to end. The same finite table at zero cost isolates the table capacity choice. Observed peak state entries and zero evictions show whether this trace actually pressures that capacity; timing congestion must not be attributed to h/c eviction when eviction is absent.")
    text.append(r"\clearpage\subsection{Inference work, admission and completion}")
    text.append(table(["Case","Eligible","Admitted","Completed","Input drops","States peak","Evictions"],[[tex(label(r))]+[fmt(res[r['run_id']].get(k),0) for k in ('eligible_callbacks','admitted_decisions','completed_decisions','input_drops','peak_state_entries','state_evictions')] for r in hardware],"lrrrrrr","scriptsize"))
    text.append(table(["Case","MAC/decision","Nonlinear/decision","Service cycles","Input peak","Output peak"],[[tex(label(r))]+[fmt(res[r['run_id']].get(k),2 if k.startswith('mean') else 0) for k in ('mean_inference_macs','mean_inference_nonlinears','inference_service_cycles','peak_input_queue','peak_output_queue')] for r in hardware],"lrrrrr","scriptsize"))
    text.append(r"Work averages use actual started decisions and decoded K. Finite service combines inference throughput, queued state dependencies and address arrival delay. Pending decisions at the endpoint remain pending. Queue pressure and redundant issued requests explain why a useful zero-cost algorithm can lose under these modeled resources. This does not demonstrate that every hardware implementation would have the same service cost.")
    text.append(r"\section{Frozen deployment storage}")
    chosen=next(r for r in hardware if int(r['service_case'])==4)
    entries=[s for s in storage if s['run_id']==chosen['run_id']]
    text.append(table(["h8, 64 entries: component","Configured KiB","Observed component peak KiB"],[[tex(s['category']),fmt(s['configured_max_kib'],3),fmt(s['observed_peak_kib'],3)] for s in entries],"lrr","small"))
    text.append(r"The h8 model has 1,908 FP32 parameters: 7,632 bytes (7.453125 KiB). H16 has 5,220 parameters: 20,880 bytes (20.390625 KiB). One state entry reserves 32 bytes for exact-PC tag/metadata and 8H bytes for FP32 h/c. Scratch, active event, decoded output and input/output queues are additional deployment storage. There is no neural shadow teacher, training buffer, gradient, optimizer state, training weight copy or publication buffer in frozen deployment.")
    text.append(table(["Method","Configured total KiB","Sum of observed component peaks KiB"],[[tex(label(r)),fmt(number(next(t['configured_max_bytes'] for t in totals if t['run_id']==r['run_id']))/1024 if number(next(t['configured_max_bytes'] for t in totals if t['run_id']==r['run_id'])) is not None else None,3),fmt(number(next(t['observed_peak_bytes'] for t in totals if t['run_id']==r['run_id']))/1024 if number(next(t['observed_peak_bytes'] for t in totals if t['run_id']==r['run_id'])) is not None else None,3)] for r in cold],"lrr","scriptsize"))
    text.append(r"\textbf{Refactored layout versus retained measurements.} The retained scientific runs used a 56-byte active event and a 16-by-56-byte input queue. The frozen-only refactor removes unused event fields, reducing these to 24 bytes and 16-by-24 bytes. Its configured h8/64-entry deployment is therefore 16,808 bytes (16.414 KiB), compared with the original measured layout's 17,352-byte reservation (16.945 KiB): a 544-byte structural reduction. These revised configured bytes are derived from the current packed layout, not retroactively measured allocation peaks. The component-peak and scientific counters above remain those recorded by the original frozen runs.")
    text.append(r"Unbounded state has no finite configured maximum (NA). Summed component peaks are an envelope, not a simultaneous allocation measurement. Fixed scratch reservations are not allocator measurements. The Stride row includes only its conventional baseline state. Measurement logging and host allocator/RSS overhead are excluded. Dedicated predictor SRAM is compared with the 256 KiB L2 by size; it is not assumed to occupy or pollute L2. These FP32 counts are not quantized deployment measurements.")
    text.append(r"\clearpage\section{Offline budget and historical frozen compatibility}")
    text.append(r"Offline i1m used 16,172 decisions, 2,486 positive decisions and 4,875 action atoms, with 12 epochs and 228 optimizer steps. I20m used 325,389 decisions, 52,059 positive decisions and 102,140 action atoms, with 12 epochs and 4,404 optimizer steps. These are offline model-preparation observations. They are not runtime observations or fixed-weight deployment allocations.")
    text.append(r"The previously reported 1M plateau concerns the offline pretraining budget across the completed prefix sweep. It does not mean a frozen runtime learns after one million observations, nor does similar IPC establish identical action behavior. I1m/i20m retain separate rows and their own traffic measurements. The completed historical live logs establish behavior directly; missing logs would not establish equivalence.")
    text.append(table(["Model","Original transport","IPC","Cycles","Coverage","Pressure"],[[f"h{r['hidden_size']} {tex(r['budget'])}",tex(r['mode']),fmt(r['ipc'],6),fmt(r['cycles'],0),fmt(r['useful_prefetch_coverage'],4),fmt(r['request_pressure'],4)] for r in historic],"llrrrr","scriptsize"))
    exact=lambda value:value is True or str(value).lower()=='true'
    comp=[]
    for run in hist:
        own=[c for c in compatibility if c['run_id']==run['run_id']]
        comp.append([tex(label(run)),f"{sum(exact(c['exact_match']) for c in own)}/{len(own)}",fmt(run['instructions'],0),fmt(run['cycles'],0)])
    text.append(table(["Preserved compatibility run","Original parser fields exact","Instructions","Cycles"],comp,"lrrr","scriptsize"))
    text.append(r"Each of the four historical frozen checks matches all 30 original parser fields, 120/120 in total. These are preserved measurements from the predecessor source, with no claim that this refactor newly executed the full historical protocol. The original no-prefetch historical reference has 203,131 measured L2 LOAD misses; it is used only for historical end metrics, never cold windows. Historical warm-cache IPC is not compared against cold-cache IPC as a performance gain.")
    text.append(r"\section{Interpretation and limits}")
    text.append(r"On this one held-out trace slice, offline-trained frozen inference provides useful zero-cost live prefetching with dynamic recurrent state. H8 already captures most of the measured i1m/i20m IPC behavior at lower weight cost than h16. The finite 4/16-MAC results expose queueing and lateness costs that erase that benefit under the stated architectural assumptions. They are retained as negative results. One trace and one pretraining seed do not establish generalization across workloads or statistical equivalence.")
    text.append(r"The branch's active entry point is run.py; design.md defines the fixed-weight conditions, and command.md records build/test/run/status/resume/report commands and report compilation. Compact raw counters and figure inputs are committed. SPEC traces, checkpoints and full raw logs remain in their existing private trees. All plotted scientific rows are actual frozen or conventional baseline measurements; weight-changing scientific rows are excluded rather than renamed. The raw reference paths continue to identify their original measurements.")
    text.append(r"\end{document}")
    destination=out_dir/'602_stride_dynamic_inference.tex'
    destination.write_text('\n\n'.join(text)+'\n')
    print(f'Wrote frozen report source {destination}')
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--historical-root',type=Path)
    parser.add_argument('--recipe',type=Path)
    args=parser.parse_args()
    generate(args.results_dir,args.out_dir,args.historical_root,args.recipe)


if __name__=='__main__':
    main()
