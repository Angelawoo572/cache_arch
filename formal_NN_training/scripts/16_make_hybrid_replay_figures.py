#!/usr/bin/env python3
"""Summarize one SPP+LSTM hybrid replay suite without overwriting final figures.

Expected suite layout created by 15_run_hybrid_replay_suite.sh:
  <suite>/replay_compare/logs/*.log
  <suite>/capacity_sweep/logs/*.log

Outputs:
  <suite>/tables/all_replay_metrics.csv
  <suite>/tables/normal_replay_metrics.csv
  <suite>/tables/timing_replay_metrics.csv
  <suite>/tables/capacity_replay_metrics.csv
  <suite>/figures/*.svg
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DEFAULT_TRACES = [
    "602.gcc_s-734B",
    "619.lbm_s-4268B",
    "605.mcf_s-994B",
    "620.omnetpp_s-874B",
    "623.xalancbmk_s-700B",
]
DEFAULT_CAPS = ["256K", "512K", "1M", "2M"]

COLORS = {
    "no_prefetch": "#888888",
    "spp": "#4C78A8",
    "hybrid_action": "#F58518",
    "LSTM_hybrid_action": "#F58518",
    "timing": "#54A24B",
}


def parse_log(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    text = path.read_text(errors="replace")
    ipc_m = re.search(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text)
    ipc = float(ipc_m.group(1)) if ipc_m else None
    m = re.search(
        r"cpu0->cpu0_L2C PREFETCH REQUESTED:\s*(\d+)\s+ISSUED:\s*(\d+)\s+USEFUL:\s*(\d+)\s+USELESS:\s*(\d+)",
        text,
    )
    if m:
        requested, issued, useful, useless = map(int, m.groups())
    else:
        requested = issued = useful = useless = 0
    return {
        "ipc": ipc,
        "requested": requested,
        "issued": issued,
        "useful": useful,
        "useless": useless,
        "useful_per_issued": useful / float(issued) if issued else 0.0,
        "useful_per_requested": useful / float(requested) if requested else 0.0,
        "useful_over_useful_plus_useless": useful / float(useful + useless) if (useful + useless) else 0.0,
    }


def add_row(rows, *, suite_tag, kind, trace, method, timing_range="", l2_capacity="", log_path: Path, metrics):
    if metrics is None or metrics.get("ipc") is None:
        return
    row = {
        "suite_tag": suite_tag,
        "kind": kind,
        "trace": trace,
        "method": method,
        "timing_range": timing_range,
        "l2_capacity": l2_capacity,
        "log": str(log_path),
    }
    row.update(metrics)
    rows.append(row)


def collect_rows(suite_root: Path, suite_tag: str, traces, caps):
    normal_log_dir = suite_root / "replay_compare" / "logs"
    cap_log_dir = suite_root / "capacity_sweep" / "logs"
    rows = []

    for trace in traces:
        add_row(
            rows,
            suite_tag=suite_tag,
            kind="normal",
            trace=trace,
            method="no_prefetch",
            log_path=normal_log_dir / f"{trace}.no_prefetch.log",
            metrics=parse_log(normal_log_dir / f"{trace}.no_prefetch.log"),
        )
        add_row(
            rows,
            suite_tag=suite_tag,
            kind="normal",
            trace=trace,
            method="spp",
            log_path=normal_log_dir / f"{trace}.spp.log",
            metrics=parse_log(normal_log_dir / f"{trace}.spp.log"),
        )
        add_row(
            rows,
            suite_tag=suite_tag,
            kind="normal",
            trace=trace,
            method="hybrid_action",
            log_path=normal_log_dir / f"{trace}.LSTM_hybrid_action.log",
            metrics=parse_log(normal_log_dir / f"{trace}.LSTM_hybrid_action.log"),
        )

        for p in sorted(normal_log_dir.glob(f"{trace}.LSTM_hybrid_action_t*.log")):
            stem = p.name[:-4]
            prefix = f"{trace}.LSTM_hybrid_action_"
            timing = stem[len(prefix):] if stem.startswith(prefix) else stem
            add_row(
                rows,
                suite_tag=suite_tag,
                kind="timing",
                trace=trace,
                method="hybrid_action_timing",
                timing_range=timing,
                log_path=p,
                metrics=parse_log(p),
            )

        for cap in caps:
            for method in ["no_prefetch", "spp", "hybrid_action"]:
                log = cap_log_dir / f"{trace}.L2_{cap}.{method}.log"
                add_row(
                    rows,
                    suite_tag=suite_tag,
                    kind="capacity",
                    trace=trace,
                    method=method,
                    l2_capacity=cap,
                    log_path=log,
                    metrics=parse_log(log),
                )

    compute_relative_metrics(rows)
    return rows


def compute_relative_metrics(rows):
    no_ipc = {}
    spp_useful = {}
    for r in rows:
        key = (r["kind"], r["trace"], r.get("l2_capacity", ""))
        if r["method"] == "no_prefetch":
            no_ipc[key] = r["ipc"]
        if r["method"] == "spp":
            spp_useful[key] = r["useful"]

    for r in rows:
        if r["kind"] == "timing":
            key = ("normal", r["trace"], "")
        else:
            key = (r["kind"], r["trace"], r.get("l2_capacity", ""))
        base = no_ipc.get(key)
        r["speedup_vs_no_prefetch"] = r["ipc"] / base if base else None
        spp_u = spp_useful.get(key)
        r["coverage_vs_spp_useful"] = r["useful"] / float(spp_u) if spp_u else None


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.8f}"
    return v


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})
    print("[write]", path)


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def grouped_bar_svg(rows, out_path: Path, *, title, metric, ylabel, x_key, method_order, method_labels=None, value_fmt="{:.2f}"):
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        print("[skip figure] no rows for", out_path)
        return
    x_values = []
    for r in rows:
        x = x_key(r)
        if x not in x_values:
            x_values.append(x)
    methods = [m for m in method_order if any(r["method"] == m for r in rows)]
    if not methods:
        print("[skip figure] no requested methods for", out_path)
        return
    data = {(x_key(r), r["method"]): float(r[metric]) for r in rows}
    maxv = max([v for v in data.values()] + [1.0])
    if metric == "speedup_vs_no_prefetch":
        maxv = max(maxv, 1.05)
    if metric in ["useful_per_issued", "coverage_vs_spp_useful"]:
        maxv = max(maxv, 1.0)
    maxv *= 1.08

    W = max(1150, 170 * len(x_values))
    H = 720
    left, right, top, bottom = 95, 40, 70, 155
    plot_w = W - left - right
    plot_h = H - top - bottom
    group_w = plot_w / max(1, len(x_values))
    bar_w = group_w / max(4.5, len(methods) + 2.0)
    labels = method_labels or {m: m for m in methods}

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{W/2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{html_escape(title)}</text>')
    svg.append(f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20,{top + plot_h/2})" text-anchor="middle" font-size="14" font-family="Arial">{html_escape(ylabel)}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')
    for i in range(6):
        val = maxv * i / 5.0
        y = top + plot_h - (val / maxv) * plot_h
        svg.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
    if metric == "speedup_vs_no_prefetch" and maxv > 1.0:
        y = top + plot_h - (1.0 / maxv) * plot_h
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#333" stroke-dasharray="5,5"/>')

    for xi, xval in enumerate(x_values):
        gx = left + xi * group_w + group_w * 0.14
        for mi, method in enumerate(methods):
            val = data.get((xval, method))
            if val is None:
                continue
            h = 0 if maxv == 0 else (val / maxv) * plot_h
            x = gx + mi * bar_w
            y = top + plot_h - h
            color = COLORS.get(method, COLORS.get("timing", "#54A24B"))
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
            svg.append(f'<text x="{x+bar_w*0.41:.1f}" y="{max(y-5, 12):.1f}" text-anchor="middle" font-size="10" font-family="Arial">{html_escape(value_fmt.format(val))}</text>')
        svg.append(f'<text x="{left + xi*group_w + group_w/2:.1f}" y="{top+plot_h+38}" text-anchor="end" transform="rotate(-25 {left + xi*group_w + group_w/2:.1f},{top+plot_h+38})" font-size="11" font-family="Arial">{html_escape(xval)}</text>')

    lx, ly = left + plot_w - min(360, 145 * len(methods)), top
    for i, method in enumerate(methods):
        x = lx + (i % 3) * 145
        y = ly + (i // 3) * 24
        color = COLORS.get(method, COLORS.get("timing", "#54A24B"))
        svg.append(f'<rect x="{x}" y="{y}" width="16" height="16" fill="{color}"/>')
        svg.append(f'<text x="{x+24}" y="{y+13}" font-size="13" font-family="Arial">{html_escape(labels.get(method, method))}</text>')
    svg.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))
    print("[write]", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--suite-tag", default="")
    ap.add_argument("--traces", default=" ".join(DEFAULT_TRACES))
    ap.add_argument("--caps", default=" ".join(DEFAULT_CAPS))
    args = ap.parse_args()

    suite_root = args.suite_root.resolve()
    suite_tag = args.suite_tag or suite_root.name
    traces = args.traces.split()
    caps = args.caps.split()

    rows = collect_rows(suite_root, suite_tag, traces, caps)
    table_dir = suite_root / "tables"
    fig_dir = suite_root / "figures"
    fields = [
        "suite_tag", "kind", "trace", "method", "timing_range", "l2_capacity",
        "ipc", "speedup_vs_no_prefetch", "requested", "issued", "useful", "useless",
        "useful_per_issued", "useful_per_requested", "useful_over_useful_plus_useless",
        "coverage_vs_spp_useful", "log",
    ]
    write_csv(table_dir / "all_replay_metrics.csv", rows, fields)
    for kind in ["normal", "timing", "capacity"]:
        write_csv(table_dir / f"{kind}_replay_metrics.csv", [r for r in rows if r["kind"] == kind], fields)

    normal = [r for r in rows if r["kind"] == "normal"]
    normal_methods = ["no_prefetch", "spp", "hybrid_action"]
    normal_labels = {"no_prefetch": "No prefetch", "spp": "SPP", "hybrid_action": "SPP+LSTM hybrid"}
    grouped_bar_svg(normal, fig_dir / "normal_ipc.svg", title=f"Normal replay IPC ({suite_tag})", metric="ipc", ylabel="IPC", x_key=lambda r: r["trace"], method_order=normal_methods, method_labels=normal_labels, value_fmt="{:.3f}")
    grouped_bar_svg(normal, fig_dir / "normal_speedup.svg", title=f"Normal replay speedup ({suite_tag})", metric="speedup_vs_no_prefetch", ylabel="Speedup vs no-prefetch", x_key=lambda r: r["trace"], method_order=normal_methods, method_labels=normal_labels, value_fmt="{:.3f}")
    grouped_bar_svg([r for r in normal if r["method"] != "no_prefetch"], fig_dir / "normal_accuracy_useful_per_issued.svg", title=f"Replay accuracy: useful / issued ({suite_tag})", metric="useful_per_issued", ylabel="Useful / issued", x_key=lambda r: r["trace"], method_order=["spp", "hybrid_action"], method_labels=normal_labels, value_fmt="{:.2%}")
    grouped_bar_svg([r for r in normal if r["method"] != "no_prefetch"], fig_dir / "normal_coverage_vs_spp.svg", title=f"Coverage proxy: useful prefetches vs SPP ({suite_tag})", metric="coverage_vs_spp_useful", ylabel="Useful / SPP useful", x_key=lambda r: r["trace"], method_order=["spp", "hybrid_action"], method_labels=normal_labels, value_fmt="{:.2f}")

    timing = [r for r in rows if r["kind"] == "timing"]
    if timing:
        timing_fig_rows = []
        for r in timing:
            rr = dict(r)
            rr["method"] = r["timing_range"]
            timing_fig_rows.append(rr)
        timing_methods = sorted(set(r["method"] for r in timing_fig_rows))
        grouped_bar_svg(timing_fig_rows, fig_dir / "timing_speedup.svg", title=f"Timing-filtered hybrid speedup ({suite_tag})", metric="speedup_vs_no_prefetch", ylabel="Speedup vs no-prefetch", x_key=lambda r: r["trace"], method_order=timing_methods, method_labels={m: m for m in timing_methods}, value_fmt="{:.3f}")
        grouped_bar_svg(timing_fig_rows, fig_dir / "timing_accuracy_useful_per_issued.svg", title=f"Timing-filtered accuracy ({suite_tag})", metric="useful_per_issued", ylabel="Useful / issued", x_key=lambda r: r["trace"], method_order=timing_methods, method_labels={m: m for m in timing_methods}, value_fmt="{:.2%}")

    capacity = [r for r in rows if r["kind"] == "capacity"]
    if capacity:
        grouped_bar_svg(capacity, fig_dir / "capacity_speedup.svg", title=f"Capacity sweep speedup ({suite_tag})", metric="speedup_vs_no_prefetch", ylabel="Speedup vs no-prefetch", x_key=lambda r: f'{r["trace"]}\n{r["l2_capacity"]}', method_order=normal_methods, method_labels=normal_labels, value_fmt="{:.3f}")
        grouped_bar_svg(capacity, fig_dir / "capacity_ipc.svg", title=f"Capacity sweep IPC ({suite_tag})", metric="ipc", ylabel="IPC", x_key=lambda r: f'{r["trace"]}\n{r["l2_capacity"]}', method_order=normal_methods, method_labels=normal_labels, value_fmt="{:.3f}")

    print("[done] suite summary:", suite_root)


if __name__ == "__main__":
    main()
