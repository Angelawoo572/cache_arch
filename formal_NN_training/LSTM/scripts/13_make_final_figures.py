#!/usr/bin/env python3
"""Generate final CSV tables and SVG figures from replay/capacity logs.

No pandas/matplotlib dependency. Intended to run on the cluster.

Examples:
  python3 formal_NN_training/LSTM/scripts/13_make_final_figures.py

  TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
    python3 formal_NN_training/LSTM/scripts/13_make_final_figures.py
"""

from __future__ import print_function

import argparse
import csv
import os
import re
from pathlib import Path

DEFAULT_TRACES = [
    "602.gcc_s-734B",
    "619.lbm_s-4268B",
    "605.mcf_s-994B",
    "620.omnetpp_s-874B",
    "623.xalancbmk_s-700B",
]

BAD_LSTM_TOKENS = [
    "timing_",
    "lead",
    "manual",
    "_threshold_",
    "threshold_th",
    "action_th",
]
GOOD_LSTM_TOKENS = [
    "L2_replayidx_hex",
    "L2_aligned_hex",
    "allow_bypass",
]

COLORS = {
    "no_prefetch": "#888888",
    "spp": "#4C78A8",
    "best_lstm": "#F58518",
    "LSTM_th0.20_bp1.00": "#F58518",
}

LABELS = {
    "no_prefetch": "No prefetch",
    "spp": "SPP",
    "best_lstm": "Best LSTM",
    "LSTM_th0.20_bp1.00": "LSTM",
}


def parse_log(path):
    if not path.exists():
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
        "useful_over_useful_plus_useless": useful / float(useful + useless) if (useful + useless) else 0.0,
    }


def method_name(trace, path):
    name = path.name
    prefix = trace + "."
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(".log"):
        name = name[:-4]
    return name


def valid_lstm_method(name):
    if not name.startswith("LSTM"):
        return False
    if any(tok in name for tok in BAD_LSTM_TOKENS):
        return False
    if "aligned_th" in name and "aligned_hex" not in name:
        return False
    return any(tok in name for tok in GOOD_LSTM_TOKENS)


def fmt_float(x, nd=8):
    if x is None:
        return "NA"
    return ("{:0." + str(nd) + "f}").format(float(x))


def collect_normal_rows(log_dir, traces):
    rows = []
    candidates = []
    for trace in traces:
        no_log = log_dir / f"{trace}.no_prefetch.log"
        spp_log = log_dir / f"{trace}.spp.log"
        no = parse_log(no_log)
        if no is None or no["ipc"] is None:
            print("[warn] missing no_prefetch log for", trace)
            continue
        base_ipc = no["ipc"]
        rows.append({
            "trace": trace,
            "method_group": "no_prefetch",
            "method": "no_prefetch",
            "ipc": base_ipc,
            "speedup_vs_no_prefetch": 1.0,
            "requested": 0,
            "issued": 0,
            "useful": 0,
            "useless": 0,
            "useful_per_issued": 0.0,
            "useful_over_useful_plus_useless": 0.0,
            "log": str(no_log),
        })
        spp = parse_log(spp_log)
        if spp is not None and spp["ipc"] is not None:
            rows.append({
                "trace": trace,
                "method_group": "spp",
                "method": "spp",
                "ipc": spp["ipc"],
                "speedup_vs_no_prefetch": spp["ipc"] / base_ipc if base_ipc else None,
                "requested": spp["requested"],
                "issued": spp["issued"],
                "useful": spp["useful"],
                "useless": spp["useless"],
                "useful_per_issued": spp["useful_per_issued"],
                "useful_over_useful_plus_useless": spp["useful_over_useful_plus_useless"],
                "log": str(spp_log),
            })
        best = None
        for p in sorted(log_dir.glob(f"{trace}.LSTM*.log")):
            name = method_name(trace, p)
            m = parse_log(p)
            if m is None or m["ipc"] is None:
                continue
            keep = valid_lstm_method(name)
            cand = {
                "trace": trace,
                "keep": keep,
                "method": name,
                "ipc": m["ipc"],
                "speedup_vs_no_prefetch": m["ipc"] / base_ipc if base_ipc else None,
                "requested": m["requested"],
                "issued": m["issued"],
                "useful": m["useful"],
                "useless": m["useless"],
                "useful_per_issued": m["useful_per_issued"],
                "useful_over_useful_plus_useless": m["useful_over_useful_plus_useless"],
                "log": str(p),
            }
            candidates.append(cand)
            if keep and (best is None or m["ipc"] > best["ipc"]):
                best = cand
        if best:
            rows.append({
                "trace": trace,
                "method_group": "best_lstm",
                "method": best["method"],
                "ipc": best["ipc"],
                "speedup_vs_no_prefetch": best["speedup_vs_no_prefetch"],
                "requested": best["requested"],
                "issued": best["issued"],
                "useful": best["useful"],
                "useless": best["useless"],
                "useful_per_issued": best["useful_per_issued"],
                "useful_over_useful_plus_useless": best["useful_over_useful_plus_useless"],
                "log": best["log"],
            })
        else:
            print("[warn] no valid LSTM for", trace)
    return rows, candidates


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if k in ["ipc", "speedup_vs_no_prefetch", "useful_per_issued", "useful_over_useful_plus_useless"] and v != "":
                    out[k] = fmt_float(v)
                else:
                    out[k] = v
            w.writerow(out)
    print("[write]", path)


def make_bar_svg(rows, metric, out_path, title, ylabel, value_fmt):
    traces = []
    for r in rows:
        if r["trace"] not in traces:
            traces.append(r["trace"])
    methods = ["no_prefetch", "spp", "best_lstm"]
    data = {(r["trace"], r["method_group"]): float(r[metric]) for r in rows if r.get(metric) not in [None, ""]}
    maxv = max(data.values()) if data else 1.0
    if metric == "speedup_vs_no_prefetch":
        maxv = max(maxv, 1.05)
    W, H = 1280, 700
    left, right, top, bottom = 95, 40, 70, 135
    plot_w = W - left - right
    plot_h = H - top - bottom
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{W/2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>')
    svg.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20,{top+plot_h/2})" text-anchor="middle" font-size="14" font-family="Arial">{ylabel}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')
    for i in range(6):
        val = maxv * i / 5.0
        y = top + plot_h - (val / maxv) * plot_h
        svg.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
    if metric == "speedup_vs_no_prefetch":
        y = top + plot_h - (1.0 / maxv) * plot_h
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#444" stroke-dasharray="5,5"/>')
    group_w = plot_w / max(1, len(traces))
    bar_w = group_w / 5.0
    for ti, trace in enumerate(traces):
        gx = left + ti * group_w + group_w * 0.16
        for mi, method in enumerate(methods):
            val = data.get((trace, method))
            if val is None:
                continue
            h = (val / maxv) * plot_h
            x = gx + mi * bar_w
            y = top + plot_h - h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" fill="{COLORS[method]}"/>')
            svg.append(f'<text x="{x+bar_w*0.42:.1f}" y="{y-5:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value_fmt.format(val)}</text>')
        svg.append(f'<text x="{left + ti*group_w + group_w/2:.1f}" y="{top+plot_h+35}" text-anchor="middle" font-size="12" font-family="Arial">{trace}</text>')
    lx, ly = left + plot_w - 320, top
    for i, method in enumerate(methods):
        y = ly + i * 24
        svg.append(f'<rect x="{lx}" y="{y}" width="16" height="16" fill="{COLORS[method]}"/>')
        svg.append(f'<text x="{lx+24}" y="{y+13}" font-size="13" font-family="Arial">{LABELS[method]}</text>')
    svg.append('</svg>')
    out_path.write_text("\n".join(svg))
    print("[write]", out_path)


def parse_capacity_logs(cap_log_dir, out_csv):
    cap_order = {"256K": 0, "512K": 1, "1M": 2, "2M": 3}
    method_order = {"no_prefetch": 0, "spp": 1, "LSTM_th0.20_bp1.00": 2}
    rows = []
    for p in sorted(cap_log_dir.glob("*.log")):
        name = p.name
        if ".L2_" not in name:
            continue
        trace, rest = name.split(".L2_", 1)
        cap, method_log = rest.split(".", 1)
        method = method_log[:-4] if method_log.endswith(".log") else method_log
        if cap not in cap_order or method not in method_order:
            continue
        m = parse_log(p)
        if m is None or m["ipc"] is None:
            continue
        row = {"trace": trace, "l2_capacity": cap, "method": method, "log": str(p)}
        row.update(m)
        rows.append(row)
    base = {}
    for r in rows:
        if r["method"] == "no_prefetch":
            base[(r["trace"], r["l2_capacity"])] = r["ipc"]
    for r in rows:
        b = base.get((r["trace"], r["l2_capacity"]))
        r["speedup_vs_no_prefetch"] = r["ipc"] / b if b else None
    fields = ["trace", "l2_capacity", "method", "ipc", "speedup_vs_no_prefetch", "requested", "issued", "useful", "useless", "useful_per_issued", "useful_over_useful_plus_useless", "log"]
    rows = sorted(rows, key=lambda r: (r["trace"], cap_order[r["l2_capacity"]], method_order[r["method"]]))
    write_csv(out_csv, rows, fields)
    return rows


def make_capacity_speedup_svg(rows, out_path):
    if not rows:
        return
    traces = []
    for r in rows:
        if r["trace"] not in traces:
            traces.append(r["trace"])
    caps = ["256K", "512K", "1M", "2M"]
    methods = ["no_prefetch", "spp", "LSTM_th0.20_bp1.00"]
    data = {(r["trace"], r["l2_capacity"], r["method"]): float(r["speedup_vs_no_prefetch"]) for r in rows if r.get("speedup_vs_no_prefetch") is not None}
    maxv = max(max(data.values()), 1.05) if data else 1.05
    W, H = 1150, 80 + 410 * len(traces)
    left, right, top0, bottom = 90, 190, 60, 70
    panel_gap, panel_h = 70, 340
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">Speedup vs L2 Capacity</text>')
    for ti, trace in enumerate(traces):
        top = top0 + ti * (panel_h + panel_gap)
        plot_w = W - left - right
        plot_h = panel_h - bottom
        y0 = top + 30
        svg.append(f'<text x="{W/2}" y="{top}" text-anchor="middle" font-size="18" font-family="Arial">{trace}</text>')
        svg.append(f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0+plot_h}" stroke="black"/>')
        svg.append(f'<line x1="{left}" y1="{y0+plot_h}" x2="{left+plot_w}" y2="{y0+plot_h}" stroke="black"/>')
        for i in range(6):
            val = maxv * i / 5.0
            y = y0 + plot_h - (val / maxv) * plot_h
            svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        y_base = y0 + plot_h - (1.0 / maxv) * plot_h
        svg.append(f'<line x1="{left}" y1="{y_base:.1f}" x2="{left+plot_w}" y2="{y_base:.1f}" stroke="#444" stroke-dasharray="5,5"/>')
        xs = []
        for ci, cap in enumerate(caps):
            x = left + ci * (plot_w / (len(caps) - 1))
            xs.append(x)
            svg.append(f'<text x="{x}" y="{y0+plot_h+30}" text-anchor="middle" font-size="12" font-family="Arial">{cap}</text>')
        for method in methods:
            pts = []
            for ci, cap in enumerate(caps):
                val = data.get((trace, cap, method))
                if val is None:
                    continue
                x = xs[ci]
                y = y0 + plot_h - (val / maxv) * plot_h
                pts.append((x, y, val))
            if len(pts) >= 2:
                d = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
                svg.append(f'<polyline points="{d}" fill="none" stroke="{COLORS[method]}" stroke-width="3"/>')
            for x, y, val in pts:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[method]}"/>')
                svg.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{val:.2f}x</text>')
    lx, ly = W - right + 20, 80
    for i, method in enumerate(methods):
        y = ly + i * 24
        svg.append(f'<rect x="{lx}" y="{y}" width="16" height="16" fill="{COLORS[method]}"/>')
        svg.append(f'<text x="{lx+24}" y="{y+13}" font-size="13" font-family="Arial">{LABELS[method]}</text>')
    svg.append('</svg>')
    out_path.write_text("\n".join(svg))
    print("[write]", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="*", default=None, help="Trace names. Defaults to TRACES env or common project traces.")
    ap.add_argument("--log-dir", type=Path, default=Path("formal_NN_training/results/LSTM/draft/replay_compare/logs"))
    ap.add_argument("--out-dir", type=Path, default=Path("formal_NN_training/results/final_tables"))
    ap.add_argument("--capacity-log-dir", type=Path, default=Path("formal_NN_training/results/LSTM/draft/capacity_sweep/logs"))
    ap.add_argument("--capacity-out-dir", type=Path, default=Path("formal_NN_training/results/capacity_sweep"))
    args = ap.parse_args()

    env_traces = os.environ.get("TRACES", "").split()
    traces = args.traces or env_traces or DEFAULT_TRACES

    rows, candidates = collect_normal_rows(args.log_dir, traces)
    fields = ["trace", "method_group", "method", "ipc", "speedup_vs_no_prefetch", "requested", "issued", "useful", "useless", "useful_per_issued", "useful_over_useful_plus_useless", "log"]
    cand_fields = ["trace", "keep", "method", "ipc", "speedup_vs_no_prefetch", "requested", "issued", "useful", "useless", "useful_per_issued", "useful_over_useful_plus_useless", "log"]

    write_csv(args.out_dir / "normal_best_by_trace.csv", rows, fields)
    write_csv(args.out_dir / "accuracy_compare_all_traces.csv", rows, fields)
    write_csv(args.out_dir / "normal_lstm_candidates_filtered.csv", candidates, cand_fields)

    make_bar_svg(rows, "ipc", args.out_dir / "normal_ipc_by_trace.svg", "IPC by Trace", "IPC", "{:.3f}")
    make_bar_svg(rows, "speedup_vs_no_prefetch", args.out_dir / "normal_speedup_by_trace.svg", "Speedup vs No Prefetch", "Speedup", "{:.2f}x")
    make_bar_svg(rows, "useful_per_issued", args.out_dir / "normal_useful_per_issued_by_trace.svg", "Useful / Issued Prefetches", "Useful per issued", "{:.2f}")

    if args.capacity_log_dir.exists():
        cap_rows = parse_capacity_logs(args.capacity_log_dir, args.capacity_out_dir / "capacity_sweep_602_619.csv")
        make_capacity_speedup_svg(cap_rows, args.capacity_out_dir / "capacity_sweep_speedup.svg")


if __name__ == "__main__":
    main()
