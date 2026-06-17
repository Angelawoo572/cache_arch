#!/usr/bin/env python3
"""Compare SPP and LSTM replay accuracy from ChampSim logs.

This script is the final SPP-vs-LSTM replay table helper. It parses:
  - CPU IPC
  - L2C PREFETCH REQUESTED / ISSUED / USEFUL / USELESS
  - useful_per_issued = USEFUL / ISSUED
  - useful_over_useful_plus_useless = USEFUL / (USEFUL + USELESS)

Example 602:
  python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
    --trace 602.gcc_s-734B \
    --include-lstm LSTM \
    --exclude-lstm action_th0.50

Example 619 valid replay-index logs only:
  python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
    --trace 619.lbm_s-4268B \
    --include-lstm replayidx \
    --exclude-lstm aligned_hex
"""

from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path


def parse_log(path):
    text = path.read_text(errors="replace")
    ipc_m = re.search(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text)
    ipc = float(ipc_m.group(1)) if ipc_m else None

    m = re.search(
        r"cpu0->cpu0_L2C PREFETCH REQUESTED:\s*(\d+)\s+ISSUED:\s*(\d+)\s+USEFUL:\s*(\d+)\s+USELESS:\s*(\d+)",
        text,
    )
    if not m:
        return None

    requested, issued, useful, useless = map(int, m.groups())
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


def parse_ipc_only(path):
    text = path.read_text(errors="replace")
    ipc_m = re.search(r"CPU 0 cumulative IPC:\s*([0-9.]+)", text)
    return float(ipc_m.group(1)) if ipc_m else None


def method_name(trace, path):
    name = path.name
    prefix = "%s." % trace
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(".log"):
        name = name[:-4]
    return name


def keep_lstm(name, include, exclude):
    if not name.startswith("LSTM"):
        return False
    if include and not all(s in name for s in include):
        return False
    if exclude and any(s in name for s in exclude):
        return False
    return True


def pct(x):
    return "{:.4f}%".format(100.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("formal_NN_training/results/LSTM/draft/replay_compare/logs"))
    ap.add_argument("--include-lstm", action="append", default=[], help="Require substring in LSTM method name. Can be repeated.")
    ap.add_argument("--exclude-lstm", action="append", default=[], help="Exclude substring in LSTM method name. Can be repeated.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logs = sorted(args.log_dir.glob("{}*.log".format(args.trace)))
    if not logs:
        raise SystemExit("[error] no logs found for trace {} under {}".format(args.trace, args.log_dir))

    rows = []
    no_ipc = None
    for p in logs:
        name = method_name(args.trace, p)
        if name == "no_prefetch":
            no_ipc = parse_ipc_only(p)
            break

    for p in logs:
        name = method_name(args.trace, p)
        if name != "spp" and not keep_lstm(name, args.include_lstm, args.exclude_lstm):
            continue
        metrics = parse_log(p)
        if metrics is None:
            print("[skip] missing L2C prefetch counters: {}".format(p))
            continue
        speedup = metrics["ipc"] / no_ipc if (metrics["ipc"] is not None and no_ipc) else None
        row = {"method": name, "speedup_vs_no_prefetch": speedup, "log": str(p)}
        row.update(metrics)
        rows.append(row)

    print("trace={}".format(args.trace))
    if no_ipc is not None:
        print("no_prefetch_ipc={:.4f}".format(no_ipc))
    print("method,ipc,speedup_vs_no_prefetch,requested,issued,useful,useless,useful_per_issued,useful_per_requested,useful_over_useful_plus_useless")

    for r in rows:
        speedup_str = "NA" if r["speedup_vs_no_prefetch"] is None else "{:.4f}".format(r["speedup_vs_no_prefetch"])
        ipc_str = "NA" if r["ipc"] is None else "{:.4f}".format(r["ipc"])
        print(
            "{method},{ipc},{speedup},{requested},{issued},{useful},{useless},{upi},{upr},{uou}".format(
                method=r["method"],
                ipc=ipc_str,
                speedup=speedup_str,
                requested=r["requested"],
                issued=r["issued"],
                useful=r["useful"],
                useless=r["useless"],
                upi=pct(r["useful_per_issued"]),
                upr=pct(r["useful_per_requested"]),
                uou=pct(r["useful_over_useful_plus_useless"]),
            )
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "method", "ipc", "speedup_vs_no_prefetch", "requested", "issued", "useful", "useless",
            "useful_per_issued", "useful_per_requested", "useful_over_useful_plus_useless", "log",
        ]
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                rr = dict(r)
                for k in ["useful_per_issued", "useful_per_requested", "useful_over_useful_plus_useless", "speedup_vs_no_prefetch"]:
                    rr[k] = "{:.8f}".format(rr[k]) if rr[k] is not None else "NA"
                w.writerow({k: rr.get(k, "") for k in fields})
        print("[write] {}".format(args.out))


if __name__ == "__main__":
    main()
