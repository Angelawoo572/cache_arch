#!/usr/bin/env python3
"""Parse ChampSim replay logs into compact accuracy/coverage/IPC metrics.

Example:
  python3 formal_NN_training/scripts/08_parse_replay_metrics.py --trace 619.lbm_s-4268B

Metrics:
  useful_per_issued    = USEFUL / ISSUED
  useful_per_requested = USEFUL / REQUESTED
  useful_over_useful_plus_useless = USEFUL / (USEFUL + USELESS)

Use useful_per_issued as the issued-prefetch precision. Do not confuse it
with USEFUL/(USEFUL+USELESS), because ChampSim's USELESS counter is not
all non-useful issued prefetches.
"""

import argparse
import csv
import re
from pathlib import Path


def parse_log(path: Path):
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
        "useful_per_issued": useful / issued if issued else 0.0,
        "useful_per_requested": useful / requested if requested else 0.0,
        "useful_over_useful_plus_useless": useful / (useful + useless) if (useful + useless) else 0.0,
    }


def fmt_pct(x):
    return f"{100.0 * x:.4f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("formal_NN_training/results/replay_compare/logs"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logs = sorted(args.log_dir.glob(f"{args.trace}*.log"))
    if not logs:
        raise SystemExit(f"[error] no logs found for trace {args.trace} under {args.log_dir}")

    rows = []
    for path in logs:
        metrics = parse_log(path)
        if metrics is None:
            continue
        method = path.name
        prefix = f"{args.trace}."
        if method.startswith(prefix):
            method = method[len(prefix):]
        if method.endswith(".log"):
            method = method[:-4]
        row = {"method": method, "log": str(path), **metrics}
        rows.append(row)

    fields = [
        "method",
        "ipc",
        "requested",
        "issued",
        "useful",
        "useless",
        "useful_per_issued",
        "useful_per_requested",
        "useful_over_useful_plus_useless",
        "log",
    ]

    print("method,ipc,requested,issued,useful,useless,useful_per_issued,useful_per_requested,useful_over_useful_plus_useless")
    for r in rows:
        print(
            f"{r['method']},{r['ipc'] if r['ipc'] is not None else 'NA'},"
            f"{r['requested']},{r['issued']},{r['useful']},{r['useless']},"
            f"{fmt_pct(r['useful_per_issued'])},{fmt_pct(r['useful_per_requested'])},{fmt_pct(r['useful_over_useful_plus_useless'])}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                out_r = dict(r)
                for k in ["useful_per_issued", "useful_per_requested", "useful_over_useful_plus_useless"]:
                    out_r[k] = f"{r[k]:.8f}"
                w.writerow(out_r)
        print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
