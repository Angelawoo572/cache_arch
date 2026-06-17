#!/usr/bin/env python3
import csv
import re
from pathlib import Path

LOG_DIR = Path("formal_NN_training/results/LSTM/draft/capacity_sweep/logs")
OUT = Path("formal_NN_training/results/LSTM/draft/capacity_sweep/capacity_sweep_602_619.csv")

def parse_log(path):
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
        "useful_per_issued": useful / issued if issued else 0.0,
        "useful_over_useful_plus_useless": useful / (useful + useless) if (useful + useless) else 0.0,
    }

rows = []

for p in sorted(LOG_DIR.glob("*.log")):
    name = p.name
    if ".L2_" not in name:
        continue

    trace, rest = name.split(".L2_", 1)
    cap, method_log = rest.split(".", 1)
    method = method_log[:-4] if method_log.endswith(".log") else method_log

    row = {
        "trace": trace,
        "l2_capacity": cap,
        "method": method,
        "log": str(p),
    }
    row.update(parse_log(p))
    rows.append(row)

# speedup within each trace/cap vs no_prefetch
base = {}
for r in rows:
    if r["method"] == "no_prefetch":
        base[(r["trace"], r["l2_capacity"])] = r["ipc"]

for r in rows:
    b = base.get((r["trace"], r["l2_capacity"]))
    r["speedup_vs_no_prefetch"] = (r["ipc"] / b) if (r["ipc"] is not None and b) else None

OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "trace", "l2_capacity", "method", "ipc", "speedup_vs_no_prefetch",
    "requested", "issued", "useful", "useless",
    "useful_per_issued", "useful_over_useful_plus_useless", "log"
]

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x["trace"], x["l2_capacity"], x["method"])):
        w.writerow({k: r.get(k, "") for k in fields})

print("[write]", OUT)
