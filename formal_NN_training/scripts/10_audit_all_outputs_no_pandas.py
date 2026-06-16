#!/usr/bin/env python3
"""Audit LSTM/SPP replay outputs without pandas.

Run from the repo root on the machine where the results were produced, e.g.
/scratch/qianruw/cache on ece000. If the audit prints everything missing, you are
probably on a different machine/filesystem than the one that produced the logs.
"""

import csv
import os
import re
import socket
from pathlib import Path

ROOT = Path.cwd().resolve()
LOG_DIR = ROOT / "formal_NN_training/results/replay_compare/logs"
CAP_LOG_DIR = ROOT / "formal_NN_training/results/capacity_sweep/logs"
BIN_DIR = ROOT / "external/ChampSim/bin"

TRACES = [
    "602.gcc_s-734B",
    "619.lbm_s-4268B",
    "605.mcf_s-994B",
    "620.omnetpp_s-874B",
]

CAP_TRACES = [
    "602.gcc_s-734B",
    "619.lbm_s-4268B",
]

CAPS = ["256K", "512K", "1M", "2M"]


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
        requested = issued = useful = useless = None

    return {
        "ipc": ipc,
        "requested": requested,
        "issued": issued,
        "useful": useful,
        "useless": useless,
    }


def ok_log(path):
    m = parse_log(path)
    return m is not None and m["ipc"] is not None


def check_action(trace):
    p = ROOT / f"formal_NN_training/artifacts/by_trace/{trace}/full_lstm_cache_actions.csv"
    if not p.exists():
        return "MISSING", 0, 0, []
    try:
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            fields = r.fieldnames or []
            if "replay_access_idx" not in fields:
                return "NO_REPLAY_IDX_FIELD", 0, 0, []
            n = blank = nonblank = 0
            examples = []
            for row in r:
                n += 1
                if row.get("replay_access_idx", "") == "":
                    blank += 1
                else:
                    nonblank += 1
                if len(examples) < 3:
                    examples.append((row.get("event_id"), row.get("replay_access_idx"), row.get("pred_good_prefetch_prob")))
                if n >= 100000:
                    break
            if blank == 0 and nonblank > 0:
                return "OK", blank, nonblank, examples
            return "BAD_BLANK_REPLAY_IDX", blank, nonblank, examples
    except Exception as e:
        return f"ERROR:{e}", 0, 0, []


def main():
    print("============================================================")
    print("AUDIT CONTEXT")
    print("============================================================")
    print("host:", socket.gethostname())
    print("cwd :", ROOT)
    print("user:", os.environ.get("USER", ""))

    print("\n============================================================")
    print("AUDIT: action files replay_access_idx")
    print("============================================================")
    for trace in TRACES:
        status, blank, nonblank, examples = check_action(trace)
        print(f"{trace}: {status} blank={blank} nonblank={nonblank} examples={examples}")

    print("\n============================================================")
    print("AUDIT: normal replay logs")
    print("============================================================")
    for trace in TRACES:
        print(f"\n[{trace}]")
        no = LOG_DIR / f"{trace}.no_prefetch.log"
        spp = LOG_DIR / f"{trace}.spp.log"
        print("  no_prefetch:", "OK" if ok_log(no) else "MISSING/BAD", no)
        print("  spp        :", "OK" if ok_log(spp) else "MISSING/BAD", spp)

        lstm_logs = sorted(LOG_DIR.glob(f"{trace}.LSTM*.log"))
        if not lstm_logs:
            print("  LSTM logs  : NONE")
        else:
            for p in lstm_logs:
                m = parse_log(p)
                if m and m["ipc"] is not None:
                    print(f"  LSTM       : OK ipc={m['ipc']} issued={m['issued']} useful={m['useful']} {p.name}")
                else:
                    print(f"  LSTM       : BAD {p.name}")

    print("\n============================================================")
    print("AUDIT: capacity binaries")
    print("============================================================")
    for cap in CAPS:
        for kind in ["baseline", "spp", "replayer"]:
            p = BIN_DIR / f"champsim.{kind}.L2_{cap}"
            print(f"{kind:8s} L2_{cap:4s}:", "OK" if p.exists() and p.stat().st_size > 0 else "MISSING", p)

    print("\n============================================================")
    print("AUDIT: capacity logs")
    print("============================================================")
    for trace in CAP_TRACES:
        print(f"\n[{trace}]")
        for cap in CAPS:
            for method in ["no_prefetch", "spp", "LSTM_th0.20_bp1.00"]:
                p = CAP_LOG_DIR / f"{trace}.L2_{cap}.{method}.log"
                print(f"  L2_{cap:4s} {method:20s}:", "OK" if ok_log(p) else "MISSING/BAD")

    print("\n============================================================")
    print("DONE")
    print("============================================================")


if __name__ == "__main__":
    main()
