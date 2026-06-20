#!/usr/bin/env python3
"""Summarize signature-validated oracle-LSTM replay versus same-binary no-pref.

Produces one CSV row per trace with:
  * replay validity (dense PC/line signature check)
  * no-pref IPC, LSTM IPC, and speedup
  * prefetch aggressiveness / usefulness / lateness counters

Use only rows with replay_valid == 1 for research conclusions.
"""

from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path

TRACES = [
    "602.gcc_s-734B",
    "619.lbm_s-4268B",
    "605.mcf_s-994B",
    "620.omnetpp_s-874B",
    "623.xalancbmk_s-700B",
]


def read_text(path):
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


def last_number(pattern, text, cast=float):
    hits = re.findall(pattern, text, flags=re.MULTILINE)
    if not hits:
        return None
    value = hits[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return cast(value)
    except Exception:
        return None


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return float(a) / float(b)


def parse_replay(log_text):
    d = {}
    d["lstm_ipc"] = last_number(r"^Core_0_IPC\s+([0-9.]+)", log_text)
    for name in ("requested", "dropped", "issued", "filled", "useful", "useless", "late"):
        d["l2_" + name] = last_number(
            r"^Core_0_L2C_prefetch_{}\s+([0-9]+)".format(name), log_text, int
        )

    # Current signature-validated ListReplayer output.
    sig = re.findall(
        r"\[list_replayer\] emitted ([0-9]+) candidates over ([0-9]+) ROI L2 LOAD accesses "
        r"\(([0-9]+) matched access indices; ([0-9]+) signature mismatches; "
        r"([0-9]+) post-reference tail accesses; (reference enabled|reference DISABLED)\)",
        log_text,
    )
    if sig:
        emitted, loads, matched, mismatches, ref_tail, ref_state = sig[-1]
        d.update({
            "replay_emitted": int(emitted),
            "replay_l2_loads": int(loads),
            "replay_matched_indices": int(matched),
            "signature_mismatches": int(mismatches),
            "post_reference_tail_accesses": int(ref_tail),
            "reference_enabled": int(ref_state == "reference enabled"),
        })
    else:
        d.update({
            "replay_emitted": None,
            "replay_l2_loads": None,
            "replay_matched_indices": None,
            "signature_mismatches": None,
            "post_reference_tail_accesses": None,
            "reference_enabled": 0,
        })

    d["replay_valid"] = int(
        d["reference_enabled"] == 1 and d["signature_mismatches"] == 0 and
        d["replay_emitted"] is not None and d["replay_emitted"] > 0
    )
    return d


def parse_baseline(log_text):
    return {"no_pref_ipc": last_number(r"^Core_0_IPC\s+([0-9.]+)", log_text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True, type=Path,
                    help="Directory containing logs/<trace>.oracle_replacer.log")
    ap.add_argument("--baseline-dir", required=True, type=Path,
                    help="Directory containing logs/<trace>.no_pref.log")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--traces", default=" ".join(TRACES),
                    help="Space-separated trace stems")
    args = ap.parse_args()

    traces = args.traces.split()
    rows = []
    for trace in traces:
        replay_log = args.replay_dir / "logs" / (trace + ".oracle_replacer.log")
        base_log = args.baseline_dir / "logs" / (trace + ".no_pref.log")
        r = parse_replay(read_text(replay_log))
        b = parse_baseline(read_text(base_log))

        row = {"trace": trace, "replay_log": str(replay_log), "baseline_log": str(base_log)}
        row.update(b)
        row.update(r)
        row["speedup_vs_same_binary_no_pref"] = ratio(row.get("lstm_ipc"), row.get("no_pref_ipc"))
        row["prefetches_per_l2_load"] = ratio(row.get("l2_issued"), row.get("replay_l2_loads"))
        row["useful_per_issued"] = ratio(row.get("l2_useful"), row.get("l2_issued"))
        row["useful_per_filled"] = ratio(row.get("l2_useful"), row.get("l2_filled"))
        row["late_per_issued"] = ratio(row.get("l2_late"), row.get("l2_issued"))
        row["useless_per_issued"] = ratio(row.get("l2_useless"), row.get("l2_issued"))
        rows.append(row)

    columns = [
        "trace", "replay_valid", "reference_enabled", "signature_mismatches",
        "post_reference_tail_accesses", "no_pref_ipc", "lstm_ipc",
        "speedup_vs_same_binary_no_pref", "replay_l2_loads",
        "replay_emitted", "replay_matched_indices", "l2_requested", "l2_dropped",
        "l2_issued", "l2_filled", "l2_useful", "l2_useless", "l2_late",
        "prefetches_per_l2_load", "useful_per_issued", "useful_per_filled",
        "useless_per_issued", "late_per_issued", "replay_log", "baseline_log",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("trace,replay_valid,no_pref_ipc,lstm_ipc,speedup,pf_per_l2,useful/issued,late/issued,sig_mismatch")
    for r in rows:
        def fnum(v, n=4):
            return "-" if v is None else ("{:.%df}" % n).format(v)
        print(
            "{},{},{},{},{},{},{},{},{}".format(
                r["trace"], r["replay_valid"], fnum(r.get("no_pref_ipc"), 5),
                fnum(r.get("lstm_ipc"), 5), fnum(r.get("speedup_vs_same_binary_no_pref"), 4),
                fnum(r.get("prefetches_per_l2_load"), 4),
                fnum(r.get("useful_per_issued"), 4),
                fnum(r.get("late_per_issued"), 4),
                "-" if r.get("signature_mismatches") is None else r.get("signature_mismatches"),
            )
        )
    print("[wrote] {}".format(args.out))


if __name__ == "__main__":
    main()
