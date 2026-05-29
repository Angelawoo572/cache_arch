#!/usr/bin/env python3
"""Convert LSTM cache-action CSV into a list_replayer prefetch list.

Input is produced by LSTM_cache_action_predictor.ipynb:
  formal_NN_training/artifacts/full_lstm_cache_actions.csv
or
  formal_NN_training/artifacts/val_lstm_cache_actions.csv

Output format is the simple list_replayer format used by the existing GRU flow:
  idx pf_addr

Where idx is the event index in the dumped/replayed simulation window and pf_addr
is a byte address. If the notebook only exports line addresses, this script
multiplies by cache-line bytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-line-bytes", type=int, default=64)
    ap.add_argument("--prefetch-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-threshold", type=float, default=0.60)
    ap.add_argument("--allow-bypass-prefetch", action="store_true",
                    help="By default, BYPASS_OR_LOW_PRIORITY_INSERT rows are not converted into prefetches.")
    args = ap.parse_args()

    df = pd.read_csv(args.actions)
    if df.empty:
        raise SystemExit(f"[error] empty action table: {args.actions}")

    idx_col = first_existing_column(df, ["event_id", "idx", "cycle", "cycle_num"])
    if idx_col is None:
        raise SystemExit("[error] action table needs one of: event_id, idx, cycle, cycle_num")

    conf_col = first_existing_column(df, ["pred_delta_conf", "pred_conf", "delta_conf"])
    bypass_col = first_existing_column(df, ["pred_bypass_prob", "bypass_prob"])
    action_col = "nn_action" if "nn_action" in df.columns else None

    pf_byte_col = first_existing_column(df, ["prefetch_addr", "pf_addr", "pred_pf_addr"])
    pf_line_col = first_existing_column(df, ["prefetch_line_addr", "pred_pf_line", "pf_line"])

    if pf_byte_col is None and pf_line_col is None:
        raise SystemExit("[error] action table needs pf_addr/prefetch_addr or prefetch_line_addr")

    work = df.copy()
    if conf_col is not None:
        work = work[pd.to_numeric(work[conf_col], errors="coerce").fillna(0) >= args.prefetch_threshold]
    if bypass_col is not None and not args.allow_bypass_prefetch:
        work = work[pd.to_numeric(work[bypass_col], errors="coerce").fillna(0) < args.bypass_threshold]
    if action_col is not None and not args.allow_bypass_prefetch:
        work = work[work[action_col].astype(str) != "BYPASS_OR_LOW_PRIORITY_INSERT"]

    work = work.dropna(subset=[idx_col])
    work[idx_col] = pd.to_numeric(work[idx_col], errors="coerce").fillna(-1).astype("int64")
    work = work[work[idx_col] >= 0]

    if pf_byte_col is not None:
        work["_pf_addr"] = pd.to_numeric(work[pf_byte_col], errors="coerce").fillna(-1).astype("int64")
    else:
        work["_pf_addr"] = (pd.to_numeric(work[pf_line_col], errors="coerce").fillna(-1).astype("int64") * args.cache_line_bytes)

    work = work[work["_pf_addr"] > 0]
    work = work.sort_values([idx_col, "_pf_addr"]).drop_duplicates([idx_col, "_pf_addr"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for _, row in work.iterrows():
            f.write(f"{int(row[idx_col])} {int(row['_pf_addr'])}\n")

    print(f"[input]  {args.actions}")
    print(f"[output] {args.out}")
    print(f"[rows]   input={len(df)} emitted={len(work)}")
    if len(work):
        print(f"[range]  idx={int(work[idx_col].min())}..{int(work[idx_col].max())}")


if __name__ == "__main__":
    main()
