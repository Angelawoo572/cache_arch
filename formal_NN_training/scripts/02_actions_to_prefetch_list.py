#!/usr/bin/env python3
"""Convert LSTM cache-action CSV into a list_replayer prefetch list.

No pandas dependency. This runs on the cluster's plain Python.

Input is produced by LSTM_cache_action_predictor.ipynb:
  formal_NN_training/artifacts/full_lstm_cache_actions.csv
or
  formal_NN_training/artifacts/val_lstm_cache_actions.csv

Output format used by list_replayer:
  idx pf_addr

Where idx is event_id/cycle index in the dumped/replayed simulation window and
pf_addr is a byte address.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def first_existing_column(fieldnames: list[str], names: list[str]) -> str | None:
    available = set(fieldnames or [])
    for name in names:
        if name in available:
            return name
    return None


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-line-bytes", type=int, default=64)
    ap.add_argument("--prefetch-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-threshold", type=float, default=0.60)
    ap.add_argument(
        "--allow-bypass-prefetch",
        action="store_true",
        help="By default, BYPASS_OR_LOW_PRIORITY_INSERT rows are not converted into prefetches.",
    )
    args = ap.parse_args()

    if not args.actions.exists() or args.actions.stat().st_size == 0:
        raise SystemExit(f"[error] empty/missing action table: {args.actions}")

    emitted_pairs: set[tuple[int, int]] = set()
    input_rows = 0
    skipped_conf = 0
    skipped_bypass = 0
    skipped_addr = 0

    with args.actions.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        idx_col = first_existing_column(fieldnames, ["event_id", "idx", "cycle", "cycle_num"])
        if idx_col is None:
            raise SystemExit("[error] action table needs one of: event_id, idx, cycle, cycle_num")

        conf_col = first_existing_column(fieldnames, ["pred_delta_conf", "pred_conf", "delta_conf"])
        bypass_col = first_existing_column(fieldnames, ["pred_bypass_prob", "bypass_prob"])
        action_col = "nn_action" if "nn_action" in fieldnames else None

        pf_byte_col = first_existing_column(fieldnames, ["prefetch_addr", "pf_addr", "pred_pf_addr"])
        pf_line_col = first_existing_column(fieldnames, ["prefetch_line_addr", "pred_pf_line", "pf_line"])
        if pf_byte_col is None and pf_line_col is None:
            raise SystemExit("[error] action table needs pf_addr/prefetch_addr or prefetch_line_addr")

        for row in reader:
            input_rows += 1

            if conf_col is not None and to_float(row.get(conf_col), 0.0) < args.prefetch_threshold:
                skipped_conf += 1
                continue

            if not args.allow_bypass_prefetch:
                if bypass_col is not None and to_float(row.get(bypass_col), 0.0) >= args.bypass_threshold:
                    skipped_bypass += 1
                    continue
                if action_col is not None and str(row.get(action_col, "")) == "BYPASS_OR_LOW_PRIORITY_INSERT":
                    skipped_bypass += 1
                    continue

            idx = to_int(row.get(idx_col), -1)
            if idx < 0:
                skipped_addr += 1
                continue

            if pf_byte_col is not None:
                pf_addr = to_int(row.get(pf_byte_col), -1)
            else:
                pf_line = to_int(row.get(pf_line_col), -1)
                pf_addr = pf_line * args.cache_line_bytes if pf_line >= 0 else -1

            if pf_addr <= 0:
                skipped_addr += 1
                continue

            emitted_pairs.add((idx, pf_addr))

    pairs = sorted(emitted_pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for idx, pf_addr in pairs:
            f.write(f"{idx} {pf_addr}\n")

    print(f"[input]  {args.actions}")
    print(f"[output] {args.out}")
    print(f"[rows]   input={input_rows} emitted={len(pairs)} unique_pairs={len(emitted_pairs)}")
    print(f"[skip]   low_conf={skipped_conf} bypass={skipped_bypass} bad_addr={skipped_addr}")
    if pairs:
        print(f"[range]  idx={pairs[0][0]}..{pairs[-1][0]}")


if __name__ == "__main__":
    main()
