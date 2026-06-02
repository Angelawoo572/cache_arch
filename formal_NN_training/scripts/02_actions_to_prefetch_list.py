#!/usr/bin/env python3
"""Convert LSTM cache-action CSV into a list_replayer prefetch list.

No pandas dependency. Compatible with older cluster Python versions.
Also strips accidental NUL bytes from CSV lines.

Output format used by list_replayer:
  idx pf_addr

Default policy is conservative and only emits rows whose nn_action is
PREFETCH_DELTA. This avoids turning INSERT_NORMAL_NO_PREFETCH rows with high
pred_delta_conf into useless self-prefetches.
"""

import argparse
import csv
from pathlib import Path


def first_existing_column(fieldnames, names):
    available = set(fieldnames or [])
    for name in names:
        if name in available:
            return name
    return None


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=-1):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def clean_text_lines(path):
    with path.open("rb") as f:
        for raw in f:
            if b"\x00" in raw:
                raw = raw.replace(b"\x00", b"")
            yield raw.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-line-bytes", type=int, default=64)
    ap.add_argument("--prefetch-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-threshold", type=float, default=0.60)
    ap.add_argument(
        "--policy",
        choices=["action", "threshold"],
        default="action",
        help="action: require nn_action=PREFETCH_DELTA. threshold: use pred_delta_conf/bypass thresholds and pred_delta!=0.",
    )
    ap.add_argument(
        "--allow-bypass-prefetch",
        action="store_true",
        help="By default, BYPASS_OR_LOW_PRIORITY_INSERT rows are not converted into prefetches.",
    )
    args = ap.parse_args()

    if not args.actions.exists() or args.actions.stat().st_size == 0:
        raise SystemExit("[error] empty/missing action table: {}".format(args.actions))

    emitted_pairs = set()
    input_rows = 0
    skipped_policy = 0
    skipped_conf = 0
    skipped_bypass = 0
    skipped_zero_delta = 0
    skipped_addr = 0

    reader = csv.DictReader(clean_text_lines(args.actions))
    fieldnames = reader.fieldnames or []

    idx_col = first_existing_column(fieldnames, ["event_id", "idx", "cycle", "cycle_num"])
    if idx_col is None:
        raise SystemExit("[error] action table needs one of: event_id, idx, cycle, cycle_num")

    conf_col = first_existing_column(fieldnames, ["pred_delta_conf", "pred_conf", "delta_conf"])
    bypass_col = first_existing_column(fieldnames, ["pred_bypass_prob", "bypass_prob"])
    pred_delta_col = first_existing_column(fieldnames, ["pred_delta", "delta", "spp_delta"])
    action_col = "nn_action" if "nn_action" in fieldnames else None

    pf_byte_col = first_existing_column(fieldnames, ["prefetch_addr", "pf_addr", "pred_pf_addr"])
    pf_line_col = first_existing_column(fieldnames, ["prefetch_line_addr", "pred_pf_line", "pf_line"])
    if pf_byte_col is None and pf_line_col is None:
        raise SystemExit("[error] action table needs pf_addr/prefetch_addr or prefetch_line_addr")

    for row in reader:
        input_rows += 1

        action = str(row.get(action_col, "")) if action_col else ""
        pred_delta = to_int(row.get(pred_delta_col), 0) if pred_delta_col else 1

        if args.policy == "action":
            if action_col is not None and action != "PREFETCH_DELTA":
                skipped_policy += 1
                continue
            if pred_delta == 0:
                skipped_zero_delta += 1
                continue
        else:
            if pred_delta == 0:
                skipped_zero_delta += 1
                continue
            if conf_col is not None and to_float(row.get(conf_col), 0.0) < args.prefetch_threshold:
                skipped_conf += 1
                continue

        if not args.allow_bypass_prefetch:
            if bypass_col is not None and to_float(row.get(bypass_col), 0.0) >= args.bypass_threshold:
                skipped_bypass += 1
                continue
            if action_col is not None and action == "BYPASS_OR_LOW_PRIORITY_INSERT":
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
            f.write("{} {}\n".format(idx, pf_addr))

    print("[input]  {}".format(args.actions))
    print("[output] {}".format(args.out))
    print("[policy] {}".format(args.policy))
    print("[rows]   input={} emitted={} unique_pairs={}".format(input_rows, len(pairs), len(emitted_pairs)))
    print("[skip]   policy={} low_conf={} bypass={} zero_delta={} bad_addr={}".format(skipped_policy, skipped_conf, skipped_bypass, skipped_zero_delta, skipped_addr))
    if pairs:
        print("[range]  idx={}..{}".format(pairs[0][0], pairs[-1][0]))


if __name__ == "__main__":
    main()
