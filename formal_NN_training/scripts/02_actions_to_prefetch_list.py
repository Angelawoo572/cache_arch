#!/usr/bin/env python3
"""Convert LSTM cache-action CSV into a list_replayer prefetch list.

Output format:
  idx pf_addr

The new outcome-aware trainer exports pred_good_prefetch_prob and candidate_addr.
For compatibility, it still emits nn_action=PREFETCH_DELTA for chosen prefetches.
This script also supports future names such as PREFETCH_CANDIDATE.
"""
import argparse, csv
from pathlib import Path

PREFETCH_ACTIONS = {"PREFETCH_DELTA", "PREFETCH_CANDIDATE", "PREFETCH_CORRECTED_LINE"}
BYPASS_ACTIONS = {"BYPASS_OR_LOW_PRIORITY_INSERT", "BYPASS", "LOW_PRIORITY_INSERT"}

def first(fields, names):
    s = set(fields or [])
    for n in names:
        if n in s: return n
    return None

def to_float(x, default=0.0):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

def to_int(x, default=-1):
    try:
        if x is None or x == "": return default
        return int(float(x))
    except Exception:
        return default

def clean(path):
    with path.open("rb") as f:
        for raw in f:
            yield raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-line-bytes", type=int, default=64)
    ap.add_argument("--prefetch-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-threshold", type=float, default=0.60)
    ap.add_argument("--policy", choices=["action", "threshold"], default="action")
    ap.add_argument("--allow-bypass-prefetch", action="store_true")
    args = ap.parse_args()

    if not args.actions.exists() or args.actions.stat().st_size == 0:
        raise SystemExit(f"[error] empty/missing action table: {args.actions}")

    reader = csv.DictReader(clean(args.actions))
    fields = reader.fieldnames or []
    idx_col = first(fields, ["replay_access_idx", "l2_replay_access_idx", "demand_access_idx", "event_id", "idx", "cycle", "cycle_num"])
    if idx_col is None:
        raise SystemExit("[error] action table needs replay_access_idx or event_id/idx/cycle/cycle_num")

    good_col = first(fields, ["pred_good_prefetch_prob", "pred_useful_prob", "pred_future_hit_prob"])
    conf_col = first(fields, ["pred_delta_conf", "pred_conf", "delta_conf"])
    bypass_col = first(fields, ["pred_bypass_prob", "bypass_prob"])
    action_col = "nn_action" if "nn_action" in fields else None
    pf_byte_col = first(fields, ["prefetch_addr", "pf_addr", "candidate_addr", "pred_pf_addr"])
    pf_line_col = first(fields, ["prefetch_line_addr", "candidate_line_addr", "pred_pf_line", "pf_line"])
    current_line_col = first(fields, ["line_addr", "current_line_addr"])
    if pf_byte_col is None and pf_line_col is None:
        raise SystemExit("[error] action table needs prefetch_addr/pf_addr/candidate_addr or prefetch_line_addr")

    emitted = set(); n = skip_policy = skip_conf = skip_bypass = skip_self = skip_addr = 0
    for row in reader:
        n += 1
        action = str(row.get(action_col, "")) if action_col else ""
        if args.policy == "action":
            if action_col is not None and action not in PREFETCH_ACTIONS:
                skip_policy += 1; continue
        else:
            score = to_float(row.get(good_col), None) if good_col else None
            if score is None:
                score = to_float(row.get(conf_col), 0.0) if conf_col else 0.0
            if score < args.prefetch_threshold:
                skip_conf += 1; continue

        if not args.allow_bypass_prefetch:
            if bypass_col and to_float(row.get(bypass_col), 0.0) >= args.bypass_threshold:
                skip_bypass += 1; continue
            if action_col and action in BYPASS_ACTIONS:
                skip_bypass += 1; continue

        idx = to_int(row.get(idx_col), -1)
        if idx < 0:
            skip_addr += 1; continue
        if pf_byte_col:
            pf_addr = to_int(row.get(pf_byte_col), -1)
            pf_line = pf_addr // args.cache_line_bytes if pf_addr > 0 else -1
        else:
            pf_line = to_int(row.get(pf_line_col), -1)
            pf_addr = pf_line * args.cache_line_bytes if pf_line >= 0 else -1
        if pf_addr <= 0:
            skip_addr += 1; continue
        if current_line_col:
            cur_line = to_int(row.get(current_line_col), -1)
            if cur_line >= 0 and pf_line == cur_line:
                skip_self += 1; continue
        emitted.add((idx, pf_addr))

    pairs = sorted(emitted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for idx, pf_addr in pairs:
            f.write(f"{idx} 0x{pf_addr:x}\n")
    print(f"[input]  {args.actions}")
    print(f"[output] {args.out}")
    print(f"[policy] {args.policy}")
    print(f"[idx_col] {idx_col}")
    print(f"[rows]   input={n} emitted={len(pairs)} unique_pairs={len(emitted)}")
    print(f"[skip]   policy={skip_policy} low_conf={skip_conf} bypass={skip_bypass} self={skip_self} bad_addr={skip_addr}")
    if pairs: print(f"[range]  idx={pairs[0][0]}..{pairs[-1][0]}")

if __name__ == "__main__": main()
