#!/usr/bin/env python3
"""Evaluate the current notebook formulation: LSTM next-delta prediction vs SPP.

This evaluator matches the current LSTM_cache_action_predictor.ipynb labels, where
future_delta is the next demand/access line delta in the exported action stream.
It is NOT an outcome_useful candidate-action evaluator.

No pandas dependency. Python 3.6 compatible.
"""

import argparse
import csv
from pathlib import Path


def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def rate(a, b):
    return float(a) / float(b) if b else 0.0


def clean_text_lines(path):
    with path.open("rb") as f:
        for raw in f:
            if b"\x00" in raw:
                raw = raw.replace(b"\x00", b"")
            yield raw.decode("utf-8", errors="replace")


def load_spp(events_path):
    spp = {}
    with events_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            eid = to_int(row.get("event_id"), -1)
            if eid < 0:
                continue
            addr = to_int(row.get("addr"), 0)
            line = addr // 64
            pf_addr = to_int(row.get("pf_addr"), -1)
            if pf_addr <= 0:
                pf_addr = to_int(row.get("prefetch_addr"), -1)
            pf_line = pf_addr // 64 if pf_addr > 0 else -1
            cand_delta = to_int(row.get("delta"), 0)
            spp_delta = to_int(row.get("spp_delta"), cand_delta)
            spp[eid] = {
                "event_line": line,
                "pf_line": pf_line,
                "cand_delta": cand_delta,
                "spp_delta": spp_delta,
            }
    return spp


def load_actions(actions_path):
    rows = []
    for row in csv.DictReader(clean_text_lines(actions_path)):
        eid = to_int(row.get("event_id"), -1)
        if eid < 0:
            continue
        line = to_int(row.get("line_addr"), -1)
        pred_delta = to_int(row.get("pred_delta"), 0)
        rows.append({
            "event_id": eid,
            "trace": row.get("trace", ""),
            "cycle": to_int(row.get("cycle_num"), eid),
            "line": line,
            "pred_delta": pred_delta,
            "pred_line": line + pred_delta,
            "nn_action": row.get("nn_action", ""),
        })
    rows.sort(key=lambda x: (x["trace"], x["cycle"], x["event_id"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()

    print("[load] events {}".format(args.events))
    spp = load_spp(args.events)
    print("[load] spp events = {}".format(len(spp)))

    print("[load] actions {}".format(args.actions))
    rows = load_actions(args.actions)
    print("[load] action rows = {}".format(len(rows)))

    K = args.window
    lines = [x["line"] for x in rows]

    total = 0
    zero_true = 0
    zero_pred = 0
    naive_zero_correct = 0

    lstm_delta_correct = 0
    spp_delta_correct = 0
    lstm_addr_correct = 0
    spp_addr_correct = 0
    lstm_window_hit = 0
    spp_window_hit = 0

    both_correct = 0
    lstm_only = 0
    spp_only = 0
    both_wrong = 0

    nonzero_total = 0
    nonzero_lstm_delta_correct = 0
    nonzero_spp_delta_correct = 0
    nonzero_lstm_addr_correct = 0
    nonzero_spp_addr_correct = 0
    nonzero_lstm_pred_rate = 0
    nonzero_spp_pred_rate = 0
    nonzero_lstm_window_hit = 0
    nonzero_spp_window_hit = 0

    # sanity checks for field consistency
    line_mismatch = 0
    pf_delta_mismatch = 0
    sanity_checked = 0

    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]
        if cur["trace"] != nxt["trace"]:
            continue
        eid = cur["event_id"]
        if eid not in spp:
            continue

        s = spp[eid]
        if s["pf_line"] < 0:
            continue

        sanity_checked += 1
        if s["event_line"] != cur["line"]:
            line_mismatch += 1
        if s["pf_line"] >= 0 and (s["pf_line"] - s["event_line"]) != s["cand_delta"]:
            pf_delta_mismatch += 1

        cur_line = cur["line"]
        true_future_line = nxt["line"]
        true_future_delta = true_future_line - cur_line

        lstm_pred_delta = cur["pred_delta"]
        lstm_pred_line = cur["pred_line"]
        spp_delta = s["cand_delta"]
        spp_pf_line = s["pf_line"]

        future_window = set(lines[i + 1:min(len(lines), i + 1 + K)])

        total += 1
        zero_true += int(true_future_delta == 0)
        zero_pred += int(lstm_pred_delta == 0)
        naive_zero_correct += int(true_future_delta == 0)

        l_delta_ok = (lstm_pred_delta == true_future_delta)
        s_delta_ok = (spp_delta == true_future_delta)
        l_addr_ok = (lstm_pred_line == true_future_line)
        s_addr_ok = (spp_pf_line == true_future_line)

        lstm_delta_correct += int(l_delta_ok)
        spp_delta_correct += int(s_delta_ok)
        lstm_addr_correct += int(l_addr_ok)
        spp_addr_correct += int(s_addr_ok)
        lstm_window_hit += int(lstm_pred_line in future_window)
        spp_window_hit += int(spp_pf_line in future_window)

        if l_addr_ok and s_addr_ok:
            both_correct += 1
        elif l_addr_ok and not s_addr_ok:
            lstm_only += 1
        elif s_addr_ok and not l_addr_ok:
            spp_only += 1
        else:
            both_wrong += 1

        if true_future_delta != 0:
            nonzero_total += 1
            nonzero_lstm_delta_correct += int(l_delta_ok)
            nonzero_spp_delta_correct += int(s_delta_ok)
            nonzero_lstm_addr_correct += int(l_addr_ok)
            nonzero_spp_addr_correct += int(s_addr_ok)
            nonzero_lstm_pred_rate += int(lstm_pred_delta != 0)
            nonzero_spp_pred_rate += int(spp_delta != 0)
            nonzero_lstm_window_hit += int((lstm_pred_line in future_window) and (lstm_pred_line != cur_line))
            nonzero_spp_window_hit += int((spp_pf_line in future_window) and (spp_pf_line != cur_line))

    metrics = [
        ("total_usable_rows", total),
        ("true_future_delta_zero", zero_true),
        ("true_future_delta_zero_rate", "{:.8f}".format(rate(zero_true, total))),
        ("lstm_pred_delta_zero", zero_pred),
        ("lstm_pred_delta_zero_rate", "{:.8f}".format(rate(zero_pred, total))),
        ("naive_always_zero_acc", "{:.8f}".format(rate(naive_zero_correct, total))),
        ("lstm_delta_top1_acc", "{:.8f}".format(rate(lstm_delta_correct, total))),
        ("spp_delta_top1_acc", "{:.8f}".format(rate(spp_delta_correct, total))),
        ("lstm_next_line_addr_acc", "{:.8f}".format(rate(lstm_addr_correct, total))),
        ("spp_next_line_addr_acc", "{:.8f}".format(rate(spp_addr_correct, total))),
        ("lstm_future_window{}_hit".format(K), "{:.8f}".format(rate(lstm_window_hit, total))),
        ("spp_future_window{}_hit".format(K), "{:.8f}".format(rate(spp_window_hit, total))),
        ("both_correct", both_correct),
        ("lstm_only", lstm_only),
        ("spp_only", spp_only),
        ("both_wrong", both_wrong),
        ("nonzero_total", nonzero_total),
        ("nonzero_rate", "{:.8f}".format(rate(nonzero_total, total))),
        ("lstm_nonzero_delta_acc", "{:.8f}".format(rate(nonzero_lstm_delta_correct, nonzero_total))),
        ("spp_nonzero_delta_acc", "{:.8f}".format(rate(nonzero_spp_delta_correct, nonzero_total))),
        ("lstm_nonzero_addr_acc", "{:.8f}".format(rate(nonzero_lstm_addr_correct, nonzero_total))),
        ("spp_nonzero_addr_acc", "{:.8f}".format(rate(nonzero_spp_addr_correct, nonzero_total))),
        ("lstm_nonzero_pred_rate", "{:.8f}".format(rate(nonzero_lstm_pred_rate, nonzero_total))),
        ("spp_nonzero_pred_rate", "{:.8f}".format(rate(nonzero_spp_pred_rate, nonzero_total))),
        ("lstm_nonzero_future_window{}_hit".format(K), "{:.8f}".format(rate(nonzero_lstm_window_hit, nonzero_total))),
        ("spp_nonzero_future_window{}_hit".format(K), "{:.8f}".format(rate(nonzero_spp_window_hit, nonzero_total))),
        ("sanity_checked_rows", sanity_checked),
        ("sanity_event_line_mismatch", line_mismatch),
        ("sanity_pf_line_minus_line_ne_delta", pf_delta_mismatch),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics:
            w.writerow([k, v])

    print("\n===== Current-label LSTM vs SPP summary =====")
    for k, v in metrics:
        print("{:<40} {}".format(k, v))
    print("\n[done] wrote {}".format(args.out))


if __name__ == "__main__":
    main()
