#!/usr/bin/env python3
"""Evaluate LSTM cache-action predictions against dumped SPP event labels.

No pandas dependency. Streams the large event CSV and joins by event_id.

Main metrics:
  - SPP candidate/issued useful rate from the event table
  - LSTM prefetch precision/recall/F1 using outcome_useful as ground truth
  - future-hit classifier accuracy/precision/recall/F1
  - bypass classifier accuracy/precision/recall/F1
  - delta top-1 agreement against the candidate delta
"""

import argparse
import csv
from pathlib import Path


def clean_text_lines(path):
    with path.open("rb") as f:
        for raw in f:
            if b"\x00" in raw:
                raw = raw.replace(b"\x00", b"")
            yield raw.decode("utf-8", errors="replace")


def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def div(a, b):
    return float(a) / float(b) if b else 0.0


def prf(tp, fp, fn):
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def cls_metrics(tp, fp, tn, fn):
    acc = div(tp + tn, tp + fp + tn + fn)
    precision, recall, f1 = prf(tp, fp, fn)
    return acc, precision, recall, f1


def parse_actions(path, policy, prefetch_threshold, bypass_threshold, future_threshold):
    actions = {}
    counts = {}
    malformed = 0

    reader = csv.DictReader(clean_text_lines(path))
    for row in reader:
        eid = to_int(row.get("event_id"), -1)
        if eid < 0:
            malformed += 1
            continue

        nn_action = str(row.get("nn_action", ""))
        counts[nn_action] = counts.get(nn_action, 0) + 1
        pred_delta = to_int(row.get("pred_delta"), 0)
        pred_delta_conf = to_float(row.get("pred_delta_conf"), 0.0)
        pred_future_hit_prob = to_float(row.get("pred_future_hit_prob"), 0.0)
        pred_bypass_prob = to_float(row.get("pred_bypass_prob"), 0.0)

        if policy == "action":
            emit_prefetch = (nn_action == "PREFETCH_DELTA" and pred_delta != 0 and pred_bypass_prob < bypass_threshold)
        else:
            emit_prefetch = (pred_delta != 0 and pred_delta_conf >= prefetch_threshold and pred_bypass_prob < bypass_threshold)

        actions[eid] = {
            "nn_action": nn_action,
            "pred_delta": pred_delta,
            "pred_delta_conf": pred_delta_conf,
            "pred_future_hit": 1 if pred_future_hit_prob >= future_threshold else 0,
            "pred_bypass": 1 if pred_bypass_prob >= bypass_threshold else 0,
            "emit_prefetch": 1 if emit_prefetch else 0,
        }

    return actions, counts, malformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path,
                    help="Large ground-truth event CSV: lstm_events_<trace>.csv")
    ap.add_argument("--actions", required=True, type=Path,
                    help="LSTM action CSV: full_lstm_cache_actions.csv")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--policy", choices=["action", "threshold"], default="action")
    ap.add_argument("--prefetch-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-threshold", type=float, default=0.60)
    ap.add_argument("--future-threshold", type=float, default=0.50)
    ap.add_argument("--bypass-target", choices=["not_useful", "not_useful_or_duplicate"], default="not_useful_or_duplicate")
    args = ap.parse_args()

    if not args.events.exists():
        raise SystemExit("[error] missing events CSV: {}".format(args.events))
    if not args.actions.exists():
        raise SystemExit("[error] missing actions CSV: {}".format(args.actions))

    print("[load actions] {}".format(args.actions))
    actions, action_counts, malformed_actions = parse_actions(
        args.actions,
        args.policy,
        args.prefetch_threshold,
        args.bypass_threshold,
        args.future_threshold,
    )
    print("[actions] parsed={} malformed={}".format(len(actions), malformed_actions))
    print("[actions] counts={}".format(action_counts))

    total_events = 0
    total_useful = 0
    total_duplicate = 0

    spp_issued = 0
    spp_useful = 0
    spp_duplicate = 0

    joined = 0
    missing_events_for_actions = set(actions.keys())

    lstm_emit = 0
    lstm_useful = 0
    lstm_duplicate = 0

    delta_total = 0
    delta_correct = 0

    # future-hit confusion matrix
    fh_tp = fh_fp = fh_tn = fh_fn = 0

    # bypass confusion matrix
    by_tp = by_fp = by_tn = by_fn = 0

    print("[stream events] {}".format(args.events))
    reader = csv.DictReader(clean_text_lines(args.events))
    for row in reader:
        total_events += 1
        eid = to_int(row.get("event_id"), -1)
        useful = 1 if to_int(row.get("outcome_useful"), 0) != 0 else 0
        duplicate = 1 if to_int(row.get("outcome_duplicate"), 0) != 0 else 0
        spp = 1 if to_int(row.get("spp_issued"), 0) != 0 else 0
        true_delta = to_int(row.get("delta"), 0)

        total_useful += useful
        total_duplicate += duplicate
        if spp:
            spp_issued += 1
            spp_useful += useful
            spp_duplicate += duplicate

        pred = actions.get(eid)
        if pred is None:
            continue

        joined += 1
        if eid in missing_events_for_actions:
            missing_events_for_actions.remove(eid)

        pred_fh = pred["pred_future_hit"]
        if pred_fh and useful:
            fh_tp += 1
        elif pred_fh and not useful:
            fh_fp += 1
        elif (not pred_fh) and (not useful):
            fh_tn += 1
        else:
            fh_fn += 1

        if args.bypass_target == "not_useful_or_duplicate":
            true_bypass = 1 if ((not useful) or duplicate) else 0
        else:
            true_bypass = 1 if not useful else 0
        pred_bypass = pred["pred_bypass"]
        if pred_bypass and true_bypass:
            by_tp += 1
        elif pred_bypass and not true_bypass:
            by_fp += 1
        elif (not pred_bypass) and (not true_bypass):
            by_tn += 1
        else:
            by_fn += 1

        if pred["emit_prefetch"]:
            lstm_emit += 1
            lstm_useful += useful
            lstm_duplicate += duplicate
            delta_total += 1
            if pred["pred_delta"] == true_delta:
                delta_correct += 1

    fh_acc, fh_precision, fh_recall, fh_f1 = cls_metrics(fh_tp, fh_fp, fh_tn, fh_fn)
    by_acc, by_precision, by_recall, by_f1 = cls_metrics(by_tp, by_fp, by_tn, by_fn)
    lstm_precision, lstm_recall, lstm_f1 = prf(lstm_useful, lstm_emit - lstm_useful, total_useful - lstm_useful)
    spp_precision, spp_recall, spp_f1 = prf(spp_useful, spp_issued - spp_useful, total_useful - spp_useful)

    rows = [
        ["metric", "value"],
        ["events_total", total_events],
        ["events_useful", total_useful],
        ["events_useful_rate", "{:.8f}".format(div(total_useful, total_events))],
        ["events_duplicate", total_duplicate],
        ["joined_action_rows", joined],
        ["action_rows_missing_event", len(missing_events_for_actions)],
        ["policy", args.policy],
        ["prefetch_threshold", args.prefetch_threshold],
        ["bypass_threshold", args.bypass_threshold],
        ["future_threshold", args.future_threshold],
        ["spp_issued", spp_issued],
        ["spp_useful", spp_useful],
        ["spp_precision_accuracy", "{:.8f}".format(spp_precision)],
        ["spp_recall_useful_coverage", "{:.8f}".format(spp_recall)],
        ["spp_f1", "{:.8f}".format(spp_f1)],
        ["spp_duplicate", spp_duplicate],
        ["spp_duplicate_rate", "{:.8f}".format(div(spp_duplicate, spp_issued))],
        ["lstm_prefetch_emitted", lstm_emit],
        ["lstm_prefetch_useful", lstm_useful],
        ["lstm_prefetch_precision_accuracy", "{:.8f}".format(lstm_precision)],
        ["lstm_prefetch_recall_useful_coverage", "{:.8f}".format(lstm_recall)],
        ["lstm_prefetch_f1", "{:.8f}".format(lstm_f1)],
        ["lstm_duplicate", lstm_duplicate],
        ["lstm_duplicate_rate", "{:.8f}".format(div(lstm_duplicate, lstm_emit))],
        ["lstm_delta_top1_on_emitted", "{:.8f}".format(div(delta_correct, delta_total))],
        ["future_hit_accuracy", "{:.8f}".format(fh_acc)],
        ["future_hit_precision", "{:.8f}".format(fh_precision)],
        ["future_hit_recall", "{:.8f}".format(fh_recall)],
        ["future_hit_f1", "{:.8f}".format(fh_f1)],
        ["future_hit_tp", fh_tp],
        ["future_hit_fp", fh_fp],
        ["future_hit_tn", fh_tn],
        ["future_hit_fn", fh_fn],
        ["bypass_accuracy", "{:.8f}".format(by_acc)],
        ["bypass_precision", "{:.8f}".format(by_precision)],
        ["bypass_recall", "{:.8f}".format(by_recall)],
        ["bypass_f1", "{:.8f}".format(by_f1)],
        ["bypass_tp", by_tp],
        ["bypass_fp", by_fp],
        ["bypass_tn", by_tn],
        ["bypass_fn", by_fn],
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\n===== Accuracy summary =====")
    for k, v in rows[1:]:
        print("{:<36} {}".format(k, v))
    print("\n[done] wrote {}".format(args.out))


if __name__ == "__main__":
    main()
