#!/usr/bin/env python3
"""Train a 602 AMPM-matched LSTM gate and export offline AMPM/LSTM lists.

Both policies receive only the causal L2 load address stream.  The LSTM sees
the same 64-page LRU state and page-offset bitmap used by AMPM; PC remains only
in the keyed replay transport and never becomes a model feature.
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import platform
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRACE = "602.gcc_s-734B"
TRACKERS = 64
PAGE_LINES = 64
PRED_DEGREE = 4
MAX_DELTA = 16
RUNTIME_FEATURES = PAGE_LINES + 2  # bitmap, current offset, tracker-hit bit
CANDIDATE_FEATURES = 3  # signed delta, target offset, AMPM rank


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def load_stream(path):
    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"trace", "demand_idx", "pc", "line", "pc_line_occ"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            if row["trace"] != TRACE or as_int(row["demand_idx"]) != index:
                raise RuntimeError("stream identity/ordering failure at row {}".format(index))
            rows.append((as_int(row["pc"]), as_int(row["line"]), as_int(row["pc_line_occ"])))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def ampm_arrays(rows, state=None):
    """Mirror pinned AMPM: 64 LRU pages, 64-bit page bitmap, degree 4.

    The bitmap is updated with the current offset before candidate discovery,
    exactly as in ``prefetcher/ampm.cc``.  This function returns the final page
    state so the 20M--25M guard can causally initialize the measured stream.
    """
    count = len(rows)
    runtime = np.zeros((count, RUNTIME_FEATURES), dtype=np.float32)
    candidate = np.zeros((count, PRED_DEGREE, CANDIDATE_FEATURES), dtype=np.float32)
    valid = np.zeros((count, PRED_DEGREE), dtype=np.bool_)
    target = np.zeros((count, PRED_DEGREE), dtype=np.int64)
    pages = OrderedDict() if state is None else state

    for index, (unused_pc, line, unused_occ) in enumerate(rows):
        del unused_pc, unused_occ
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        was_tracked = page in pages
        if was_tracked:
            bitmap = pages.pop(page)
        else:
            if len(pages) >= TRACKERS:
                pages.popitem(last=False)
            bitmap = np.zeros(PAGE_LINES, dtype=np.bool_)
        bitmap[offset] = True
        pages[page] = bitmap

        runtime[index, :PAGE_LINES] = bitmap
        runtime[index, PAGE_LINES] = offset / float(PAGE_LINES - 1)
        runtime[index, PAGE_LINES + 1] = float(was_tracked)

        selected = []
        # AMPM prioritizes positive deltas, each scanned largest to smallest.
        # The C++ implementation does candidate-pattern discovery first, then
        # silently drops an address whose final page offset would be out of
        # range.  Keep scanning after such a drop; do not consume a degree
        # slot for it.
        for delta in range(MAX_DELTA, 0, -1):
            if len(selected) >= PRED_DEGREE:
                break
            one_hop = offset - delta
            two_hop = offset - 2 * delta
            if one_hop >= 0 and two_hop >= 0 and bitmap[one_hop] and bitmap[two_hop]:
                target_offset = offset + delta
                if target_offset < PAGE_LINES:
                    selected.append((delta, target_offset))
        for delta in range(MAX_DELTA, 0, -1):
            if len(selected) >= PRED_DEGREE:
                break
            one_hop = offset + delta
            two_hop = offset + 2 * delta
            if one_hop < PAGE_LINES and two_hop < PAGE_LINES and bitmap[one_hop] and bitmap[two_hop]:
                target_offset = offset - delta
                if target_offset >= 0:
                    selected.append((-delta, target_offset))

        for slot, (signed_delta, target_offset) in enumerate(selected):
            valid[index, slot] = True
            target[index, slot] = page * PAGE_LINES + target_offset
            candidate[index, slot] = [
                signed_delta / float(MAX_DELTA),
                target_offset / float(PAGE_LINES - 1),
                (slot + 1) / float(PRED_DEGREE),
            ]

    return runtime, candidate, valid, target, pages


def self_test_ampm_policy():
    page = 11 * PAGE_LINES
    positive_rows = [(1, page + x, 0) for x in (1, 2, 3)]
    _, _, valid, target, _ = ampm_arrays(positive_rows)
    assert not bool(valid[0].any())
    assert not bool(valid[1].any())
    assert target[2][valid[2]].tolist() == [page + 4]

    negative_rows = [(1, page + x, 0) for x in (10, 9, 8)]
    _, _, valid, target, _ = ampm_arrays(negative_rows)
    assert target[2][valid[2]].tolist() == [page + 7]

    # A valid two-hop pattern can point outside the current page.  Pinned
    # AMPM detects that pattern but then drops the final out-of-page address.
    boundary_rows = [(1, page + x, 0) for x in (18, 34, 50)]
    _, _, valid, _, _ = ampm_arrays(boundary_rows)
    assert not bool(valid[2].any())

    seed_rows = [(1, page + x, 0) for x in (1, 2, 3)]
    _, _, _, _, state = ampm_arrays(seed_rows)
    _, _, valid, target, _ = ampm_arrays([(1, page + 4, 0)], state)
    assert target[0][valid[0]].tolist() == [page + 5]


def training_labels(rows, valid, target, min_lead, max_lead, fit_end):
    positions = defaultdict(list)
    for index, (_, line, _) in enumerate(rows):
        positions[line].append(index)
    labels = np.zeros(valid.shape, dtype=np.float32)

    for index in range(len(rows)):
        split_end = fit_end if index < fit_end else len(rows)
        if index + min_lead >= split_end:
            continue
        latest = min(index + max_lead, split_end - 1)
        for slot in np.flatnonzero(valid[index]):
            values = positions[int(target[index, slot])]
            pos = bisect.bisect_left(values, index + min_lead)
            if pos < len(values) and values[pos] <= latest:
                labels[index, slot] = 1.0
    return labels


class AMPMGateLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(RUNTIME_FEATURES, hidden_size, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + CANDIDATE_FEATURES, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, runtime, candidate, state=None):
        hidden, state = self.lstm(runtime, state)
        expanded = hidden.unsqueeze(2).expand(-1, -1, candidate.shape[2], -1)
        joined = torch.cat([expanded, candidate], dim=-1)
        logits = self.head(self.projection(joined)).squeeze(-1)
        return logits, state


def parameter_count(hidden_size):
    # LSTM(66,h) + Linear(h+3,h) + Linear(h,1).
    return 5 * hidden_size * hidden_size + 277 * hidden_size + 1


def chunk_tensors(runtime, candidate, labels, valid, end, chunk_len):
    usable = (end // chunk_len) * chunk_len
    count = usable // chunk_len
    if count == 0:
        raise RuntimeError("training stream is shorter than one chunk")
    return (
        torch.from_numpy(runtime[:usable]).reshape(count, chunk_len, RUNTIME_FEATURES),
        torch.from_numpy(candidate[:usable]).reshape(count, chunk_len, PRED_DEGREE, CANDIDATE_FEATURES),
        torch.from_numpy(labels[:usable]).reshape(count, chunk_len, PRED_DEGREE),
        torch.from_numpy(valid[:usable]).reshape(count, chunk_len, PRED_DEGREE),
    )


def train_model(model, runtime, candidate, labels, valid, fit_end, device, epochs, chunk_len, batch_chunks, learning_rate):
    x, c, y, mask = chunk_tensors(runtime, candidate, labels, valid, fit_end, chunk_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        generator = torch.Generator().manual_seed(1000 + epoch)
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.0
        total_valid = 0
        model.train()
        for start in range(0, len(order), batch_chunks):
            indices = order[start : start + batch_chunks]
            xb = x[indices].to(device)
            cb = c[indices].to(device)
            yb = y[indices].to(device)
            mb = mask[indices].to(device)
            if not bool(mb.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb, cb)
            loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            selected = int(mb.sum().item())
            total_loss += float(loss.item()) * selected
            total_valid += selected
        row = {"epoch": epoch, "loss": total_loss / max(1, total_valid), "valid_candidates": total_valid}
        history.append(row)
        print("[train] epoch={epoch} loss={loss:.6f} valid={valid_candidates}".format(**row))
    return history


def score_continuous(model, runtime, candidate, device, initial_state=None, chunk_len=8192):
    model.eval()
    scores = np.zeros((len(runtime), PRED_DEGREE), dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start in range(0, len(runtime), chunk_len):
            stop = min(len(runtime), start + chunk_len)
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores, state


def metrics(selected, labels, valid):
    chosen = selected & valid
    issued = int(chosen.sum())
    useful = int((chosen & (labels > 0.5)).sum())
    positives = int(((labels > 0.5) & valid).sum())
    precision = useful / float(issued) if issued else 0.0
    recall = useful / float(positives) if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return issued, useful, precision, recall, f1


def calibrate(scores, labels, valid, start):
    finite = scores[start:][valid[start:]]
    if finite.size == 0:
        raise RuntimeError("calibration split has no AMPM candidates")
    thresholds = list(np.linspace(0.05, 0.95, 19))
    thresholds += [float(np.quantile(finite, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)]
    events = max(1, len(valid) - start)
    budget = float(valid[start:].sum()) / events
    rows = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = scores[start:] >= threshold
        issued, useful, precision, recall, f1 = metrics(selected, labels[start:], valid[start:])
        rate = issued / float(events)
        rows.append({
            "threshold": threshold,
            "issued": issued,
            "useful": useful,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidates_per_event": rate,
            "offline_ampm_candidates_per_event_budget": budget,
            "within_budget": int(rate <= budget + 1.0 / events),
        })
    eligible = [row for row in rows if row["within_budget"]]
    if not eligible:
        raise RuntimeError("no threshold satisfies the offline AMPM budget")
    eligible.sort(key=lambda row: (-row["f1"], -row["precision"], -row["recall"], row["candidates_per_event"]))
    return eligible[0], rows


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_replay(path, rows, selected, valid, target):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for index, (pc, line, occ) in enumerate(rows):
            slots = np.flatnonzero(selected[index] & valid[index])
            if slots.size:
                triggers += 1
            for slot in slots:
                writer.writerow([pc, line, occ, "0x{:x}".format(int(target[index, slot]) * 64)])
                entries += 1
    return entries, triggers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    parser.add_argument("--guard-stream", required=True, type=Path)
    parser.add_argument("--eval-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=16)
    args = parser.parse_args()
    if args.hidden_size < 1:
        raise RuntimeError("--hidden-size must be positive")
    self_test_ampm_policy()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_stream(args.train_stream)
    guard_rows = load_stream(args.guard_stream)
    eval_rows = load_stream(args.eval_stream)
    train_runtime, train_candidate, train_valid, train_target, _ = ampm_arrays(train_rows)
    fit_end = int(0.8 * len(train_rows))
    labels = training_labels(train_rows, train_valid, train_target, args.min_lead, args.max_lead, fit_end)

    model = AMPMGateLSTM(args.hidden_size)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != parameter_count(args.hidden_size):
        raise RuntimeError("parameter-count formula mismatch: {} != {}".format(parameters, parameter_count(args.hidden_size)))
    history = train_model(model, train_runtime, train_candidate, labels, train_valid, fit_end, device, args.epochs, args.chunk_len, args.batch_chunks, args.learning_rate)
    train_scores, _ = score_continuous(model, train_runtime, train_candidate, device)
    policy, sweep = calibrate(train_scores, labels, train_valid, fit_end)

    guard_runtime, guard_candidate, _, _, page_state = ampm_arrays(guard_rows)
    eval_runtime, eval_candidate, eval_valid, eval_target, _ = ampm_arrays(eval_rows, page_state)
    _, guard_lstm_state = score_continuous(model, guard_runtime, guard_candidate, device)
    eval_scores, _ = score_continuous(model, eval_runtime, eval_candidate, device, guard_lstm_state)
    lstm_selected = eval_scores >= policy["threshold"]
    ampm_selected = eval_valid.copy()

    ampm_entries, ampm_triggers = write_replay(args.out_dir / "offline_ampm.replay.csv", eval_rows, ampm_selected, eval_valid, eval_target)
    lstm_entries, lstm_triggers = write_replay(args.out_dir / "offline_lstm.replay.csv", eval_rows, lstm_selected, eval_valid, eval_target)
    torch.save({"state_dict": model.cpu().state_dict(), "parameters": parameters, "hidden_size": args.hidden_size, "trace": TRACE, "matched_normal_prefetcher": "ampm"}, args.out_dir / "model.pt")
    write_table(args.out_dir / "training_history.csv", history)
    write_table(args.out_dir / "policy_sweep.csv", sweep)

    metadata = {
        "trace": TRACE,
        "model_family": "LSTM candidate gate",
        "matched_normal_prefetcher": "ampm",
        "parameter_count": parameters,
        "hidden_size": args.hidden_size,
        "parameter_count_formula": "5*h^2 + 277*h + 1",
        "seed": args.seed,
        "ampm_pb_size": TRACKERS,
        "ampm_pred_degree": PRED_DEGREE,
        "ampm_pref_degree": PRED_DEGREE,
        "ampm_pref_buffer_enabled": False,
        "ampm_max_delta": MAX_DELTA,
        "training_and_inference_location": "Colab_or_any_PyTorch_host",
        "inference_mode": "offline_causal_list_generation",
        "primary_methods": ["offline_ampm", "offline_lstm_ampm_gate"],
        "shared_eval_raw_input": ["current_cache_line_address", "causal_prior_address_history"],
        "runtime_features_derived_from_shared_address_stream": ["64_page_offset_bitmap_after_current_access", "current_page_offset", "page_tracker_hit", "64_page_lru_tracker_state"],
        "candidate_features_derived_from_ampm_candidates": ["signed_delta_within_16_lines", "candidate_page_offset", "AMPM_candidate_rank_1_to_4"],
        "model_does_not_use_pc": True,
        "pc_line_occ_role": "replay_transport_identity_only",
        "forbidden_inputs": ["hit_miss", "cycle", "queue_state", "metadata", "future_evaluation_rows"],
        "training_labels": "future addresses only inside each chronological training/calibration split of the disjoint 0_to_20M training stream",
        "guard_stream_role": "causal 20M_to_25M state initialization only; never used for fitting, calibration, or replay output",
        "evaluation_stream_role": "causal inference only; never used for fitting or threshold calibration",
        "transport": "same keyed PC-line-occ ListReplayer for offline AMPM and LSTM",
        "train_stream_sha256": sha256(args.train_stream),
        "guard_stream_sha256": sha256(args.guard_stream),
        "eval_stream_sha256": sha256(args.eval_stream),
        "train_stream_content_sha256": gzip_content_sha256(args.train_stream),
        "guard_stream_content_sha256": gzip_content_sha256(args.guard_stream),
        "eval_stream_content_sha256": gzip_content_sha256(args.eval_stream),
        "train_rows": len(train_rows),
        "fit_rows": fit_end,
        "calibration_rows": len(train_rows) - fit_end,
        "guard_rows": len(guard_rows),
        "eval_rows": len(eval_rows),
        "threshold": policy["threshold"],
        "offline_ampm_entries": ampm_entries,
        "offline_ampm_triggers": ampm_triggers,
        "offline_lstm_entries": lstm_entries,
        "offline_lstm_triggers": lstm_triggers,
        "offline_ampm_list_sha256": sha256(args.out_dir / "offline_ampm.replay.csv"),
        "offline_lstm_list_sha256": sha256(args.out_dir / "offline_lstm.replay.csv"),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ampm_policy_self_test": "PASS",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[ok] " + json.dumps({"device": str(device), "hidden_size": args.hidden_size, "parameters": parameters, "threshold": policy["threshold"], "ampm_entries": ampm_entries, "lstm_entries": lstm_entries}, sort_keys=True))


if __name__ == "__main__":
    main()
