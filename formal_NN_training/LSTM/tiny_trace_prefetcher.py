#!/usr/bin/env python3
"""Trace-specialized tiny LSTM prefetcher.

This module is deliberately a Colab-side training/export component, not a
Sacramento simulator driver.  It trains from the raw standalone no-prefetch
oracle, produces a hardware-bounded candidate bank, exports a complete decision
ledger, and writes a rich prefetch list that is consumed by scripts/07 and the
keyed ListReplayer.

The trainable model has exactly 742 parameters:
  * LSTM(8 input features, 8 hidden units): 576
  * candidate projection Linear(13, 8): 112
  * utility head Linear(8, 1): 9
  * lead-bin head Linear(8, 5): 45
No trainable address/PC embedding table is used.  Candidate tables are
nonparametric bounded metadata and are reported separately from NN parameters.
"""
from __future__ import print_function

import bisect
import csv
import gzip
import json
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LINE_BYTES = 64
LEAD_BINS = np.asarray([4, 8, 16, 32, 64], dtype=np.int64)
TINY_VERSION = "tiny_trace_lstm_07_05"

# A trace chooses the candidate representation and policy objective, not a
# different neural-network size. This keeps the experiment comparable.
TRACE_PROFILES = {
    "602.gcc_s-734B": {
        "short_name": "602_gcc",
        "goal": "reproduce_strong_normal",
        "normal_reference": "sandbox",
        "sources": [("assoc", 2), ("pc_delta", 1), ("global", 1)],
        "min_lead": 4, "max_lead": 64,
        "dedup_capacity": 256, "min_precision": 0.80,
        "table_capacity": 16384,
        "rationale": "Sandbox is the strong normal reference. Use a conservative one/few-action association policy and measure remaining coverage/timeliness gap rather than optimizing for raw issue volume.",
    },
    "605.mcf_s-994B": {
        "short_name": "605_mcf",
        "goal": "exceed_weak_normal",
        "normal_reference": "ampm",
        "sources": [("dependency", 2), ("pc_delta", 1), ("global", 1)],
        "min_lead": 4, "max_lead": 128,
        "dedup_capacity": 256, "min_precision": 0.25,
        "table_capacity": 16384,
        "rationale": "AMPM is weak for this irregular/pointer-like workload. Dependency edge candidates are used as an explicit semantic proposal source; the tiny LSTM gates issue timing.",
    },
    "619.lbm_s-4268B": {
        "short_name": "619_lbm",
        "goal": "reproduce_strong_normal",
        "normal_reference": "sms",
        "sources": [("pc_delta", 2), ("pc_prev_delta", 1), ("global", 1)],
        "min_lead": 8, "max_lead": 128,
        "dedup_capacity": 0, "min_precision": 0.80,
        "table_capacity": 8192,
        "rationale": "SMS is already strong. The policy emphasizes lead-aware useful candidates; the smallest lead bin is eight events to avoid the v3.9 late-prefetch failure.",
    },
    "620.omnetpp_s-874B": {
        "short_name": "620_omnetpp",
        "goal": "reproduce_strong_normal",
        "normal_reference": "sms",
        "sources": [("region_pair", 2), ("predecessor", 1), ("global", 1)],
        "min_lead": 4, "max_lead": 128,
        "dedup_capacity": 256, "min_precision": 0.70,
        "table_capacity": 16384,
        "rationale": "SMS is the normal reference. Region-pair and predecessor contexts target indirect behavior while the policy penalizes broad low-value coverage.",
    },
    "623.xalancbmk_s-700B": {
        "short_name": "623_xalancbmk",
        "goal": "exceed_weak_normal",
        "normal_reference": "spp",
        "sources": [("context3", 2), ("phase_offset", 1), ("global", 1)],
        "min_lead": 4, "max_lead": 64,
        "dedup_capacity": 256, "min_precision": 0.55,
        "table_capacity": 8192,
        "rationale": "SPP is almost neutral on this trace. Context/phase proposals retain the successful v3.3 idea, but the model is a 742-parameter LSTM rather than a multi-million-parameter network.",
    },
}


def _as_int(value, default=0):
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def _open_csv(path, mode="rt"):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="")


def _safe_rel(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _csv_write(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stable_unit(value, modulus=4096):
    return float(int(value) % modulus) / float(modulus - 1)


def _clip_scale(value, cap):
    value = max(-cap, min(cap, int(value)))
    return float(value) / float(cap)


def _log_scale(value, cap):
    return min(1.0, math.log1p(max(0, int(value))) / math.log1p(cap))


def _lead_bin(lead):
    if lead <= 0:
        return 0
    index = int(np.searchsorted(LEAD_BINS, int(lead), side="right") - 1)
    return max(0, min(index, len(LEAD_BINS) - 1))


def _choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyCandidateLSTM(nn.Module):
    """742-parameter causal LSTM scorer."""

    def __init__(self):
        super(TinyCandidateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, batch_first=True)
        self.candidate_projection = nn.Sequential(nn.Linear(13, 8), nn.Tanh())
        self.utility_head = nn.Linear(8, 1)
        self.lead_head = nn.Linear(8, len(LEAD_BINS))

    def forward(self, runtime_features, candidate_features, state=None):
        hidden, state = self.lstm(runtime_features, state)
        batch, time_steps, slots, _ = candidate_features.shape
        h = hidden.unsqueeze(2).expand(batch, time_steps, slots, hidden.size(-1))
        combined = torch.cat([h, candidate_features], dim=-1)
        combined = self.candidate_projection(combined)
        utility = self.utility_head(combined).squeeze(-1)
        lead = self.lead_head(combined)
        return utility, lead, state


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def assert_tiny_model(model):
    observed = parameter_count(model)
    if observed != 742:
        raise RuntimeError("Tiny model parameter-count regression: got {}, expected 742".format(observed))
    if observed >= 1000:
        raise RuntimeError("Tiny model violates <1000 parameter requirement")


def read_oracle(oracle_path, max_rows=0):
    """Read stable standalone raw-stream oracle; never read normal-prefetch output."""
    records = []
    with _open_csv(oracle_path) as handle:
        reader = csv.DictReader(handle)
        required = {"demand_idx", "cycle", "pc", "line", "addr", "page_offset", "delta", "no_pref_miss", "pc_line_occ"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("oracle missing columns: {}".format(sorted(missing)))
        expected_idx = 0
        for raw in reader:
            idx = _as_int(raw.get("demand_idx"))
            if idx != expected_idx:
                raise ValueError("oracle demand_idx must be contiguous from zero")
            records.append({
                "demand_idx": idx,
                "cycle": _as_int(raw.get("cycle")),
                "pc": _as_int(raw.get("pc")),
                "line": _as_int(raw.get("line")),
                "addr": _as_int(raw.get("addr")),
                "page_offset": _as_int(raw.get("page_offset")),
                "delta": _as_int(raw.get("delta")),
                "no_pref_miss": int(_as_int(raw.get("no_pref_miss")) != 0),
                "pc_line_occ": _as_int(raw.get("pc_line_occ")),
            })
            expected_idx += 1
            if max_rows and expected_idx >= int(max_rows):
                break
    if len(records) < 4096:
        raise RuntimeError("oracle has too few rows for train/validation: {}".format(len(records)))
    return records


def _load_dependency_profile(repo_root, trace):
    path = Path(repo_root) / "formal_NN_training/data/upload/v3_9_dependency_profiles/{}.v3_9_dependency_profile.csv.gz".format(trace)
    if not path.is_file():
        return {}, path
    output = {}
    with _open_csv(path) as handle:
        for raw in csv.DictReader(handle):
            pc = _as_int(raw.get("pc"))
            support = _as_int(raw.get("parent_is_load_ppm"), 0)
            observations = _as_int(raw.get("dependency_observations"), 0)
            output[pc] = max(float(support) / 1000000.0, min(1.0, math.log1p(observations) / math.log1p(4096)))
    return output, path


def _load_dependency_edges(repo_root, trace):
    path = Path(repo_root) / "formal_NN_training/data/upload/v3_9_dependency_profiles/{}.v3_9_dependency_edge_vocab.csv.gz".format(trace)
    by_pc = defaultdict(list)
    if not path.is_file():
        return by_pc, path
    with _open_csv(path) as handle:
        for raw in csv.DictReader(handle):
            pc = _as_int(raw.get("producer_pc"))
            delta = _as_int(raw.get("producer_to_target_line_delta"))
            support = max(_as_int(raw.get("support_lower_bound")), _as_int(raw.get("estimated_support")))
            if delta:
                by_pc[pc].append((delta, support))
    for pc in by_pc:
        by_pc[pc].sort(key=lambda item: (-item[1], abs(item[0]), item[0]))
    return by_pc, path


def make_runtime_features(records, dependency_feature):
    """Eight strictly causal runtime features from raw no-prefetch history."""
    total = len(records)
    features = np.zeros((total, 8), dtype=np.float32)
    last_pc = {}
    previous_cycle = 0
    previous_delta = 0
    miss_prefix = 0
    for index, row in enumerate(records):
        pc = row["pc"]
        features[index, 0] = _stable_unit(pc)
        features[index, 1] = _clip_scale(row["delta"], 256)
        features[index, 2] = _clip_scale(previous_delta, 256)
        features[index, 3] = float(row["page_offset"] % 64) / 63.0
        features[index, 4] = _log_scale(index - last_pc.get(pc, index), 4096)
        features[index, 5] = _log_scale(row["cycle"] - previous_cycle, 1 << 20)
        features[index, 6] = float(miss_prefix) / float(min(128, max(1, index)))
        features[index, 7] = float(dependency_feature.get(pc, 0.0))
        last_pc[pc] = index
        previous_cycle = row["cycle"]
        previous_delta = row["delta"]
        miss_prefix += row["no_pref_miss"]
        if index >= 128:
            miss_prefix -= records[index - 128]["no_pref_miss"]
    return features


def _context_key(source, records, index, previous_pc):
    row = records[index]
    pc = row["pc"]
    delta = int(row["delta"])
    prev_delta = int(records[index - 1]["delta"]) if index else 0
    if source == "assoc":
        return (pc, row["line"])
    if source == "pc_delta":
        return (pc, max(-64, min(64, delta)))
    if source == "pc_prev_delta":
        return (pc, max(-64, min(64, delta)), max(-64, min(64, prev_delta)))
    if source == "region_pair":
        return (pc, (row["line"] // 64) & 0xF, row["page_offset"] // 8)
    if source == "predecessor":
        return (pc, previous_pc)
    if source == "context3":
        return (pc, max(-32, min(32, delta)), max(-32, min(32, prev_delta)))
    if source == "phase_offset":
        return (pc, (index // 64) & 0xF, row["page_offset"] // 8)
    if source == "global":
        return (0,)
    raise ValueError("unknown candidate source: {}".format(source))


def _top_contexts(records, source, train_end, capacity):
    counter = Counter()
    previous_pc = 0
    for index in range(train_end):
        counter[_context_key(source, records, index, previous_pc)] += 1
        previous_pc = records[index]["pc"]
    return set(key for key, _ in counter.most_common(int(capacity)))


def build_candidate_tables(records, profile, train_end, repo_root, trace):
    """Build bounded candidate tables only from the chronological training prefix."""
    dynamic_sources = [source for source, _ in profile["sources"] if source != "dependency"]
    tables, allowed = {}, {}
    for source in dynamic_sources:
        allowed[source] = {(0,)} if source == "global" else _top_contexts(records, source, train_end, profile["table_capacity"])
        tables[source] = defaultdict(Counter)

    horizons = [lead for lead in LEAD_BINS.tolist() if lead <= profile["max_lead"]]
    for source in dynamic_sources:
        previous_pc = 0
        table = tables[source]
        for index in range(train_end):
            key = _context_key(source, records, index, previous_pc)
            previous_pc = records[index]["pc"]
            if key not in allowed[source]:
                continue
            base = records[index]["line"]
            for lead in horizons:
                target_index = index + lead
                if target_index >= train_end:
                    continue
                target = records[target_index]
                if not target["no_pref_miss"]:
                    continue
                delta = int(target["line"] - base)
                if delta and abs(delta) <= 4096:
                    table[key][delta] += 1

    compact = {}
    for source, table in tables.items():
        compact[source] = {}
        for key, counter in table.items():
            values = counter.most_common(8)
            if values:
                compact[source][key] = values
    dependency, dependency_path = _load_dependency_edges(repo_root, trace)
    if "dependency" in [source for source, _ in profile["sources"]] and not dependency:
        raise FileNotFoundError("605 tiny notebook requires committed dependency edge vocabulary: {}".format(dependency_path))
    return compact, dependency, {
        "dynamic_sources": dynamic_sources,
        "context_capacity": int(profile["table_capacity"]),
        "dependency_edge_path": str(dependency_path),
        "dependency_edge_pcs": int(len(dependency)),
        "contexts_retained": {source: len(compact.get(source, {})) for source in dynamic_sources},
    }


def make_candidates(records, profile, tables, dependency_edges):
    """Materialize <=4 deterministic proposal slots per event from train-only tables."""
    slots = int(sum(count for _, count in profile["sources"]))
    total = len(records)
    deltas = np.zeros((total, slots), dtype=np.int64)
    supports = np.zeros((total, slots), dtype=np.float32)
    source_ids = np.zeros((total, slots), dtype=np.float32)
    valid = np.zeros((total, slots), dtype=np.uint8)
    source_names = np.empty((total, slots), dtype=object)
    source_to_id = {"assoc": 0, "pc_delta": 1, "pc_prev_delta": 2, "region_pair": 3, "predecessor": 4, "context3": 5, "phase_offset": 6, "global": 7, "dependency": 8}
    previous_pc = 0
    for index, row in enumerate(records):
        chosen, cursor = set(), 0
        for source, quota in profile["sources"]:
            if source == "dependency":
                values = dependency_edges.get(row["pc"], [])
            else:
                key = _context_key(source, records, index, previous_pc)
                values = tables.get(source, {}).get(key, [])
                if not values and source != "global":
                    values = tables.get("global", {}).get((0,), [])
            taken = 0
            for delta, support in values:
                delta = int(delta)
                if not delta or abs(delta) > 4096 or delta in chosen or cursor >= slots:
                    continue
                chosen.add(delta)
                deltas[index, cursor] = delta
                supports[index, cursor] = min(1.0, math.log1p(float(support)) / math.log1p(4096.0))
                source_ids[index, cursor] = float(source_to_id[source]) / 8.0
                valid[index, cursor] = 1
                source_names[index, cursor] = source
                cursor += 1
                taken += 1
                if taken >= int(quota):
                    break
        previous_pc = row["pc"]
    return deltas, supports, source_ids, valid, source_names


def make_candidate_features(deltas, supports, source_ids, valid):
    total, slots = deltas.shape
    output = np.zeros((total, slots, 5), dtype=np.float32)
    output[:, :, 0] = np.clip(deltas, -4096, 4096).astype(np.float32) / 4096.0
    output[:, :, 1] = supports
    output[:, :, 2] = source_ids
    for rank in range(slots):
        output[:, rank, 3] = float(rank) / float(max(1, slots - 1))
    output[:, :, 4] = valid.astype(np.float32)
    return output


def make_labels(records, deltas, valid, profile):
    """Positive iff a candidate is a future no-prefetch miss inside the lead window."""
    misses_by_line = defaultdict(list)
    for index, row in enumerate(records):
        if row["no_pref_miss"]:
            misses_by_line[row["line"]].append(index)
    total, slots = deltas.shape
    utility = np.zeros((total, slots), dtype=np.float32)
    lead_class = np.zeros((total, slots), dtype=np.int64)
    min_lead, max_lead = int(profile["min_lead"]), int(profile["max_lead"])
    for index, row in enumerate(records):
        lower, upper = index + min_lead, index + max_lead
        base = row["line"]
        for slot in range(slots):
            if not valid[index, slot]:
                continue
            targets = misses_by_line.get(base + int(deltas[index, slot]), [])
            pos = bisect.bisect_left(targets, lower)
            if pos < len(targets) and targets[pos] <= upper:
                utility[index, slot] = 1.0
                lead_class[index, slot] = _lead_bin(targets[pos] - index)
    return utility, lead_class


def _predict_sequence(model, features, candidate_features, device, chunk_len):
    model.eval()
    total, slots = len(features), candidate_features.shape[1]
    utility = np.zeros((total, slots), dtype=np.float32)
    lead_logits = np.zeros((total, slots, len(LEAD_BINS)), dtype=np.float32)
    state = None
    with torch.no_grad():
        for start in range(0, total, int(chunk_len)):
            end = min(total, start + int(chunk_len))
            x = torch.from_numpy(features[start:end]).to(device).unsqueeze(0)
            c = torch.from_numpy(candidate_features[start:end]).to(device).unsqueeze(0)
            scores, leads, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            utility[start:end] = torch.sigmoid(scores[0]).cpu().numpy()
            lead_logits[start:end] = leads[0].cpu().numpy()
    return utility, lead_logits


def _fit_model(model, features, candidate_features, labels, lead_classes, valid, train_end, device, epochs, chunk_len, learning_rate):
    model.to(device)
    assert_tiny_model(model)
    train_positive = float(np.sum(labels[:train_end][valid[:train_end].astype(bool)]))
    train_total = float(np.sum(valid[:train_end]))
    if train_positive <= 0.0:
        raise RuntimeError("candidate bank has zero train-prefix positives; do not train/replay")
    pos_weight = min(20.0, max(1.0, train_total - train_positive) / train_positive)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history, best_state, best_val, stale = [], None, None, 0
    for epoch in range(int(epochs)):
        model.train()
        state, total_loss, batches = None, 0.0, 0
        for start in range(0, train_end, int(chunk_len)):
            end = min(train_end, start + int(chunk_len))
            x = torch.from_numpy(features[start:end]).to(device).unsqueeze(0)
            c = torch.from_numpy(candidate_features[start:end]).to(device).unsqueeze(0)
            y = torch.from_numpy(labels[start:end]).to(device).unsqueeze(0)
            lead_y = torch.from_numpy(lead_classes[start:end]).to(device).unsqueeze(0)
            mask = torch.from_numpy(valid[start:end].astype(np.float32)).to(device).unsqueeze(0)
            utility_logits, lead_logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            bce = F.binary_cross_entropy_with_logits(utility_logits, y, pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
            utility_loss = torch.sum(bce * mask) / torch.clamp(mask.sum(), min=1.0)
            positive = ((y > 0.5) & (mask > 0.5))
            lead_loss = F.cross_entropy(lead_logits[positive], lead_y[positive]) if bool(positive.any()) else torch.sum(lead_logits * 0.0)
            loss = utility_loss + 0.10 * lead_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        utility_probability, _ = _predict_sequence(model, features, candidate_features, device, chunk_len)
        valid_mask = valid[train_end:].astype(bool)
        clipped = np.clip(utility_probability[train_end:][valid_mask], 1e-6, 1.0 - 1e-6)
        target = labels[train_end:][valid_mask]
        val_bce = float(np.mean(-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))) if len(target) else float("inf")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / float(max(1, batches)), "val_bce": val_bce, "train_positive": int(train_positive), "train_candidate_rows": int(train_total), "pos_weight": float(pos_weight)})
        if best_val is None or val_bce < best_val - 1e-5:
            best_val = val_bce
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return history


def _select_policy(scores, labels, valid, profile):
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
    slots = scores.shape[1]
    total_positive_events = int(np.sum(np.any((labels > 0) & valid, axis=1)))
    best = None
    for threshold in thresholds:
        for degree in range(1, slots + 1):
            selected = np.zeros_like(valid, dtype=bool)
            for index in range(scores.shape[0]):
                candidates = [slot for slot in range(slots) if valid[index, slot] and scores[index, slot] >= threshold]
                candidates.sort(key=lambda slot: (-float(scores[index, slot]), slot))
                for slot in candidates[:degree]:
                    selected[index, slot] = True
            issued = int(selected.sum())
            useful = int(np.sum((labels > 0) & selected))
            precision = float(useful) / float(issued) if issued else 0.0
            coverage = float(np.sum(np.any((labels > 0) & selected, axis=1))) / float(max(1, total_positive_events))
            issue_per_event = float(issued) / float(max(1, scores.shape[0]))
            meets_precision = precision >= float(profile["min_precision"])
            objective = (0.60 * precision + 0.35 * coverage - 0.05 * issue_per_event) if profile["goal"] == "reproduce_strong_normal" else (0.25 * precision + 0.70 * coverage - 0.05 * issue_per_event)
            candidate = {"threshold": float(threshold), "degree": int(degree), "issued": issued, "useful": useful, "precision": precision, "candidate_coverage": coverage, "issue_per_event": issue_per_event, "meets_min_precision": bool(meets_precision), "objective": objective}
            if best is None or (int(candidate["meets_min_precision"]), candidate["objective"], -candidate["issue_per_event"]) > (int(best["meets_min_precision"]), best["objective"], -best["issue_per_event"]):
                best = candidate
    if best is None:
        raise RuntimeError("no validation policy candidates")
    return best


def _score_to_selection(scores, valid, policy):
    selected = np.zeros_like(valid, dtype=bool)
    for index in range(scores.shape[0]):
        slots = [slot for slot in range(scores.shape[1]) if valid[index, slot] and scores[index, slot] >= float(policy["threshold"])]
        slots.sort(key=lambda slot: (-float(scores[index, slot]), slot))
        for slot in slots[:int(policy["degree"])]:
            selected[index, slot] = True
    return selected


def export_outputs(repo_root, artifact_dir, trace, run_id, profile, records, deltas, supports, valid, source_names, scores, lead_logits, labels, lead_classes, policy, model, history, ledger_scope, candidate_metadata):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    slots = deltas.shape[1]
    lead_probability = torch.softmax(torch.from_numpy(lead_logits), dim=-1).numpy()
    lead_index = np.argmax(lead_probability, axis=-1)
    lead_lo = LEAD_BINS[lead_index]
    selected = _score_to_selection(scores, valid.astype(bool), policy)
    rich_path = artifact_dir / "prefetch_list_{}_{}_lru{}.csv".format(trace, TINY_VERSION, profile["dedup_capacity"])
    ledger_path = artifact_dir / "decision_ledger_{}_{}.csv.gz".format(trace, TINY_VERSION)
    history_path, checkpoint_path, metadata_path, plan_path = artifact_dir / "training_history.csv", artifact_dir / "tiny_lstm.pt", artifact_dir / "run_metadata.json", artifact_dir / "replay_plan.csv"
    recent_issue, rich_rows = OrderedDict(), []
    ledger_scope = str(ledger_scope).lower()
    if ledger_scope not in {"full", "val"}:
        raise ValueError("ledger_scope must be 'full' or 'val'")
    ledger_start = 0 if ledger_scope == "full" else int(0.8 * len(records))
    ledger_fields = ["trace", "run_id", "demand_idx", "cycle", "pc", "line", "pc_line_occ", "candidate_rank", "candidate_source", "candidate_delta", "candidate_line", "candidate_valid", "future_label", "future_cycle_label", "prefetch_addr", "candidate_support", "utility_prob", "candidate_score", "lead4_prob", "lead8_prob", "lead16_prob", "lead32_prob", "lead64_prob", "predicted_lead_lo", "policy_threshold", "policy_degree", "selected_pre_dedup", "selected", "reject_reason"]
    ledger_rows = 0
    with gzip.open(str(ledger_path), "wt", newline="") as ledger_handle:
        writer = csv.DictWriter(ledger_handle, fieldnames=ledger_fields)
        writer.writeheader()
        for index, row in enumerate(records):
            for slot in range(slots):
                if index < ledger_start:
                    continue
                is_valid = bool(valid[index, slot])
                delta = int(deltas[index, slot])
                candidate_line = int(row["line"] + delta) if is_valid else 0
                address = int(candidate_line * LINE_BYTES) if is_valid else 0
                pre_dedup = bool(selected[index, slot]) if is_valid else False
                accepted = False
                if not is_valid:
                    reason = "invalid_candidate"
                elif not pre_dedup:
                    reason = "below_threshold" if float(scores[index, slot]) < float(policy["threshold"]) else "rank_budget"
                elif address <= 0 or address % LINE_BYTES:
                    reason = "invalid_address"
                else:
                    last = recent_issue.get(address)
                    if last is not None and index - last <= int(profile["dedup_capacity"]):
                        reason = "dedup_recent_address"
                    else:
                        accepted, reason = True, "selected"
                        recent_issue[address] = index
                        recent_issue.move_to_end(address)
                        while recent_issue and index - next(iter(recent_issue.values())) > int(profile["dedup_capacity"]):
                            recent_issue.popitem(last=False)
                lead_probs = lead_probability[index, slot]
                future_label = int(lead_classes[index, slot] + 1) if (is_valid and labels[index, slot] > 0.5) else 0
                writer.writerow({"trace": trace, "run_id": run_id, "demand_idx": row["demand_idx"], "cycle": row["cycle"], "pc": "0x{:x}".format(row["pc"]), "line": row["line"], "pc_line_occ": row["pc_line_occ"], "candidate_rank": slot, "candidate_source": str(source_names[index, slot]) if is_valid else "", "candidate_delta": delta if is_valid else "", "candidate_line": candidate_line if is_valid else "", "candidate_valid": int(is_valid), "future_label": future_label, "future_cycle_label": 0, "prefetch_addr": "0x{:x}".format(address) if address else "", "candidate_support": float(supports[index, slot]) if is_valid else "", "utility_prob": float(scores[index, slot]) if is_valid else "", "candidate_score": float(scores[index, slot]) if is_valid else "", "lead4_prob": float(lead_probs[0]) if is_valid else "", "lead8_prob": float(lead_probs[1]) if is_valid else "", "lead16_prob": float(lead_probs[2]) if is_valid else "", "lead32_prob": float(lead_probs[3]) if is_valid else "", "lead64_prob": float(lead_probs[4]) if is_valid else "", "predicted_lead_lo": int(lead_lo[index, slot]) if is_valid else "", "policy_threshold": float(policy["threshold"]), "policy_degree": int(policy["degree"]), "selected_pre_dedup": int(pre_dedup), "selected": int(accepted), "reject_reason": reason})
                ledger_rows += 1
                if accepted:
                    rich_rows.append({"trace": trace, "order": row["cycle"], "demand_idx": row["demand_idx"], "replay_idx": row["demand_idx"], "pc": row["pc"], "line": row["line"], "candidate_rank": slot, "candidate_delta": delta, "candidate_source": str(source_names[index, slot]), "candidate_support": float(supports[index, slot]), "utility_prob": float(scores[index, slot]), "far_prob": float(1.0 - lead_probs[0]), "issue_prob": float(scores[index, slot]), "predicted_lead_lo": int(lead_lo[index, slot]), "predicted_cycle_lo": 0, "candidate_score": float(scores[index, slot]), "prefetch_addr": int(address)})
    rich_fields = ["trace", "order", "demand_idx", "replay_idx", "pc", "line", "candidate_rank", "candidate_delta", "candidate_source", "candidate_support", "utility_prob", "far_prob", "issue_prob", "predicted_lead_lo", "predicted_cycle_lo", "candidate_score", "prefetch_addr"]
    _csv_write(rich_path, rich_rows, rich_fields)
    _csv_write(history_path, history, list(history[0].keys()) if history else ["epoch"])
    torch.save({"state_dict": model.state_dict(), "parameter_count": parameter_count(model), "architecture": "TinyCandidateLSTM(8->8, candidate 5->8, utility+5-way lead)", "lead_bins": LEAD_BINS.tolist(), "trace": trace}, str(checkpoint_path))
    source_rel = _safe_rel(rich_path, repo_root)
    tag = "tiny_{}".format(profile["short_name"])
    plan_rows = [{"tag": tag, "trace": trace, "source_rel": source_rel, "candidate_role": "tiny_trace_specialized", "model_family": "tiny_lstm_742p", "recipe": ",".join("{}x{}".format(source, count) for source, count in profile["sources"]), "policy_tag": "threshold_{:.2f}_top{}".format(float(policy["threshold"]), int(policy["degree"])), "artifact_tag": TINY_VERSION, "provisional_primary": 1, "goal": profile["goal"]}]
    _csv_write(plan_path, plan_rows, list(plan_rows[0].keys()))
    metadata = {"tiny_version": TINY_VERSION, "trace": trace, "run_id": run_id, "parameter_count": parameter_count(model), "neuron_count": 8, "candidate_slots": slots, "candidate_sources": profile["sources"], "goal": profile["goal"], "normal_reference_for_evaluation_only": profile["normal_reference"], "policy": policy, "ledger_scope": ledger_scope, "ledger_rows": ledger_rows, "exported_prefetch_rows": len(rich_rows), "rich_list": str(rich_path), "decision_ledger": str(ledger_path), "replay_plan": str(plan_path), "replay_plan_root": str(repo_root), "causality": "Runtime features use the current/past raw no-prefetch oracle stream only. Candidate tables are fitted from train-prefix labels only. The list is an offline keyed replay export, not in-simulator PyTorch inference.", "candidate_metadata": candidate_metadata}
    _json_dump(metadata_path, metadata)
    return metadata


def run_trace(repo_root, trace, run_id=None, artifact_root=None, seed=7, max_rows=0, epochs=6, chunk_len=1024, learning_rate=0.003, ledger_scope="full"):
    repo_root = Path(repo_root).resolve()
    if trace not in TRACE_PROFILES:
        raise ValueError("unsupported trace: {}".format(trace))
    profile = dict(TRACE_PROFILES[trace])
    run_id = run_id or "{}_seed{}".format(TINY_VERSION, seed)
    oracle_path = repo_root / "formal_NN_training/results/standalone_nn_data/oracle/{}.oracle.csv.gz".format(trace)
    if not oracle_path.is_file():
        raise FileNotFoundError("missing oracle: {}".format(oracle_path))
    records = read_oracle(oracle_path, max_rows=max_rows)
    dependency_feature, dependency_profile_path = _load_dependency_profile(repo_root, trace)
    if trace == "605.mcf_s-994B" and not dependency_feature:
        raise FileNotFoundError("605 tiny notebook requires dependency profile: {}".format(dependency_profile_path))
    features = make_runtime_features(records, dependency_feature)
    train_end = int(0.80 * len(records))
    if train_end <= int(chunk_len):
        raise RuntimeError("train partition too small for chunk length")
    tables, dependency_edges, candidate_metadata = build_candidate_tables(records, profile, train_end, repo_root, trace)
    deltas, supports, source_ids, valid, source_names = make_candidates(records, profile, tables, dependency_edges)
    candidate_features = make_candidate_features(deltas, supports, source_ids, valid)
    labels, lead_classes = make_labels(records, deltas, valid, profile)
    train_positive, val_positive = int(np.sum(labels[:train_end])), int(np.sum(labels[train_end:]))
    if train_positive <= 0 or val_positive <= 0:
        raise RuntimeError("candidate-bank ceiling is zero in train or validation (train={}, val={}); improve the representation before replay".format(train_positive, val_positive))
    _seed_everything(seed)
    device = _choose_device()
    if device.type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    model = TinyCandidateLSTM()
    assert_tiny_model(model)
    history = _fit_model(model, features, candidate_features, labels, lead_classes, valid, train_end, device, int(epochs), int(chunk_len), float(learning_rate))
    scores, lead_logits = _predict_sequence(model, features, candidate_features, device, int(chunk_len))
    policy = _select_policy(scores[train_end:], labels[train_end:], valid[train_end:].astype(bool), profile)
    if artifact_root is None:
        artifact_root = repo_root / "formal_NN_training/artifacts" / TINY_VERSION / trace / run_id
    else:
        artifact_root = Path(artifact_root)
    candidate_metadata.update({"oracle": str(oracle_path), "dependency_profile_path": str(dependency_profile_path), "train_rows": train_end, "validation_rows": len(records) - train_end, "train_positive_candidate_labels": train_positive, "validation_positive_candidate_labels": val_positive, "candidate_valid_rows_train": int(np.sum(valid[:train_end])), "candidate_valid_rows_validation": int(np.sum(valid[train_end:]))})
    metadata = export_outputs(repo_root, artifact_root, trace, run_id, profile, records, deltas, supports, valid, source_names, scores, lead_logits, labels, lead_classes, policy, model, history, ledger_scope, candidate_metadata)
    metadata["artifact_dir"] = str(Path(artifact_root).resolve())
    metadata["device"] = str(device)
    return metadata
