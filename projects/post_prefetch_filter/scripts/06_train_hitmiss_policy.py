#!/usr/bin/env python3
"""Train and offline-replay post-SPP admission policies with hit/miss-rate inputs.

This is the next step after 05_events_to_candidate_table.py. It does not claim
final IPC improvement; it replays the learned admit/suppress policy over a held-out
candidate window and saves the policy table/results for later ChampSim integration.

Example:
  python3 projects/post_prefetch_filter/scripts/06_train_hitmiss_policy.py \
    --input projects/post_prefetch_filter/data/generated/spp_candidate_log.csv.xz \
    --out-dir projects/post_prefetch_filter/results/hitmiss_policy_replay/605_mcf_s_994B \
    --trace 605.mcf_s-994B
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "trace", "cycle", "ip", "addr", "pf_addr", "delta",
    "spp_confidence", "spp_fill_l2", "cache_hit",
    "mshr_occupancy", "mshr_size", "pq_occupancy", "pq_size",
    "recent_spp_accuracy", "recent_pc_accuracy", "recent_delta_accuracy",
    "recent_cache_hit_rate", "recent_cache_miss_rate",
    "recent_pc_cache_hit_rate", "recent_delta_cache_hit_rate",
    "bandwidth_bucket", "set_pressure",
    "outcome_useful", "outcome_late", "outcome_evicted_unused", "outcome_duplicate",
]

DEFAULTS = {
    "spp_fill_l2": 0,
    "cache_hit": 0,
    "mshr_occupancy": 0,
    "mshr_size": 640,
    "pq_occupancy": 0,
    "pq_size": 16,
    "recent_spp_accuracy": 0.5,
    "recent_pc_accuracy": 0.5,
    "recent_delta_accuracy": 0.5,
    "recent_cache_hit_rate": 0.5,
    "recent_cache_miss_rate": 0.5,
    "recent_pc_cache_hit_rate": 0.5,
    "recent_delta_cache_hit_rate": 0.5,
    "bandwidth_bucket": 0,
    "set_pressure": 0,
    "outcome_late": 0,
    "outcome_evicted_unused": 0,
    "outcome_duplicate": 0,
}


def open_input(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    if str(path).endswith(".xz"):
        return lzma.open(path, "rt")
    return path.open("rt")


def bucketize_float(x: float, bins: Tuple[float, ...]) -> int:
    for i, b in enumerate(bins):
        if x <= b:
            return i
    return len(bins)


def load_table(path: Path, max_rows: int | None = None, trace: str | None = None) -> pd.DataFrame:
    compression = "xz" if str(path).endswith(".xz") else ("gzip" if str(path).endswith(".gz") else None)
    chunks = []
    remaining = max_rows
    for chunk in pd.read_csv(path, compression=compression, chunksize=100_000):
        if trace:
            chunk = chunk[chunk["trace"] == trace]
        for col in BASE_COLUMNS:
            if col not in chunk.columns:
                chunk[col] = DEFAULTS.get(col, 0)
        chunk = chunk[BASE_COLUMNS]
        if remaining is not None:
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        if len(chunk):
            chunks.append(chunk)
        if remaining is not None and remaining <= 0:
            break
    if not chunks:
        raise SystemExit(f"[error] no rows loaded from {path}")
    df = pd.concat(chunks, ignore_index=True)
    return df


def compute_reward(df: pd.DataFrame) -> pd.Series:
    useful = df["outcome_useful"].astype("float32")
    duplicate = df["outcome_duplicate"].astype("float32")
    late = df["outcome_late"].astype("float32")
    evicted_unused = df["outcome_evicted_unused"].astype("float32")
    mshr_pressure = df["mshr_occupancy"].astype("float32") / df["mshr_size"].replace(0, 1).astype("float32")
    pq_pressure = df["pq_occupancy"].astype("float32") / df["pq_size"].replace(0, 1).astype("float32")
    return (+2.0 * useful - 1.0 * (1.0 - useful) - 0.5 * duplicate - 0.5 * late - 0.5 * evicted_unused - 0.3 * mshr_pressure - 0.1 * pq_pressure).astype("float32")


def add_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pc_bucket"] = (out["ip"].astype("int64") % 256).astype("int16")
    out["delta_bucket"] = out["delta"].astype("int64").clip(-32, 32).astype("int16")
    out["confidence_bucket"] = out["spp_confidence"].astype(float).map(lambda x: bucketize_float(x, (90, 92, 95, 98))).astype("int8")

    mshr_pressure = out["mshr_occupancy"].astype(float) / out["mshr_size"].replace(0, 1).astype(float)
    pq_pressure = out["pq_occupancy"].astype(float) / out["pq_size"].replace(0, 1).astype(float)
    out["mshr_bucket"] = mshr_pressure.map(lambda x: bucketize_float(x, (0.25, 0.50, 0.75, 0.90))).astype("int8")
    out["pq_bucket"] = pq_pressure.map(lambda x: bucketize_float(x, (0.25, 0.50, 0.75, 0.90))).astype("int8")

    out["recent_spp_acc_bucket"] = out["recent_spp_accuracy"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_pc_acc_bucket"] = out["recent_pc_accuracy"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_delta_acc_bucket"] = out["recent_delta_accuracy"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")

    out["cache_hit"] = out["cache_hit"].astype(int).clip(0, 1).astype("int8")
    out["recent_hit_bucket"] = out["recent_cache_hit_rate"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_miss_bucket"] = out["recent_cache_miss_rate"].astype(float).map(lambda x: bucketize_float(x, (0.05, 0.20, 0.40, 0.60, 0.80))).astype("int8")
    out["recent_pc_hit_bucket"] = out["recent_pc_cache_hit_rate"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_delta_hit_bucket"] = out["recent_delta_cache_hit_rate"].astype(float).map(lambda x: bucketize_float(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")

    out["bandwidth_bucket"] = out["bandwidth_bucket"].astype(int).clip(0, 7).astype("int8")
    out["set_pressure"] = out["set_pressure"].astype(int).clip(0, 7).astype("int8")
    out["admit_reward"] = compute_reward(out)
    return out


FEATURE_SETS: Dict[str, List[str]] = {
    "F0_candidate": ["pc_bucket", "delta_bucket", "confidence_bucket"],
    "F1_candidate_mshr_pq": ["pc_bucket", "delta_bucket", "confidence_bucket", "mshr_bucket", "pq_bucket"],
    "F4_cache_hit_only": ["pc_bucket", "delta_bucket", "confidence_bucket", "cache_hit"],
    "F5_recent_hitmiss": ["pc_bucket", "delta_bucket", "confidence_bucket", "recent_hit_bucket", "recent_miss_bucket", "recent_pc_hit_bucket", "recent_delta_hit_bucket"],
    "F6_resource_hitmiss": ["pc_bucket", "delta_bucket", "confidence_bucket", "mshr_bucket", "pq_bucket", "cache_hit", "recent_hit_bucket", "recent_miss_bucket", "recent_pc_hit_bucket", "recent_delta_hit_bucket"],
}


@dataclass
class BanditConfig:
    alpha: float = 0.05
    epsilon: float = 0.10
    episodes: int = 2
    seed: int = 1


def make_state_key(row: pd.Series, features: List[str]) -> Tuple[int, ...]:
    return tuple(int(row[f]) for f in features)


def train_contextual_bandit(train_df: pd.DataFrame, features: List[str], cfg: BanditConfig):
    rng = np.random.default_rng(cfg.seed)
    q = defaultdict(lambda: np.zeros(2, dtype=np.float32))
    rows = train_df.sort_values("cycle").reset_index(drop=True)
    for _ in range(cfg.episodes):
        for _, row in rows.iterrows():
            s = make_state_key(row, features)
            a = int(rng.integers(0, 2)) if rng.random() < cfg.epsilon else int(np.argmax(q[s]))
            reward = float(row["admit_reward"]) if a == 1 else 0.0
            q[s][a] += cfg.alpha * (reward - q[s][a])
    return q


def decide_with_policy(row, features, q, min_confidence: int, threshold: float = 0.0) -> int:
    values = q.get(make_state_key(row, features))
    if values is None:
        mshr_pressure = row["mshr_occupancy"] / max(1, row["mshr_size"])
        return int((row["spp_confidence"] >= min_confidence) and (mshr_pressure < 0.90))
    return int(values[1] > values[0] + threshold)


def evaluate_policy(eval_df, features, q, min_confidence: int):
    decisions = eval_df.apply(lambda r: decide_with_policy(r, features, q, min_confidence), axis=1).astype("int8")
    issued = int(decisions.sum())
    total = len(eval_df)
    available_useful = int(eval_df["outcome_useful"].sum())
    useful = int(((decisions == 1) & (eval_df["outcome_useful"] == 1)).sum())
    useless_total = total - available_useful
    bad_suppressed = int(((decisions == 0) & (eval_df["outcome_useful"] == 0)).sum())
    reward = float((decisions * eval_df["admit_reward"]).sum())
    return {
        "candidates": total,
        "issued": issued,
        "suppressed": total - issued,
        "issued_ratio": issued / max(1, total),
        "useful": useful,
        "available_useful": available_useful,
        "useful_kept_ratio": useful / max(1, available_useful),
        "bad_suppressed": bad_suppressed,
        "bad_suppressed_ratio": bad_suppressed / max(1, useless_total),
        "accuracy": useful / max(1, issued),
        "estimated_reward": reward,
    }


def serialize_policy(q, features: List[str], limit: int | None = None):
    items = []
    for state, values in q.items():
        action = int(values[1] > values[0])
        if action == 1:
            items.append({"state": list(map(int, state)), "q_suppress": float(values[0]), "q_admit": float(values[1]), "action": action})
    items.sort(key=lambda x: x["q_admit"] - x["q_suppress"], reverse=True)
    if limit is not None:
        items = items[:limit]
    return {"features": features, "admit_states": items, "num_admit_states": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--max-rows", type=int, default=300_000)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--min-confidence", type=int, default=90)
    ap.add_argument("--policy-limit", type=int, default=20000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_table(args.input, args.max_rows, args.trace)
    df = add_bucket_features(df)
    df = df.sort_values("cycle").reset_index(drop=True)

    n_train = int(len(df) * args.train_frac)
    train_df = df.iloc[:n_train].copy()
    eval_df = df.iloc[n_train:].copy()

    cfg = BanditConfig()
    rows = []
    policies = {}
    for name, features in FEATURE_SETS.items():
        q = train_contextual_bandit(train_df, features, cfg)
        metrics = evaluate_policy(eval_df, features, q, args.min_confidence)
        metrics.update({"feature_set": name, "num_features": len(features), "num_states": len(q), "trace": args.trace or "ALL"})
        rows.append(metrics)
        policies[name] = serialize_policy(q, features, args.policy_limit)

    summary = pd.DataFrame(rows)
    base = summary[summary["feature_set"] == "F0_candidate"].iloc[0]
    for col in ["accuracy", "issued_ratio", "useful_kept_ratio", "bad_suppressed_ratio", "estimated_reward"]:
        summary["delta_" + col + "_vs_F0"] = summary[col] - base[col]

    summary_path = args.out_dir / "hitmiss_policy_replay_summary.csv"
    policy_path = args.out_dir / "hitmiss_policy_tables.json"
    manifest_path = args.out_dir / "run_manifest.json"

    summary.to_csv(summary_path, index=False)
    policy_path.write_text(json.dumps(policies, indent=2))
    manifest_path.write_text(json.dumps({
        "input": str(args.input),
        "trace": args.trace,
        "max_rows": args.max_rows,
        "train_frac": args.train_frac,
        "min_confidence": args.min_confidence,
        "feature_sets": FEATURE_SETS,
        "note": "Offline candidate replay only. Use this policy table for later ChampSim online integration.",
    }, indent=2))

    print("[loaded rows]", len(df))
    print("[train rows]", len(train_df))
    print("[eval rows]", len(eval_df))
    print("[summary]", summary_path)
    print("[policy]", policy_path)
    print(summary[["feature_set", "issued_ratio", "accuracy", "useful_kept_ratio", "bad_suppressed_ratio", "estimated_reward", "delta_estimated_reward_vs_F0"]].to_string(index=False))


if __name__ == "__main__":
    main()
