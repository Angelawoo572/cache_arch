#!/usr/bin/env python3
"""Colab training stage for post-SPP admission policies.

Pipeline:
  cluster: generate candidate table input with 05_events_to_candidate_table.py
  Colab:   run this script on data/generated/spp_candidate_log.csv.xz
           save policy table + held-out replay summary + interpretation
           push result directory to GitHub from notebook
  cluster: pull best_policy artifacts and patch ChampSim spp_dev for final replay

This script does not run ChampSim and does not claim final IPC/hit-rate changes.
It trains/evaluates admit/suppress policies on a held-out candidate window.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQ = [
    "trace", "cycle", "ip", "addr", "pf_addr", "delta", "spp_confidence", "spp_fill_l2", "cache_hit",
    "mshr_occupancy", "mshr_size", "pq_occupancy", "pq_size",
    "recent_spp_accuracy", "recent_pc_accuracy", "recent_delta_accuracy",
    "recent_cache_hit_rate", "recent_cache_miss_rate", "recent_pc_cache_hit_rate", "recent_delta_cache_hit_rate",
    "bandwidth_bucket", "set_pressure",
    "outcome_useful", "outcome_late", "outcome_evicted_unused", "outcome_duplicate",
]

DEFAULTS = {
    "spp_fill_l2": 0, "cache_hit": 0,
    "mshr_occupancy": 0, "mshr_size": 640,
    "pq_occupancy": 0, "pq_size": 16,
    "recent_spp_accuracy": 0.5, "recent_pc_accuracy": 0.5, "recent_delta_accuracy": 0.5,
    "recent_cache_hit_rate": 0.5, "recent_cache_miss_rate": 0.5,
    "recent_pc_cache_hit_rate": 0.5, "recent_delta_cache_hit_rate": 0.5,
    "bandwidth_bucket": 0, "set_pressure": 0,
    "outcome_late": 0, "outcome_evicted_unused": 0, "outcome_duplicate": 0,
}

FEATURE_SETS: Dict[str, List[str]] = {
    "F0_candidate": ["pc_bucket", "delta_bucket", "confidence_bucket"],
    "F1_candidate_mshr_pq": ["pc_bucket", "delta_bucket", "confidence_bucket", "mshr_bucket", "pq_bucket"],
    "F4_cache_hit_only": ["pc_bucket", "delta_bucket", "confidence_bucket", "cache_hit"],
    "F5_recent_hitmiss": ["pc_bucket", "delta_bucket", "confidence_bucket", "recent_hit_bucket", "recent_miss_bucket", "recent_pc_hit_bucket", "recent_delta_hit_bucket"],
    "F6_resource_hitmiss": ["pc_bucket", "delta_bucket", "confidence_bucket", "mshr_bucket", "pq_bucket", "cache_hit", "recent_hit_bucket", "recent_miss_bucket", "recent_pc_hit_bucket", "recent_delta_hit_bucket"],
}

COMPLEXITY_RANK = {
    "F0_candidate": 0,
    "F1_candidate_mshr_pq": 1,
    "F4_cache_hit_only": 2,
    "F5_recent_hitmiss": 3,
    "F6_resource_hitmiss": 4,
}


def safe_tag(x: str) -> str:
    return str(x).replace(".", "_").replace("-", "_").replace("/", "_")


def compression_for(path: Path):
    if str(path).endswith(".xz"):
        return "xz"
    if str(path).endswith(".gz"):
        return "gzip"
    return None


def load_table(path: Path, max_rows: int | None, min_conf: int) -> pd.DataFrame:
    parts = []
    remaining = max_rows
    seen = seen_useful = kept = kept_useful = 0
    for chunk in pd.read_csv(path, compression=compression_for(path), chunksize=100_000):
        for c in REQ:
            if c not in chunk.columns:
                chunk[c] = DEFAULTS.get(c, 0)
        seen += len(chunk)
        seen_useful += int(chunk["outcome_useful"].sum())
        # Keep this notebook consistent with the cluster-generated l2 scope.
        chunk = chunk[(chunk["spp_fill_l2"].astype(int) == 1) | (chunk["spp_confidence"].astype(int) >= min_conf)]
        kept += len(chunk)
        kept_useful += int(chunk["outcome_useful"].sum())
        chunk = chunk[REQ]
        if remaining is not None:
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        if len(chunk):
            parts.append(chunk)
        if remaining is not None and remaining <= 0:
            break
    if not parts:
        raise SystemExit(f"[error] no candidates loaded from {path}")
    df = pd.concat(parts, ignore_index=True)
    print(f"[source prefix] rows={seen:,} useful_rate={seen_useful/max(1, seen):.6f}")
    print(f"[scope prefix] rows={kept:,} useful_rate={kept_useful/max(1, kept):.6f}")
    print(f"[loaded] rows={len(df):,}")
    return df


def bucketize(x: float, bins: Tuple[float, ...]) -> int:
    for i, b in enumerate(bins):
        if x <= b:
            return i
    return len(bins)


def reward(df: pd.DataFrame) -> pd.Series:
    useful = df["outcome_useful"].astype("float32")
    duplicate = df["outcome_duplicate"].astype("float32")
    late = df["outcome_late"].astype("float32")
    evicted = df["outcome_evicted_unused"].astype("float32")
    mshr = df["mshr_occupancy"].astype("float32") / df["mshr_size"].replace(0, 1).astype("float32")
    pq = df["pq_occupancy"].astype("float32") / df["pq_size"].replace(0, 1).astype("float32")
    return (2.0 * useful - 1.0 * (1.0 - useful) - 0.5 * duplicate - 0.5 * late - 0.5 * evicted - 0.3 * mshr - 0.1 * pq).astype("float32")


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pc_bucket"] = (out["ip"].astype("int64") % 256).astype("int16")
    out["delta_bucket"] = out["delta"].astype("int64").clip(-32, 32).astype("int16")
    out["confidence_bucket"] = out["spp_confidence"].astype(float).map(lambda x: bucketize(x, (90, 92, 95, 98))).astype("int8")
    mshr = out["mshr_occupancy"].astype(float) / out["mshr_size"].replace(0, 1).astype(float)
    pq = out["pq_occupancy"].astype(float) / out["pq_size"].replace(0, 1).astype(float)
    out["mshr_bucket"] = mshr.map(lambda x: bucketize(x, (0.25, 0.50, 0.75, 0.90))).astype("int8")
    out["pq_bucket"] = pq.map(lambda x: bucketize(x, (0.25, 0.50, 0.75, 0.90))).astype("int8")
    out["cache_hit"] = out["cache_hit"].astype(int).clip(0, 1).astype("int8")
    out["recent_hit_bucket"] = out["recent_cache_hit_rate"].astype(float).map(lambda x: bucketize(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_miss_bucket"] = out["recent_cache_miss_rate"].astype(float).map(lambda x: bucketize(x, (0.05, 0.20, 0.40, 0.60, 0.80))).astype("int8")
    out["recent_pc_hit_bucket"] = out["recent_pc_cache_hit_rate"].astype(float).map(lambda x: bucketize(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["recent_delta_hit_bucket"] = out["recent_delta_cache_hit_rate"].astype(float).map(lambda x: bucketize(x, (0.20, 0.40, 0.60, 0.80, 0.95))).astype("int8")
    out["admit_reward"] = reward(out)
    return out


@dataclass
class BanditConfig:
    alpha: float = 0.05
    epsilon: float = 0.10
    episodes: int = 2
    seed: int = 1


def state_key(row: pd.Series, features: List[str]) -> Tuple[int, ...]:
    return tuple(int(row[f]) for f in features)


def train(train_df: pd.DataFrame, features: List[str], cfg: BanditConfig):
    rng = np.random.default_rng(cfg.seed)
    q = defaultdict(lambda: np.zeros(2, dtype=np.float32))
    rows = train_df.sort_values("cycle").reset_index(drop=True)
    for _ in range(cfg.episodes):
        for _, row in rows.iterrows():
            s = state_key(row, features)
            a = int(rng.integers(0, 2)) if rng.random() < cfg.epsilon else int(np.argmax(q[s]))
            r = float(row["admit_reward"]) if a == 1 else 0.0
            q[s][a] += cfg.alpha * (r - q[s][a])
    return q


def decide(row: pd.Series, features: List[str], q, min_conf: int) -> int:
    values = q.get(state_key(row, features))
    if values is None:
        mshr = row["mshr_occupancy"] / max(1, row["mshr_size"])
        return int((row["spp_confidence"] >= min_conf) and (mshr < 0.90))
    return int(values[1] > values[0])


def evaluate(eval_df: pd.DataFrame, features: List[str], q, min_conf: int):
    decisions = eval_df.apply(lambda r: decide(r, features, q, min_conf), axis=1).astype("int8")
    total = len(eval_df)
    issued = int(decisions.sum())
    available_useful = int(eval_df["outcome_useful"].sum())
    useful = int(((decisions == 1) & (eval_df["outcome_useful"] == 1)).sum())
    useless_total = total - available_useful
    bad_suppressed = int(((decisions == 0) & (eval_df["outcome_useful"] == 0)).sum())
    rew = float((decisions * eval_df["admit_reward"]).sum())
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
        "estimated_reward": rew,
    }


def serialize(q, features: List[str], limit: int):
    states = []
    for s, v in q.items():
        if float(v[1]) > float(v[0]):
            states.append({"state": list(map(int, s)), "q_suppress": float(v[0]), "q_admit": float(v[1]), "margin": float(v[1] - v[0]), "action": 1})
    states.sort(key=lambda r: r["margin"], reverse=True)
    truncated = len(states) > limit
    return {"features": features, "num_admit_states_total": len(states), "num_admit_states_saved": min(len(states), limit), "truncated": truncated, "admit_states": states[:limit]}


def sanity(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "trace": str(df["trace"].iloc[0]),
        "rows": len(df),
        "useful_rate": float(df["outcome_useful"].mean()),
        "cache_hit_rate_at_candidate": float(df["cache_hit"].mean()),
        "recent_cache_hit_rate_avg": float(df["recent_cache_hit_rate"].mean()),
        "recent_cache_miss_rate_avg": float(df["recent_cache_miss_rate"].mean()),
        "mshr_avg": float(df["mshr_occupancy"].mean()),
        "mshr_max": int(df["mshr_occupancy"].max()),
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--scope", default="spp_l2_issue")
    ap.add_argument("--max-rows", type=int, default=300_000)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--min-confidence", type=int, default=90)
    ap.add_argument("--state-limit", type=int, default=50_000)
    args = ap.parse_args()

    df = featurize(load_table(args.input, args.max_rows, args.min_confidence))
    trace = str(df["trace"].iloc[0])
    out_dir = args.out_root / f"{safe_tag(trace)}_{safe_tag(args.scope)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(len(df) * args.train_frac)
    train_df = df.sort_values("cycle").iloc[:n_train].copy()
    eval_df = df.sort_values("cycle").iloc[n_train:].copy()

    cfg = BanditConfig()
    rows = []
    policies = {}
    for name, features in FEATURE_SETS.items():
        q = train(train_df, features, cfg)
        m = evaluate(eval_df, features, q, args.min_confidence)
        m.update({"feature_set": name, "trace": trace, "num_features": len(features), "num_states": len(q), "scope": args.scope})
        rows.append(m)
        policies[name] = serialize(q, features, args.state_limit)

    summary = pd.DataFrame(rows)
    base = summary[summary["feature_set"] == "F0_candidate"].iloc[0]
    for col in ["accuracy", "issued_ratio", "useful_kept_ratio", "bad_suppressed_ratio", "estimated_reward"]:
        summary["delta_" + col + "_vs_F0"] = summary[col] - base[col]

    best_reward = summary["estimated_reward"].max()
    near = summary[summary["estimated_reward"] >= best_reward - max(1e-6, abs(best_reward) * 0.001)].copy()
    near["complexity"] = near["feature_set"].map(COMPLEXITY_RANK)
    best_name = str(near.sort_values("complexity").iloc[0]["feature_set"])
    best_policy = {"feature_set": best_name, **policies[best_name]}

    admit_rows = []
    for item in best_policy["admit_states"]:
        r = {"feature_set": best_name, "action": 1, "margin": item["margin"], "q_suppress": item["q_suppress"], "q_admit": item["q_admit"]}
        for i, v in enumerate(item["state"]):
            r[f"s{i}"] = v
        admit_rows.append(r)

    sanity_df = sanity(df)
    sanity_df.to_csv(out_dir / "candidate_sanity.csv", index=False)
    summary.to_csv(out_dir / "policy_replay_summary.csv", index=False)
    summary.to_csv(out_dir / "feature_sweep_compare_vs_F0.csv", index=False)
    (out_dir / "policy_tables.json").write_text(json.dumps(policies, indent=2))
    (out_dir / "best_policy.json").write_text(json.dumps(best_policy, indent=2))
    pd.DataFrame(admit_rows).to_csv(out_dir / "best_policy_admit_states.csv", index=False)

    f0 = summary[summary["feature_set"] == "F0_candidate"].iloc[0]
    best = summary[summary["feature_set"] == best_name].iloc[0]
    lines = [
        f"# Interpretation\n",
        f"trace: {trace}\n",
        f"scope: {args.scope}\n",
        f"candidate useful rate={sanity_df.iloc[0].useful_rate:.4f}",
        f"candidate cache_hit={sanity_df.iloc[0].cache_hit_rate_at_candidate:.4f}",
        f"recent hit avg={sanity_df.iloc[0].recent_cache_hit_rate_avg:.4f}, recent miss avg={sanity_df.iloc[0].recent_cache_miss_rate_avg:.4f}",
        f"F0 reward={f0.estimated_reward:.2f}, accuracy={f0.accuracy:.4f}, useful_kept={f0.useful_kept_ratio:.4f}",
        f"best policy={best_name}, reward={best.estimated_reward:.2f}, accuracy={best.accuracy:.4f}, useful_kept={best.useful_kept_ratio:.4f}",
    ]
    if best_name in ["F5_recent_hitmiss", "F6_resource_hitmiss"] and best.estimated_reward > f0.estimated_reward:
        lines.append("Conclusion: online hit/miss-rate features improve held-out candidate replay over F0.")
    else:
        lines.append("Conclusion: hit/miss features do not beat F0 under this held-out candidate replay.")
    (out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "stage": "Colab training output for cluster ChampSim replay",
        "input": str(args.input),
        "trace": trace,
        "scope": args.scope,
        "max_rows": args.max_rows,
        "train_frac": args.train_frac,
        "min_confidence": args.min_confidence,
        "feature_sets": FEATURE_SETS,
        "best_policy_name": best_name,
        "best_policy_features": best_policy["features"],
        "next_cluster_step": "pull best_policy.json/best_policy_admit_states.csv and patch ChampSim spp_dev before prefetch_line",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[out_dir] {out_dir}")
    print(summary[["feature_set", "issued_ratio", "accuracy", "useful_kept_ratio", "bad_suppressed_ratio", "estimated_reward", "delta_estimated_reward_vs_F0"]].to_string(index=False))
    print(f"[best_policy] {best_name}")


if __name__ == "__main__":
    main()
