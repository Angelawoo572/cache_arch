#!/usr/bin/env bash
# Build and package the v3.9 605 dependency sidecars in one decoded raw-prefix
# instruction pass:
#   (1) PC-static dependency profile; and
#   (2) producer-PC keyed dependency edge vocabulary.
#
# Start it with:
#   nohup bash formal_NN_training/scripts/17_prepare_v3_9_605_dependency_sidecar.sh &
#
# The wrapper creates OUT_DIR before redirecting, so no extra shell redirection
# is necessary. The generated sidecars are static raw-prefix artifacts; the
# notebook reads sidecars plus the canonical no-prefetch oracle and never scans
# the raw .xz trace in Colab.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
WARMUP_RECORDS="${WARMUP_RECORDS:-25000000}"
PROFILE_RECORDS="${PROFILE_RECORDS:-20000000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"
EDGE_TOP_K="${EDGE_TOP_K:-8}"
EDGE_MIN_SUPPORT="${EDGE_MIN_SUPPORT:-16}"
EDGE_MAX_DELTAS_PER_PRODUCER="${EDGE_MAX_DELTAS_PER_PRODUCER:-256}"
EDGE_MAX_CONSUMERS_PER_DELTA="${EDGE_MAX_CONSUMERS_PER_DELTA:-4}"
EDGE_MAX_PRODUCERS="${EDGE_MAX_PRODUCERS:-250000}"

BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"
PROFILE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_profile.csv.gz"
PROFILE_META="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_profile.json"
EDGE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_edge_vocab.csv.gz"
EDGE_META="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_edge_vocab.json"
PACKAGE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_sidecar.tar.gz"
LOG="$OUT_DIR/605_sidecar_build.out"

[[ -d "$REPO_ROOT/.git" ]] || { echo "[error] REPO_ROOT is not a cache repo: $REPO_ROOT" >&2; exit 2; }
[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder script not found: $BUILDER" >&2; exit 2; }
mkdir -p "$OUT_DIR"

exec >> "$LOG" 2>&1

echo "===== v3.9 605 dependency-sidecar build ====="
date
echo "[build] repo=$REPO_ROOT"
echo "[build] trace=$TRACE"
echo "[build] profile=$PROFILE"
echo "[build] edge_vocab=$EDGE"
echo "[build] warmup_records=$WARMUP_RECORDS profile_records=$PROFILE_RECORDS"
echo "[build] edge_top_k=$EDGE_TOP_K edge_min_support_lower_bound=$EDGE_MIN_SUPPORT"

rm -f "$PROFILE" "$PROFILE_META" "$EDGE" "$EDGE_META" "$PACKAGE" \
      "${PROFILE}.partial" "${PROFILE_META}.partial" "${EDGE}.partial" "${EDGE_META}.partial"

python3 "$BUILDER" \
  --trace "$TRACE" \
  --output "$PROFILE" \
  --meta "$PROFILE_META" \
  --edge-output "$EDGE" \
  --edge-meta "$EDGE_META" \
  --edge-top-k "$EDGE_TOP_K" \
  --edge-min-support "$EDGE_MIN_SUPPORT" \
  --edge-max-deltas-per-producer "$EDGE_MAX_DELTAS_PER_PRODUCER" \
  --edge-max-consumers-per-delta "$EDGE_MAX_CONSUMERS_PER_DELTA" \
  --edge-max-producers "$EDGE_MAX_PRODUCERS" \
  --warmup-records "$WARMUP_RECORDS" \
  --profile-records "$PROFILE_RECORDS" \
  --progress-every "$PROGRESS_EVERY"

[[ -s "$PROFILE" && -s "$PROFILE_META" ]] || { echo "[error] profile outputs were not produced" >&2; exit 3; }
[[ -s "$EDGE" && -s "$EDGE_META" ]] || { echo "[error] edge-vocabulary outputs were not produced" >&2; exit 3; }

python3 - "$PROFILE_META" "$EDGE_META" "$EDGE" <<'PY'
from __future__ import print_function
import csv
import gzip
import json
import sys

profile = json.load(open(sys.argv[1]))
edge = json.load(open(sys.argv[2]))
edge_csv = sys.argv[3]

assert profile.get("schema") == "v3_9_pc_static_dependency_profile", profile
assert profile.get("profile_scope") == "raw-trace training prefix only", profile
assert profile.get("uses_oracle_alignment") is False, profile
assert edge.get("schema") == "v3_9_pc_dependency_edge_vocab", edge
assert edge.get("schema_version") == "v3_9_producer_delta_vocab_1", edge
assert edge.get("profile_scope") == "raw-trace training prefix only", edge
assert edge.get("uses_oracle_alignment") is False, edge
assert profile.get("trace_sha256"), "profile trace sha256 missing"
assert edge.get("trace_sha256"), "edge trace sha256 missing"
assert edge.get("trace_sha256") == profile.get("trace_sha256"), "profile/edge trace mismatch"
assert profile.get("warmup_records_skipped") == edge.get("warmup_records"), "warmup mismatch"
assert profile.get("profile_records") == edge.get("profile_records"), "profile window mismatch"
assert profile.get("cache_line_bytes") == edge.get("cache_line_bytes"), "cache-line mismatch"
assert edge.get("runtime_lookup_key") == "producer_pc", edge
assert int(edge.get("top_k_per_producer", 0)) > 0, edge
assert int(edge.get("min_support_lower_bound", 0)) > 0, edge

expected = [
    "producer_pc", "producer_to_target_line_delta", "estimated_support",
    "support_lower_bound", "support_error_bound", "rank_within_producer",
    "representative_consumer_pc", "consumer_pc_slots", "producer_is_load",
]
seen_ranks = {}
with gzip.open(edge_csv, "rt", newline="") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == expected, reader.fieldnames
    for row in reader:
        pc = int(row["producer_pc"])
        rank = int(row["rank_within_producer"])
        assert rank >= 0 and rank < int(edge["top_k_per_producer"]), row
        assert int(row["support_lower_bound"]) >= int(edge["min_support_lower_bound"]), row
        assert int(row["estimated_support"]) >= int(row["support_lower_bound"]), row
        assert int(row["support_error_bound"]) == int(row["estimated_support"]) - int(row["support_lower_bound"]), row
        seen_ranks.setdefault(pc, []).append(rank)
for pc, ranks in seen_ranks.items():
    assert sorted(ranks) == list(range(len(ranks))), (pc, ranks)

assert len(seen_ranks) == int(edge["distinct_producers_exported"]), edge
print("[verified] profile: unique_pcs={:,} dependency_pcs={:,} profile_records={:,}".format(
    int(profile["unique_pcs"]),
    int(profile["pcs_with_dependency_observations"]),
    int(profile["profile_records"]),
))
print("[verified] edge vocab: rows={:,} producers={:,} top_k={} min_lower_bound={}".format(
    int(edge["rows"]),
    int(edge["distinct_producers_exported"]),
    int(edge["top_k_per_producer"]),
    int(edge["min_support_lower_bound"]),
))
PY

tar -C "$OUT_DIR" -czf "$PACKAGE" \
  "$(basename "$PROFILE")" "$(basename "$PROFILE_META")" \
  "$(basename "$EDGE")" "$(basename "$EDGE_META")"

sha256sum "$PROFILE" "$PROFILE_META" "$EDGE" "$EDGE_META" "$PACKAGE"
echo "[done] package=$PACKAGE"
echo "[next] copy the four sidecars (not the duplicate tarball) into:"
echo "       formal_NN_training/data/upload/v3_9_dependency_profiles/"
date
