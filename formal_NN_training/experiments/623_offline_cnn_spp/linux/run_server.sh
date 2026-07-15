#!/usr/bin/env bash
# Independent 623 track: normal SPP versus direct SPP-interface causal CNN only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_cnn_spp"
TRACE="623.xalancbmk_s-700B"
POLICY="spp"
RUN_ID="${RUN_ID:-623_offline_cnn_spp_threshold_free_v9_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-threshold_free_spp_cnn_c8,threshold_free_spp_cnn_c13,threshold_free_spp_cnn_c22}"
BASE_TAG="${BASE_TAG:-threshold_free_spp_cnn_c8}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_direct_spp_cnn_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
VALIDATE_INPUTS="$EXP/python/validate_collected_inputs.py"
ANALYZE="$EXP/python/analyze_replay.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"
SOURCE_CONTRACT_REPO="$EXP/data/spp_source_contract.json"
SOURCE_CONTRACT_INPUT="$STREAM_DIR/spp_source_contract.json"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"

audit_spp_source() {
  python3 - "$CHAMP_DIR/prefetcher/spp_dev2.cc" \
    "$CHAMP_DIR/inc/spp_dev2.h" "$SOURCE_CONTRACT_REPO" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

source_path, header_path, contract_path = map(Path, sys.argv[1:])
for path in (source_path, header_path, contract_path):
    if not path.is_file():
        raise SystemExit("missing SPP source-audit file {}".format(path))
source = source_path.read_text(errors="ignore")
contract = json.loads(contract_path.read_text())
if (
    contract.get("self_target_action_semantics")
    != "allowed_by_source_lookahead_and_replayed"
    or contract.get("queue_effect_canonicalization")
    != "per_target_min_fill_queue_effect"
    or contract.get("decision_effective_external_input")
    != ["callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr"]
):
    raise SystemExit("unexpected direct-SPP self-target/queue contract")
missing = [marker for marker in contract["required_markers"] if marker not in source]
if missing:
    raise SystemExit("SPP source contract markers missing: {}".format(missing))
match = re.search(
    r"void\s+SPP_dev2::invoke_prefetcher\s*\([^)]*\)\s*\{(.*?)\n\}",
    source,
    flags=re.S,
)
if not match:
    raise SystemExit("cannot isolate SPP_dev2::invoke_prefetcher body")
signature_and_body = source[source.find("void SPP_dev2::invoke_prefetcher"):match.end()]
for unused in ("cache_hit", "type"):
    if len(re.findall(r"\b{}\b".format(unused), signature_and_body)) != 1:
        raise SystemExit(
            "SPP {} is no longer signature-only; revisit neural input contract".format(unused)
        )
if not re.search(r"\baddr\b", match.group(1)):
    raise SystemExit("SPP invoke body no longer consumes addr")
fill_match = re.search(
    r"void\s+SPP_dev2::cache_fill\s*\([^)]*\)\s*\{(.*?)\n\}",
    source,
    flags=re.S,
)
if not fill_match:
    raise SystemExit("cannot isolate SPP_dev2::cache_fill body")
if "FILTER.check(evicted_addr, L2C_EVICT, GHR)" not in fill_match.group(1):
    raise SystemExit("SPP cache_fill no longer consumes evicted_addr as audited")
fill_signature_and_body = source[
    source.find("void SPP_dev2::cache_fill"):fill_match.end()
]
for unused in ("addr", "set", "way", "prefetch"):
    if len(re.findall(r"\b{}\b".format(unused), fill_signature_and_body)) != 1:
        raise SystemExit(
            "SPP cache_fill {} is no longer signature-only; revisit input contract".format(
                unused
            )
        )
print("[PASS] audited SPP source sha256={}".format(
    hashlib.sha256(source.encode()).hexdigest()
))
PY
}

ensure_libbf() {
  if [[ -e "$CHAMP_DIR/libbf" && ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    echo "[error] $CHAMP_DIR/libbf exists but is not the expected git checkout" >&2
    exit 2
  fi
  if [[ ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    git clone https://github.com/mavam/libbf.git "$CHAMP_DIR/libbf"
    git -C "$CHAMP_DIR/libbf" checkout --detach "$EXPECTED_LIBBF_HEAD"
  fi
  local observed
  observed="$(git -C "$CHAMP_DIR/libbf" rev-parse HEAD)"
  [[ "$observed" == "$EXPECTED_LIBBF_HEAD" ]] || {
    echo "[error] libbf HEAD $observed != pinned $EXPECTED_LIBBF_HEAD" >&2
    exit 2
  }
  if [[ ! -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]]; then
    cmake -S "$CHAMP_DIR/libbf" -B "$CHAMP_DIR/libbf/build"
    cmake --build "$CHAMP_DIR/libbf/build" -j"$JOBS"
  fi
}

build() {
  command -v flock >/dev/null 2>&1 || {
    echo "[error] flock is required for safe ChampSim builds" >&2
    exit 2
  }
  mkdir -p "$(dirname "$BUILD_LOCK")"
  (
    echo "[build-lock] waiting for $BUILD_LOCK"
    flock -x 9
    echo "[build-lock] acquired by 623 SPP run $RUN_ID"
    audit_spp_source
    RUN_DIR="$RUN_DIR" RESET_PATCH="${RESET_PATCH:-0}" \
      CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] 623 direct-SPP build complete"
  ) 9>"$BUILD_LOCK"
}

assert_live_policy() {
  local log="$1"
  grep -Eiq 'adding L2C_PREFETCHER:.*SPP_dev2' "$log" || {
    echo "[error] live SPP_dev2 was not registered" >&2
    exit 3
  }
  grep -Eq '^fill_threshold: 90$' "$log" || {
    echo "[error] SPP fill threshold is not 90" >&2
    exit 3
  }
  grep -Eq '^pf_threshold: 40$' "$log" || {
    echo "[error] SPP prefetch threshold is not 40" >&2
    exit 3
  }
  grep -Eq '^Core_0_L2C_prefetch_requested [1-9][0-9]*$' "$log" || {
    echo "[error] SPP generated zero direct actions" >&2
    exit 3
  }
  grep -Eq '^Core_0_L2C_prefetch_dropped 0$' "$log" || {
    echo "[error] SPP dropped requests; captured teacher action stream is incomplete" >&2
    exit 3
  }
}

run_policy_events() {
  local role="$1" warmup="$2" simulation="$3"
  local raw="$EVENT_DIR/$TRACE.$POLICY.$role.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/$TRACE.$POLICY.$role.collect.log"
  if [[ "$FORCE" != 1 && -s "$gz" && -s "$log" ]] && gzip -t "$gz"; then
    echo "[skip] $POLICY $role"
    return
  fi
  rm -f "$raw" "$gz"
  DEMAND_EVENT_LOG="$raw" "$BIN" \
    --l2c_prefetcher_types=spp_dev2 \
    --spp_dev2_fill_threshold=90 --spp_dev2_pf_threshold=40 \
    --warmup_instructions="$warmup" \
    --simulation_instructions="$simulation" \
    -traces "$TRACE_FILE" > "$log" 2>&1
  assert_live_policy "$log"
  [[ -s "$raw" ]] || { echo "[error] missing event output for $role" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

assert_collection_count() {
  local role="$1"
  local log="$LOG_DIR/$TRACE.$POLICY.$role.collect.log"
  local stream="$STREAM_DIR/$TRACE.$POLICY.${role}_stream.csv.gz"
  python3 - "$log" "$stream" "$role" <<'PY'
import csv
import gzip
import re
import sys

log_path, stream_path, role = sys.argv[1:]
matches = re.findall(
    r"^Core_0_L2C_loads\s+(\d+)\s*$",
    open(log_path, errors="ignore").read(),
    flags=re.MULTILINE,
)
if not matches:
    raise SystemExit("missing Core_0_L2C_loads in {}".format(log_path))
expected = int(matches[-1])
with gzip.open(stream_path, "rt", newline="") as handle:
    rows = list(csv.DictReader(handle))
observed = sum(row.get("event_kind") == "DEMAND" for row in rows)
fills = sum(row.get("event_kind") == "FILL" for row in rows)
if observed != expected:
    raise SystemExit(
        "SPP {} completed demand callbacks {} != simulator L2 loads {}".format(
            role, observed, expected
        )
    )
if fills <= 0:
    raise SystemExit("SPP {} captured zero cache-fill callbacks".format(role))
print("[PASS] SPP {} demand callbacks={} cache-fill callbacks={}".format(
    role, observed, fills
))
PY
}

collect() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then
    build
  else
    audit_spp_source
    echo "[reuse] existing direct-SPP binary and raw event logs when present"
  fi
  local role warmup simulation
  local input_files=()
  for role in train guard eval; do
    case "$role" in
      train) warmup=0; simulation=20000000 ;;
      guard) warmup=20000000; simulation=5000000 ;;
      eval) warmup=25000000; simulation=25000000 ;;
    esac
    run_policy_events "$role" "$warmup" "$simulation"
    python3 "$NORMALIZE" \
      --events "$EVENT_DIR/$TRACE.$POLICY.$role.events.csv.gz" \
      --policy "$POLICY" \
      --stream-out "$STREAM_DIR/$TRACE.$POLICY.${role}_stream.csv.gz" \
      --teacher-actions-out "$STREAM_DIR/$TRACE.$POLICY.${role}_teacher_actions.csv.gz"
    assert_collection_count "$role"
    input_files+=(
      "$TRACE.$POLICY.${role}_stream.csv.gz"
      "$TRACE.$POLICY.${role}_teacher_actions.csv.gz"
    )
  done
  cp -f "$SOURCE_CONTRACT_REPO" "$SOURCE_CONTRACT_INPUT"
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST" \
    --source-contract "$SOURCE_CONTRACT_INPUT"
  input_files+=("spp_source_contract.json" "collection_manifest.json")
  ( cd "$STREAM_DIR" && sha256sum "${input_files[@]}" > SHA256SUMS )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "${input_files[@]}" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata() {
  python3 - "$1" "$SOURCE_CONTRACT_INPUT" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

metadata = json.load(open(sys.argv[1]))
source_contract = Path(sys.argv[2])
tag = metadata.get("model_tag", "")
family = metadata.get("model_family")
common = {
    "trace": "623.xalancbmk_s-700B",
    "matched_normal_prefetcher": "spp",
    "neural_role": "standalone_direct_action_prefetcher",
    "track_model_family": "cnn",
    "model_does_not_use_pc": True,
    "pc_is_replay_transport_only": True,
    "model_input_is_causal_external_event_sequence_only": True,
    "cache_fill_feedback_used_as_raw_external_input": True,
    "cache_fill_private_state_used_as_model_input": False,
    "cache_hit_and_type_are_audit_only": True,
    "teacher_actions_are_model_inputs": False,
    "same_external_input_contract": True,
    "normal_policy_outputs_used_as_model_inputs": False,
    "normal_policy_candidates_used_as_model_inputs": False,
    "normal_policy_private_state_used_as_model_inputs": False,
    "teacher_action_canonicalization": "per_target_min_fill_queue_effect",
    "training_chunks_shuffled": False,
    "normal_policy_outputs_used_as_training_targets": True,
    "normal_policy_request_rate_used_as_budget": False,
    "normal_policy_constants_used_by_neural_inference": False,
    "probability_threshold_used": False,
    "neural_degree_cap": None,
    "future_label_window_used": False,
    "fill_lead_cutoff_used": False,
    "handcrafted_semantic_features_used": False,
    "manual_loss_weights_used": False,
    "training_regularization_used": False,
    "inference_policy_hardcodes_used": False,
    "learned_request_count": True,
    "causal_no_future_self_test": "PASS",
    "cnn_architecture_self_test": "PASS",
    "event_logger_schema": "623_causal_trigger_fill_v6",
    "action_attachment_mode": "explicit_trigger_event_id",
    "experiment_revision": "spp_threshold_free_fill_feedback_split_v9",
    "replay_preserves_explicit_fill_level": True,
    "source_decision_effective_external_input": [
        "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr"
    ],
    "runtime_feature_count": 65,
}
bad = {key: (metadata.get(key), expected) for key, expected in common.items()
       if metadata.get(key) != expected}
if family != "cnn":
    bad["model_family"] = (family, "cnn")
if not tag.startswith("threshold_free_spp_cnn_"):
    bad["model_tag"] = (tag, "threshold_free_spp_cnn_<size>")
expected_points = {
    ("cnn", 8): ("p0", 4665),
    ("cnn", 13): ("p1", 9240),
    ("cnn", 22): ("p2", 21003),
}
point = expected_points.get((family, metadata.get("model_size")))
if point is None:
    bad["model_point"] = ((family, metadata.get("model_size")), "pinned point")
else:
    if metadata.get("architecture_pair_id") != point[0]:
        bad["architecture_pair_id"] = (metadata.get("architecture_pair_id"), point[0])
    if metadata.get("parameter_count") != point[1]:
        bad["parameter_count"] = (metadata.get("parameter_count"), point[1])
expected = {
    "training_state_mode": "causal_dilated_tcn_over_chronological_stream",
    "training_state_carried_across_chunks": False,
    "training_state_detached_between_chunks": False,
    "inference_history_mode": "chronological_sliding_context_with_exact_1554_event_overlap",
    "cnn_temporal_layers": 4,
    "cnn_kernel_size": 7,
    "cnn_stride": 1,
    "cnn_dilations": [1, 6, 36, 216],
    "cnn_receptive_field_events": 1555,
    "training_left_context_overlap": 1554,
    "cnn_processes_complete_stream_in_order": True,
    "cnn_chunking_changes_visible_history": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        bad[key] = (metadata.get(key), value)
if not source_contract.is_file():
    bad["source_contract"] = ("missing", str(source_contract))
elif metadata.get("source_contract_sha256") != hashlib.sha256(source_contract.read_bytes()).hexdigest():
    bad["source_contract_sha256"] = (
        metadata.get("source_contract_sha256"),
        hashlib.sha256(source_contract.read_bytes()).hexdigest(),
    )
behavior = metadata.get("heldout_behavior_metrics", {})
for key in ("count_exact_match_rate", "target_precision", "target_recall", "target_f1"):
    value = behavior.get(key)
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        bad["heldout_behavior_metrics." + key] = (value, "[0,1]")
fill_accuracy = behavior.get("fill_accuracy_on_matched_targets")
if fill_accuracy is not None and (
    not isinstance(fill_accuracy, (int, float)) or not 0 <= fill_accuracy <= 1
):
    bad["heldout_behavior_metrics.fill_accuracy_on_matched_targets"] = (
        fill_accuracy, "None or [0,1]"
    )

def inspect_replay(path, allow_empty):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["pc", "line", "occ", "prefetch_addr", "fill_level"]:
            raise SystemExit("invalid SPP replay header in {}".format(path))
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 5:
                raise SystemExit("invalid SPP replay row {} in {}".format(line_number, path))
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occ = int(fields[2], 10)
                address = int(fields[3], 0)
                fill_level = int(fields[4], 0)
            except ValueError as exc:
                raise SystemExit("invalid SPP replay integer at {}: {}".format(line_number, exc))
            if min(pc, line, occ, address) < 0 or address % 64:
                raise SystemExit("unaligned/negative SPP replay row {}".format(line_number))
            if fill_level not in (2, 4):
                raise SystemExit("invalid SPP fill level at row {}".format(line_number))
            fill_counts["FILL_L2" if fill_level == 2 else "FILL_LLC"] += 1
            count += 1
    if count <= 0 and not allow_empty:
        raise SystemExit("empty SPP replay list {}".format(path))
    return count, digest, fill_counts

root = Path(sys.argv[1]).parent
for name, count_key, hash_key, fill_key, allow_empty in (
    ("offline_spp.replay.csv", "offline_normal_entries", "normal_list_sha256", "offline_normal_fill_level_counts", False),
    ("offline_nn.replay.csv", "offline_nn_entries", "nn_list_sha256", "offline_nn_fill_level_counts", True),
):
    path = root / name
    if not path.is_file():
        bad[name] = ("missing", "nonempty validated replay list")
        continue
    count, digest, fill_counts = inspect_replay(path, allow_empty)
    if metadata.get(count_key) != count:
        bad[count_key] = (metadata.get(count_key), count)
    if metadata.get(hash_key) != digest:
        bad[hash_key] = (metadata.get(hash_key), digest)
    if metadata.get(fill_key) != fill_counts:
        bad[fill_key] = (metadata.get(fill_key), fill_counts)
if bad:
    raise SystemExit("invalid 623 SPP metadata: {}".format(bad))
PY
}

run_method() {
  local method="$1"
  local log="$LOG_DIR/$TRACE.$method.log"
  local raw="$EVENT_DIR/$TRACE.$method.events.csv"
  local gz="$raw.gz"
  if [[ "$FORCE" != 1 && -s "$log" && -s "$gz" ]] \
      && grep -q '^Core_0_IPC ' "$log" && gzip -t "$gz"; then
    echo "[skip] $method"
    return
  fi
  rm -f "$raw" "$gz"
  case "$method" in
    no_pref)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    live_spp_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=spp_dev2 \
        --spp_dev2_fill_threshold=90 --spp_dev2_pf_threshold=40 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_policy "$log"
      ;;
    offline_spp)
      local list="$(colab_dir "$BASE_TAG")/offline_spp.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer_fill \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_threshold_free_spp_cnn_*)
      local tag="${method#offline_}"
      local list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer_fill \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *) echo "[error] unknown method $method" >&2; exit 2 ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC for $method" >&2; exit 3; }
  [[ -s "$raw" ]] || { echo "[error] missing event output for $method" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

require_colab_outputs() {
  local tag
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST" \
    --source-contract "$SOURCE_CONTRACT_INPUT"
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_spp.replay.csv \
      offline_nn.replay.csv model.pt training_history.csv; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || {
        echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2
        exit 2
      }
    done
    assert_model_metadata "$(colab_dir "$tag")/run_metadata.json"
  done
}

analyze() {
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST" \
    --source-contract "$SOURCE_CONTRACT_INPUT"
  python3 "$ANALYZE" --run-dir "$RUN_DIR" --model-tags "$MODEL_TAGS_CSV"
}

replay() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_spp_reference
  run_method offline_spp
  local tag
  for tag in "${MODEL_TAGS[@]}"; do run_method "offline_$tag"; done
  analyze
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
