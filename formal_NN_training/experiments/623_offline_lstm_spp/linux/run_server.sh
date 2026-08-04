#!/usr/bin/env bash
# Independent 623 track: normal SPP versus direct SPP-interface LSTM only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
MODEL_POINTS_SCRIPT="$EXP/python/model_points_v19.py"
TRACE="$(python3 "$MODEL_POINTS_SCRIPT" --field trace)"
POLICY="$(python3 "$MODEL_POINTS_SCRIPT" --field policy)"
DEFAULT_RUN_ID="$(python3 "$MODEL_POINTS_SCRIPT" --field run_id)"
DEFAULT_MODEL_TAGS="$(python3 "$MODEL_POINTS_SCRIPT" --tags-csv)"
DEFAULT_BASE_TAG="$(python3 "$MODEL_POINTS_SCRIPT" --base-tag)"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
STAGE="${STAGE:-replay}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-$DEFAULT_MODEL_TAGS}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_direct_spp_lstm_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
VALIDATE_INPUTS="$EXP/python/validate_collected_inputs.py"
ANALYZE="$EXP/python/analyze_replay.py"
INSTALL_COLAB_OUTPUT="$ROOT/formal_NN_training/common/install_colab_output.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"
SOURCE_CONTRACT_REPO="$EXP/data/spp_source_contract.json"
SOURCE_CONTRACT_INPUT="$STREAM_DIR/spp_source_contract.json"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"

require_repo_file() {
  [[ -f "$1" ]] || {
    echo "[error] missing required repository file $1" >&2
    exit 2
  }
}
for required_file in \
  "$PATCH_LOGGER" "$BUILD_REPLAYER" "$NORMALIZE" "$VALIDATE_INPUTS" \
  "$ANALYZE" "$INSTALL_COLAB_OUTPUT" "$SOURCE_CONTRACT_REPO" \
  "$MODEL_POINTS_SCRIPT"; do
  require_repo_file "$required_file"
done

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
    observed = 0
    fills = 0
    for row in csv.DictReader(handle):
        kind = row.get("event_kind")
        if kind == "DEMAND":
            observed += 1
        elif kind == "FILL":
            fills += 1
        else:
            raise SystemExit(
                "SPP {} stream contains invalid event kind {!r}".format(
                    role, kind
                )
            )
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

validate_preserved_inputs() {
  local validated_manifest
  validated_manifest="$(mktemp "$RUN_DIR/.spp_collection_manifest.XXXXXX")"
  if ! python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$validated_manifest" \
    --source-contract "$SOURCE_CONTRACT_INPUT"; then
    rm -f "$validated_manifest"
    return 1
  fi
  if ! cmp -s "$COLLECTION_MANIFEST" "$validated_manifest"; then
    rm -f "$validated_manifest"
    echo "[error] collected SPP input manifest no longer reproduces byte-for-byte" >&2
    return 1
  fi
  rm -f "$validated_manifest"
  ( cd "$STREAM_DIR" && sha256sum -c SHA256SUMS )
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata_v18_legacy() {
  python3 - "$1" "$SOURCE_CONTRACT_INPUT" "$MODEL_POINTS_SCRIPT" <<'PY'
import csv
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
source_contract = Path(sys.argv[2])
point_contract = runpy.run_path(sys.argv[3])["describe_model_points"]()
metadata = json.loads(metadata_path.read_text())
root = metadata_path.parent
tag = metadata.get("model_tag", "")
family = metadata.get("model_family")
source_inputs = point_contract["external_input_fields"]
common = {
    "trace": point_contract["trace"],
    "matched_normal_prefetcher": point_contract["policy"],
    "neural_role": "standalone_direct_action_prefetcher",
    "track_model_family": "lstm",
    "operation": "train-v18",
    "model_revision": "compact_crn_hard_distinct_delta_keyed_fill_v18",
    "decoder_revision": "hard_distinct_delta_keyed_fill_v18",
    "model_does_not_use_pc": True,
    "pc_is_replay_transport_only": True,
    "model_input_is_causal_external_event_sequence_only": True,
    "cache_fill_feedback_used_as_raw_external_input": True,
    "cache_fill_private_state_used_as_model_input": False,
    "cache_hit_and_type_are_audit_only": True,
    "teacher_actions_are_model_inputs": False,
    "same_external_input_contract": True,
    "training_inference_input_encoder_identical": True,
    "decoder_training_mode": "teacher_count_scheduled_loss_with_hard_self_action_feedback",
    "decoder_previous_teacher_action_used_as_input": False,
    "decoder_free_running_self_test": "PASS",
    "teacher_count_role": "schedules_loss_bearing_action_ranks_only",
    "teacher_count_used_as_decoder_feedback": False,
    "training_runtime_fields": source_inputs,
    "inference_runtime_fields": source_inputs,
    "normal_policy_outputs_used_as_model_inputs": False,
    "normal_policy_candidates_used_as_model_inputs": False,
    "normal_policy_private_state_used_as_model_inputs": False,
    "teacher_action_canonicalization": "per_target_min_fill_queue_effect",
    "training_chunks_shuffled": False,
    "normal_policy_outputs_used_as_training_targets": True,
    "normal_policy_request_rate_used_as_budget": False,
    "normal_policy_constants_used_by_neural_inference": False,
    "probability_threshold_used": False,
    "threshold_related_hardcodes_used": False,
    "neural_degree_cap": None,
    "fixed_page_offset_classes": None,
    "same_page_rule_used_by_neural_inference": False,
    "future_label_window_used": False,
    "fill_lead_cutoff_used": False,
    "handcrafted_semantic_features_used": False,
    "manual_loss_weights_used": False,
    "gate_class_weighting_used": False,
    "gate_training_objective": "unweighted_bernoulli_nll",
    "gate_decoding_rule": "deterministic_raw_logit_sign",
    "request_count_training_objective": "unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll",
    "request_count_decoding_rule": "deterministic_raw_hurdle_plus_rounded_conditional_excess_mean",
    "request_count_residual_scope": "none_event_local",
    "joint_delta_fill_dependency_modeled": False,
    "joint_pair_classes": 0,
    "joint_delta_fill_training_objective": None,
    "joint_delta_fill_decoding_rule": None,
    "delta_mixture_components": 4,
    "delta_training_objective": "four_component_signed_log_delta_mixture_nll",
    "delta_mixture_decoding_rule": "component_peak_density_order_then_hard_quantized_legal_delta",
    "fill_training_objective": "unweighted_two_class_cross_entropy",
    "fill_decoding_rule": "event_keyed_categorical_inverse_cdf",
    "fill_argmax_used": False,
    "fill_probability_feedback_used": False,
    "hard_fill_one_hot_feedback_used": True,
    "keyed_fill_uniform_dtype": "float64",
    "address_confidence_fill_heuristic_used": False,
    "delta_decoder_feedback_rule": "actual_hard_quantized_emitted_delta_with_straight_through_training",
    "fill_decoder_feedback_rule": "actual_keyed_hard_fill_one_hot_with_straight_through_training",
    "straight_through_hard_action_feedback_used": True,
    "delta_component_order_score": "log_mixture_mass_minus_log_scale",
    "delta_component_score_tie_break": "ascending_component_index_stable",
    "delta_legality_constraints": [
        "nonzero_signed_delta", "distinct_target_within_callback",
    ],
    "delta_legality_fallback": "nearest_signed_delta_only_if_all_component_means_are_illegal",
    "delta_legality_uses_teacher_or_private_state": False,
    "signed_delta_canonicalization": "58_bit_modulo_with_positive_half_range_mapped_to_negative",
    "decoder_probability_mass_carries_train_guard_history": False,
    "cross_event_probability_credit_used": False,
    "sampled_outputs_used_as_decoder_feedback": True,
    "stochastic_decoding_reproducible": True,
    "training_regularization_used": False,
    "inference_policy_hardcodes_used": False,
    "learned_request_count": True,
    "causal_no_future_self_test": "PASS",
    "deterministic_hurdle_count_self_test": "PASS",
    "hard_distinct_action_feedback_self_test": "PASS",
    "keyed_sampling_self_test": "PASS",
    "factorized_delta_fill_sampling_self_test": "PASS",
    "cnn_architecture_self_test": "NOT_APPLICABLE",
    "event_logger_schema": "623_causal_trigger_fill_v6",
    "action_attachment_mode": "explicit_trigger_event_id",
    "experiment_revision": "spp_source_input_variable_delta_fill_feedback_free_running_v11",
    "replay_preserves_explicit_fill_level": True,
    "source_decision_effective_external_input": source_inputs,
    "runtime_feature_count": 59,
    "runtime_encoding": "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit",
    "same_source_input_offline_claim_allowed": True,
    "closed_loop_live_claim_allowed": False,
    "common_random_numbers_across_capacities": True,
    "strict_common_random_numbers_across_capacities": True,
    "cross_event_rng_state_used": False,
    "decoder_sampling_roles": ["train", "eval"],
    "decoder_train_sampling_performed": True,
    "decoder_guard_sampling_performed": False,
    "decoder_count_sampling_performed": False,
    "guard_selected_decoder": False,
    "joint_map_used": False,
    "weights_retrained": True,
    "checkpoint_reused": False,
    "collection_manifest_role": "historical_input_package_provenance_only",
    "collection_manifest_decoder_fields_are_current_contract": False,
}
bad = {
    key: (metadata.get(key), expected)
    for key, expected in common.items()
    if metadata.get(key) != expected
}
expected_key_fields = [
    "revision", "decoder_seed", "trace", "policy", "role",
    "event_key", "head", "action_rank",
]
sampler = metadata.get("decoder_sampler")
if (
    not isinstance(sampler, dict)
    or sampler.get("sampler_revision")
    != "sha256_event_keyed_inverse_cdf_crn_v1"
    or sampler.get("key_fields") != expected_key_fields
    or sampler.get("poisson_backend") != "scipy.stats.poisson.ppf"
    or sampler.get("cross_event_rng_state") is not False
    or metadata.get("decoder_key_fields") != expected_key_fields
    or metadata.get("decoder_sampler_key_fields") != expected_key_fields
):
    bad["decoder_sampler"] = (
        sampler, "stateless keyed inverse-CDF CRN v1"
    )
for key in (
    "decoder_sampler_source_sha256",
    "decoder_sampler_key_schedule_sha256",
    "decoder_eval_event_key_stream_sha256",
    "decoder_eval_sampling_schedule_sha256",
    "decoder_train_sampling_schedule_sha256",
    "decision_router_source_sha256",
    "train_decision_router_sha256",
    "guard_decision_router_sha256",
    "eval_decision_router_sha256",
    "model_checkpoint_sha256",
    "training_history_sha256",
):
    value = metadata.get(key)
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        bad[key] = (value, "64 lowercase hex characters")
if family != "lstm":
    bad["model_family"] = (family, "lstm")
if not tag.startswith("hard_distinct_delta_fill_spp_lstm_h"):
    bad["model_tag"] = (tag, "hard_distinct_delta_fill_spp_lstm_h<size>")
expected_points = {
    ("lstm", 8): ("p0", 2664),
    ("lstm", 16): ("p1", 6208),
    ("lstm", 32): ("p2", 15984),
    ("lstm", 64): ("p3", 46288),
    ("lstm", 128): ("p4", 149904),
}
point = expected_points.get((family, metadata.get("model_size")))
if point is None:
    bad["model_point"] = (
        (family, metadata.get("model_size")), "pinned v18 point"
    )
else:
    if metadata.get("architecture_pair_id") != point[0]:
        bad["architecture_pair_id"] = (
            metadata.get("architecture_pair_id"), point[0]
        )
    if metadata.get("parameter_count") != point[1]:
        bad["parameter_count"] = (
            metadata.get("parameter_count"), point[1]
        )
encoder_hashes = {
    metadata.get("runtime_encoder_sha256"),
    metadata.get("training_runtime_encoder_sha256"),
    metadata.get("inference_runtime_encoder_sha256"),
}
encoder_hash = (
    next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
)
if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
    bad["runtime_encoder_sha256"] = (
        encoder_hashes, "one shared 64-hex digest"
    )
for key, value in {
    "training_state_mode": "chronological_stateful_tbptt",
    "training_state_carried_across_chunks": True,
    "training_state_detached_between_chunks": True,
    "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
    "cnn_temporal_layers": 0,
}.items():
    if metadata.get(key) != value:
        bad[key] = (metadata.get(key), value)
if not source_contract.is_file():
    bad["source_contract"] = ("missing", str(source_contract))
else:
    observed_source_hash = hashlib.sha256(
        source_contract.read_bytes()
    ).hexdigest()
    if metadata.get("source_contract_sha256") != observed_source_hash:
        bad["source_contract_sha256"] = (
            metadata.get("source_contract_sha256"), observed_source_hash
        )

def inspect_replay(path, allow_empty):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != [
            "pc", "line", "occ", "prefetch_addr", "fill_level"
        ]:
            raise SystemExit("invalid SPP replay header in {}".format(path))
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 5:
                raise SystemExit(
                    "invalid SPP replay row {}".format(line_number)
                )
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occurrence = int(fields[2], 10)
                address = int(fields[3], 0)
                fill_level = int(fields[4], 0)
            except ValueError as exc:
                raise SystemExit(
                    "invalid SPP replay integer at {}: {}".format(
                        line_number, exc
                    )
                )
            if min(pc, line, occurrence, address) < 0 or address % 64:
                raise SystemExit(
                    "unaligned/negative SPP replay row {}".format(line_number)
                )
            if fill_level not in (2, 4):
                raise SystemExit(
                    "invalid SPP fill level at row {}".format(line_number)
                )
            fill_counts[
                "FILL_L2" if fill_level == 2 else "FILL_LLC"
            ] += 1
            count += 1
    if count <= 0 and not allow_empty:
        raise SystemExit("empty SPP replay list {}".format(path))
    return count, digest, fill_counts

for name, count_key, hash_key, fill_key, allow_empty in (
    (
        "offline_spp.replay.csv", "offline_normal_entries",
        "normal_list_sha256", "offline_normal_fill_level_counts", False,
    ),
    (
        "offline_nn.replay.csv", "offline_nn_entries",
        "nn_list_sha256", "offline_nn_fill_level_counts", True,
    ),
):
    path = root / name
    if not path.is_file():
        bad[name] = ("missing", "validated replay list")
        continue
    count, digest, fill_counts = inspect_replay(path, allow_empty)
    if metadata.get(count_key) != count:
        bad[count_key] = (metadata.get(count_key), count)
    if metadata.get(hash_key) != digest:
        bad[hash_key] = (metadata.get(hash_key), digest)
    if metadata.get(fill_key) != fill_counts:
        bad[fill_key] = (metadata.get(fill_key), fill_counts)
for name, key in (
    ("model.pt", "model_checkpoint_sha256"),
    ("training_history.csv", "training_history_sha256"),
):
    path = root / name
    observed = (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file() else None
    )
    if metadata.get(key) != observed:
        bad[key] = (metadata.get(key), observed)
legality = metadata.get("action_legality_diagnostics")
if (
    not isinstance(legality, dict)
    or legality.get("self_target_actions") != 0
    or legality.get("duplicate_target_actions") != 0
    or metadata.get("raw_predicted_action_count")
    != legality.get("raw_predicted_action_count")
    or metadata.get("materialized_distinct_action_count")
    != legality.get("materialized_distinct_action_count")
    or metadata.get("offline_nn_entries")
    != metadata.get("materialized_distinct_action_count")
):
    bad["action_legality_diagnostics"] = (
        legality, "zero self/duplicates and metadata-bound raw/materialized counts"
    )
if bad:
    raise SystemExit("invalid 623 SPP v18 metadata: {}".format(bad))
PY
}

assert_model_metadata_v19() {
  python3 - "$1" "$SOURCE_CONTRACT_INPUT" "$MODEL_POINTS_SCRIPT" <<'PY'
import csv
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
source_contract = Path(sys.argv[2])
point_contract = runpy.run_path(sys.argv[3])["describe_model_points"]()
metadata = json.loads(metadata_path.read_text())
root = metadata_path.parent
source_inputs = point_contract["external_input_fields"]
expected = {
    "run_id": point_contract["run_id"],
    "trace": point_contract["trace"],
    "matched_normal_prefetcher": point_contract["policy"],
    "neural_role": "standalone_direct_action_prefetcher",
    "model_family": "lstm",
    "track_model_family": "lstm",
    "operation": point_contract["operation"],
    "experiment_revision": point_contract["experiment_revision"],
    "model_revision": point_contract["model_revision"],
    "decoder_revision": point_contract["decoder_revision"],
    "source_decision_effective_external_input": source_inputs,
    "training_runtime_fields": source_inputs,
    "inference_runtime_fields": source_inputs,
    "runtime_feature_count": point_contract["runtime_feature_count"],
    "same_external_input_contract": True,
    "training_inference_input_encoder_identical": True,
    "decoder_training_mode": "sampled_rank_grammar_rollout_with_separate_teacher_prefix_output_nll",
    "decoder_previous_teacher_action_used_as_input": True,
    "decoder_previous_teacher_action_used_as_input_scope": "isolated_loss_only_teacher_prefix_likelihood_branch",
    "decoder_previous_teacher_action_used_as_main_rollout_input": False,
    "teacher_count_role": "labels_STOP_or_EMIT_only_at_ranks_reached_by_sampled_rollout",
    "teacher_count_used_as_decoder_feedback": False,
    "teacher_prefix_role": "loss_only_exact_autoregressive_target_likelihood_branch",
    "teacher_prefix_advances_loss_only_likelihood_byte_state": True,
    "teacher_prefix_used_as_main_rollout_recurrent_feedback": False,
    "teacher_target_conditions_loss_only_fill_factor": True,
    "teacher_action_values_used_as_main_rollout_recurrent_feedback": False,
    "model_does_not_use_pc": True,
    "model_input_is_causal_external_event_sequence_only": True,
    "cache_fill_feedback_used_as_raw_external_input": True,
    "teacher_actions_are_model_inputs": False,
    "teacher_actions_are_model_inputs_scope": "external_or_runtime_inference_inputs_only",
    "teacher_actions_used_as_supervised_output_conditioning": True,
    "normal_policy_outputs_used_as_model_inputs": False,
    "normal_policy_candidates_used_as_model_inputs": False,
    "normal_policy_private_state_used_as_model_inputs": False,
    "normal_policy_outputs_used_as_training_targets": True,
    "normal_policy_request_rate_used_as_budget": False,
    "probability_threshold_used": False,
    "threshold_related_hardcodes_used": False,
    "neural_degree_cap": None,
    "gate_training_objective": None,
    "gate_decoding_rule": None,
    "request_count_training_objective": "rankwise_unweighted_stop_emit_categorical_nll",
    "request_count_decoding_rule": "first_keyed_learned_STOP_token_ends_action_sequence",
    "request_count_sampling_performed": True,
    "stop_emit_sampling_rule": "event_rank_keyed_categorical_inverse_cdf",
    "stop_emit_sampler_representability_check": "STOP_mass_strictly_above_open_uniform_half_bin",
    "action_rollout_fail_closed_watchdog_ranks": point_contract["action_rollout_watchdog_ranks"],
    "action_rollout_watchdog_role": "error_without_replay_not_truncation_or_forced_STOP",
    "action_rollout_watchdog_is_neural_degree_cap": False,
    "delta_mixture_components": 0,
    "delta_training_objective": "exact_autoregressive_teacher_prefix_canonical_leb128_nll_with_sampled_history_duplicate_support",
    "delta_decoding_rule": "keyed_exact_signed_zigzag_canonical_leb128",
    "delta_zero_allowed": True,
    "self_target_actions_allowed": True,
    "delta_legality_constraints": ["distinct_target_within_callback"],
    "delta_legality_fallback": None,
    "duplicate_target_handling": "mask_categorical_probability_and_renormalize",
    "duplicate_prefix_feasibility_mask_used": True,
    "fill_training_objective": "unweighted_two_class_cross_entropy_conditioned_on_teacher_target_loss_only",
    "fill_conditioned_on_actual_emitted_target": True,
    "fill_argmax_used": False,
    "optimizer_gradient_normalization": "total_categorical_atom_count_per_accumulation_group",
    "routed_demand_fill_recurrent_paths": True,
    "page_local_causal_state": True,
    "common_random_numbers_across_capacities": True,
    "strict_common_random_numbers_across_capacities": True,
    "cross_event_rng_state_used": False,
    "decoder_sampling_roles": ["train", "eval"],
    "decoder_train_sampling_performed": True,
    "decoder_guard_sampling_performed": False,
    "decoder_count_sampling_performed": True,
    "sampled_outputs_used_as_decoder_feedback": True,
    "decoder_previous_teacher_action_used_as_input": True,
    "weights_retrained": True,
    "checkpoint_reused": False,
    "guard_selected_decoder": False,
    "same_source_input_offline_claim_allowed": True,
    "closed_loop_live_claim_allowed": False,
    "keyed_sampling_self_test": "PASS",
    "rank_stop_emit_grammar_self_test": "PASS",
    "exact_leb128_codec_self_test": "PASS",
    "duplicate_prefix_no_dead_end_self_test": "PASS",
    "teacher_prefix_state_isolation_self_test": "PASS",
    "stop_sampler_representability_self_test": "PASS",
    "always_emit_watchdog_self_test": "PASS",
    "integer_csv_exactness_self_test": "PASS",
    "target_conditioned_fill_self_test": "PASS",
    "routed_page_state_self_test": "PASS",
    "collection_manifest_role": "historical_input_package_provenance_only",
    "collection_manifest_decoder_fields_are_current_contract": False,
}
bad = {
    key: (metadata.get(key), value)
    for key, value in expected.items() if metadata.get(key) != value
}
points = {
    (item["size"], item["pair_id"], item["tag"]): item["parameter_count"]
    for item in point_contract["points"]
}
point = (
    metadata.get("model_size"), metadata.get("architecture_pair_id"),
    metadata.get("model_tag"),
)
if point not in points or metadata.get("parameter_count") != points.get(point):
    bad["model_point"] = (point, "pinned v19 point")
if metadata.get("model_point_contract") != point_contract:
    bad["model_point_contract"] = (metadata.get("model_point_contract"), point_contract)
encoder_hashes = {
    metadata.get("runtime_encoder_sha256"),
    metadata.get("training_runtime_encoder_sha256"),
    metadata.get("inference_runtime_encoder_sha256"),
}
if len(encoder_hashes) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(encoder_hashes), "")) is None:
    bad["runtime_encoder_sha256"] = (encoder_hashes, "one shared SHA256")
for key in (
    "decoder_sampler_source_sha256", "decoder_sampler_key_schedule_sha256",
    "decoder_eval_event_key_stream_sha256", "decoder_eval_sampling_schedule_sha256",
    "decoder_train_sampling_schedule_sha256", "decision_router_source_sha256",
    "train_decision_router_sha256", "guard_decision_router_sha256",
    "eval_decision_router_sha256", "model_checkpoint_sha256",
    "training_history_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) is None:
        bad[key] = (metadata.get(key), "SHA256")
if not source_contract.is_file():
    bad["source_contract"] = (None, str(source_contract))
else:
    observed = hashlib.sha256(source_contract.read_bytes()).hexdigest()
    if metadata.get("source_contract_sha256") != observed:
        bad["source_contract_sha256"] = (metadata.get("source_contract_sha256"), observed)

def replay_info(path, allow_empty):
    count = 0
    fills = {"FILL_L2": 0, "FILL_LLC": 0}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["pc", "line", "occ", "prefetch_addr", "fill_level"]:
            raise SystemExit("invalid replay header {}".format(path))
        for fields in reader:
            if len(fields) != 5 or int(fields[4], 0) not in (2, 4):
                raise SystemExit("invalid replay row {}".format(path))
            fills["FILL_L2" if int(fields[4], 0) == 2 else "FILL_LLC"] += 1
            count += 1
    if not allow_empty and not count:
        raise SystemExit("empty normal replay")
    return count, hashlib.sha256(path.read_bytes()).hexdigest(), fills

for name, count_key, hash_key, fill_key, allow_empty in (
    ("offline_spp.replay.csv", "offline_normal_entries", "normal_list_sha256", "offline_normal_fill_level_counts", False),
    ("offline_nn.replay.csv", "offline_nn_entries", "nn_list_sha256", "offline_nn_fill_level_counts", True),
):
    path = root / name
    if not path.is_file():
        bad[name] = (None, "present")
        continue
    count, digest, fills = replay_info(path, allow_empty)
    for key, actual in ((count_key, count), (hash_key, digest), (fill_key, fills)):
        if metadata.get(key) != actual:
            bad[key] = (metadata.get(key), actual)
legality = metadata.get("action_legality_diagnostics")
if (
    not isinstance(legality, dict)
    or legality.get("duplicate_target_actions") != 0
    or legality.get("self_target_actions_allowed") is not True
    or legality.get("delta_legality_fallback") is not None
    or metadata.get("offline_nn_entries")
       != metadata.get("materialized_distinct_action_count")
    or metadata.get("raw_predicted_action_count")
       != metadata.get("materialized_distinct_action_count")
):
    bad["action_legality_diagnostics"] = (legality, "v19 exact distinct action accounting")
if (
    not isinstance(metadata.get("peak_persistent_recurrent_state_bytes"), int)
    or metadata.get("peak_persistent_recurrent_state_bytes") <= 0
    or not isinstance(metadata.get("dynamic_page_state_pages"), int)
    or metadata.get("dynamic_page_state_pages") <= 0
):
    bad["dynamic_state_accounting"] = (
        metadata.get("peak_persistent_recurrent_state_bytes"),
        metadata.get("dynamic_page_state_pages"),
    )
if bad:
    raise SystemExit("invalid 623 SPP v19 metadata: {}".format(bad))
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
    offline_routed_grammar_spp_lstm_*)
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
  validate_preserved_inputs
  python3 "$INSTALL_COLAB_OUTPUT" \
    --archive "$RUN_DIR/$RUN_ID.colab_output.tar.gz" \
    --output-dir "$COLAB_ROOT" --model-tags "$MODEL_TAGS_CSV"
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_spp.replay.csv \
      offline_nn.replay.csv model.pt training_history.csv; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || {
        echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2
        exit 2
      }
    done
    assert_model_metadata_v19 "$(colab_dir "$tag")/run_metadata.json"
  done
}

analyze() {
  validate_preserved_inputs
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
