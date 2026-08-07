#!/usr/bin/env bash
# Independent 623 v25 track: normal SPP versus the active direct-interface LSTM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
MODEL_CONTRACT="$EXP/python/model_contract.py"
TRACE="$(python3 "$MODEL_CONTRACT" --field trace)"
POLICY="$(python3 "$MODEL_CONTRACT" --field policy)"
DEFAULT_RUN_ID="$(python3 "$MODEL_CONTRACT" --field run_id)"
PARENT_INPUT_RUN_ID="$(python3 "$MODEL_CONTRACT" --field parent_input_run_id)"
DEFAULT_MODEL_TAGS="$(python3 "$MODEL_CONTRACT" --tags-csv)"
DEFAULT_BASE_TAG="$(python3 "$MODEL_CONTRACT" --base-tag)"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
STAGE="${STAGE:-replay}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-$DEFAULT_MODEL_TAGS}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"

[[ "$MODEL_TAGS_CSV" == "$DEFAULT_MODEL_TAGS" ]] || {
  echo "[error] active v25 replay requires the exact model_contract MODEL_TAGS" >&2
  exit 2
}
[[ "$BASE_TAG" == "$DEFAULT_BASE_TAG" ]] || {
  echo "[error] active v25 replay requires BASE_TAG=$DEFAULT_BASE_TAG" >&2
  exit 2
}
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
TRAINER="$EXP/python/train_and_offline_infer.py"
THRESHOLD_FREE_POLICY="$ROOT/formal_NN_training/common/threshold_free_policy.py"
INSTALL_COLAB_OUTPUT="$ROOT/formal_NN_training/common/install_colab_output.py"
SPLIT_COLAB_ARCHIVE="$ROOT/formal_NN_training/common/split_colab_archive.py"
VALIDATE_MODEL_METADATA="$EXP/python/validate_active_metadata.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"
SOURCE_CONTRACT_REPO="$EXP/data/spp_source_contract.json"
SOURCE_CONTRACT_INPUT="$STREAM_DIR/spp_source_contract.json"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }

require_safe_path_token() {
  local label="$1" value="$2"
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "[error] $label must be one safe path token: $value" >&2
    exit 2
  }
}

require_safe_path_token RUN_ID "$RUN_ID"
require_safe_path_token BASE_TAG "$BASE_TAG"
seen_model_tags=","
base_tag_is_configured=0
for tag in "${MODEL_TAGS[@]}"; do
  require_safe_path_token MODEL_TAG "$tag"
  [[ "$seen_model_tags" != *",$tag,"* ]] || {
    echo "[error] duplicate MODEL_TAG $tag" >&2
    exit 2
  }
  seen_model_tags+="$tag,"
  [[ "$tag" != "$BASE_TAG" ]] || base_tag_is_configured=1
done
[[ "$base_tag_is_configured" == 1 ]] || {
  echo "[error] BASE_TAG must be one of MODEL_TAGS: $BASE_TAG" >&2
  exit 2
}
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"

reuse_input() {
  local parent_dir="$EXP/runs/$PARENT_INPUT_RUN_ID"
  local parent_stream="$parent_dir/colab_input"
  local parent_archive="$parent_dir/$PARENT_INPUT_RUN_ID.colab_input.tar.gz"
  local archive="$RUN_DIR/$RUN_ID.colab_input.tar.gz"
  [[ -d "$parent_stream" && -s "$parent_archive" ]] || {
    echo "[error] missing v23 parent input under $parent_dir" >&2
    exit 2
  }
  if find "$STREAM_DIR" -mindepth 1 -print -quit | grep -q .; then
    diff -qr "$parent_stream" "$STREAM_DIR"
  else
    cp -a "$parent_stream/." "$STREAM_DIR/"
  fi
  if [[ -e "$archive" ]]; then
    cmp "$parent_archive" "$archive"
  else
    cp -p "$parent_archive" "$archive"
  fi
  gzip -t "$archive"
  validate_preserved_inputs
  echo "[PASS] reused v23 input byte-for-byte for $RUN_ID"
  echo "[ready for Colab] $archive"
}

require_repo_file() {
  [[ -f "$1" ]] || {
    echo "[error] missing required repository file $1" >&2
    exit 2
  }
}
for required_file in \
  "$PATCH_LOGGER" "$BUILD_REPLAYER" "$NORMALIZE" "$VALIDATE_INPUTS" \
  "$ANALYZE" "$TRAINER" "$THRESHOLD_FREE_POLICY" \
  "$INSTALL_COLAB_OUTPUT" "$SPLIT_COLAB_ARCHIVE" "$SOURCE_CONTRACT_REPO" \
  "$MODEL_CONTRACT" "$VALIDATE_MODEL_METADATA"; do
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

prepare_colab_output_archive() {
  local archive="$RUN_DIR/$RUN_ID.colab_output.tar.gz"
  local manifest="$archive.parts.json"
  if [[ -s "$manifest" ]]; then
    python3 "$SPLIT_COLAB_ARCHIVE" join "$manifest" \
      --parts-dir "$RUN_DIR" --output "$archive" --overwrite
  fi
  [[ -s "$archive" ]] || {
    echo "[error] missing $archive or verified multipart manifest $manifest" >&2
    exit 2
  }
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
  python3 "$SPLIT_COLAB_ARCHIVE" split \
    "$RUN_DIR/$RUN_ID.colab_input.tar.gz" --output-dir "$RUN_DIR" \
    --max-part-mib 90 --overwrite
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz.parts.json"
}

validate_preserved_inputs() {
  local validated_manifest installed_validation
  validated_manifest="$(mktemp "$RUN_DIR/.spp_collection_manifest.XXXXXX")"
  if ! python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$validated_manifest" \
    --source-contract "$SOURCE_CONTRACT_INPUT"; then
    rm -f "$validated_manifest"
    return 1
  fi
  # The source collection manifest inside colab_input is historical provenance.
  # Once Colab output is installed, compare its separate fresh v25 validation
  # byte-for-byte with a new Sacramento validation of the preserved inputs.
  installed_validation="$COLAB_ROOT/validated_collection_manifest.json"
  if [[ -s "$installed_validation" ]]; then
    if ! cmp "$validated_manifest" "$installed_validation"; then
      echo "[error] Colab fresh validation differs from current Sacramento input validation" >&2
      rm -f "$validated_manifest"
      return 1
    fi
    echo "[PASS] Colab fresh validation matches current preserved SPP inputs byte-for-byte"
  fi
  rm -f "$validated_manifest"
  # SHA256SUMS independently pins every reused input byte, including the
  # historical collection manifest itself.
  ( cd "$STREAM_DIR" && sha256sum -c SHA256SUMS )
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

# Active v25 validation imports only torch-free modules on the replay host.
assert_active_model_metadata() {
  python3 "$VALIDATE_MODEL_METADATA" \
    --metadata "$1" --input-dir "$STREAM_DIR"
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
    offline_modal_llc_control)
      local list="$(colab_dir "$BASE_TAG")/offline_modal_llc_control.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer_fill \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *)
      local tag="${method#offline_}"
      local configured=0 configured_tag list
      if [[ "$method" == offline_* ]]; then
        for configured_tag in "${MODEL_TAGS[@]}"; do
          [[ "$tag" != "$configured_tag" ]] || configured=1
        done
      fi
      [[ "$configured" == 1 ]] || {
        echo "[error] unknown or unconfigured method $method" >&2
        exit 2
      }
      list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer_fill \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC for $method" >&2; exit 3; }
  [[ -s "$raw" ]] || { echo "[error] missing event output for $method" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

require_colab_outputs() {
  local tag
  prepare_colab_output_archive
  python3 "$INSTALL_COLAB_OUTPUT" \
    --archive "$RUN_DIR/$RUN_ID.colab_output.tar.gz" \
    --output-dir "$COLAB_ROOT" --model-tags "$MODEL_TAGS_CSV"
  validate_preserved_inputs
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_spp.replay.csv \
      offline_nn.replay.csv offline_modal_llc_control.replay.csv \
      model.pt training_history.csv trainer.stdout_stderr.log; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || {
        echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2
        exit 2
      }
    done
    assert_active_model_metadata "$(colab_dir "$tag")/run_metadata.json"
  done
}

run_analyzer() {
  python3 "$ANALYZE" --run-dir "$RUN_DIR" --model-tags "$MODEL_TAGS_CSV"
}

analyze() {
  require_colab_outputs
  run_analyzer
}

replay() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_spp_reference
  run_method offline_spp
  run_method offline_modal_llc_control
  local tag
  for tag in "${MODEL_TAGS[@]}"; do run_method "offline_$tag"; done
  run_analyzer
}

case "$STAGE" in
  reuse-input) reuse_input ;;
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be reuse-input, build, collect, replay, or analyze" >&2; exit 2 ;;
esac
