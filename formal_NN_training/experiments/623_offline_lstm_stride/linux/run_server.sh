#!/usr/bin/env bash
# Independent 623 track: normal Stride versus the v24 natural-cardinality LSTM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_stride"
TRAINER="$EXP/python/train_and_offline_infer.py"
MODEL_CONTRACT="$EXP/python/model_contract.py"
TRACE="$(python3 "$MODEL_CONTRACT" --field trace)"
POLICY="$(python3 "$MODEL_CONTRACT" --field policy)"
DEFAULT_RUN_ID="$(python3 "$MODEL_CONTRACT" --field run_id)"
PARENT_INPUT_RUN_ID="$(python3 "$MODEL_CONTRACT" --field parent_input_run_id)"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
STAGE="${STAGE:-replay}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_stride_lstm_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
VALIDATE_INPUTS="$EXP/python/validate_collected_inputs.py"
ANALYZE="$EXP/python/analyze_replay.py"
INSTALL_COLAB_OUTPUT="$ROOT/formal_NN_training/common/install_colab_output.py"
SPLIT_COLAB_ARCHIVE="$ROOT/formal_NN_training/common/split_colab_archive.py"
VALIDATE_MODEL_METADATA="$EXP/python/validate_active_metadata.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"

DEFAULT_MODEL_TAGS="$(python3 "$MODEL_CONTRACT" --tags-csv)"
DEFAULT_BASE_TAG="$(python3 "$MODEL_CONTRACT" --base-tag)"
MODEL_TAGS_CSV="${MODEL_TAGS:-$DEFAULT_MODEL_TAGS}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"

[[ "$MODEL_TAGS_CSV" == "$DEFAULT_MODEL_TAGS" ]] || {
  echo "[error] active v24 replay requires the exact five configured MODEL_TAGS" >&2
  exit 2
}
[[ "$BASE_TAG" == "$DEFAULT_BASE_TAG" ]] || {
  echo "[error] active v24 replay requires BASE_TAG=$DEFAULT_BASE_TAG" >&2
  exit 2
}

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
  "$ANALYZE" "$TRAINER" "$MODEL_CONTRACT" "$INSTALL_COLAB_OUTPUT" \
  "$SPLIT_COLAB_ARCHIVE" "$VALIDATE_MODEL_METADATA"; do
  require_repo_file "$required_file"
done

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

build() {
  command -v flock >/dev/null 2>&1 || {
    echo "[error] flock is required for safe ChampSim builds" >&2
    exit 2
  }
  mkdir -p "$(dirname "$BUILD_LOCK")"
  (
    echo "[build-lock] waiting for $BUILD_LOCK"
    flock -x 9
    echo "[build-lock] acquired by 623 stride run $RUN_ID"
    RUN_DIR="$RUN_DIR" RESET_PATCH="${RESET_PATCH:-0}" \
      CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] 623 stride build complete"
  ) 9>"$BUILD_LOCK"
}

assert_live_policy() {
  local log="$1"
  grep -Eiq 'adding L2C_PREFETCHER:.*stride' "$log" || {
    echo "[error] live stride was not registered" >&2
    exit 3
  }
  grep -Fq 'stride_num_trackers 64' "$log" || {
    echo "[error] stride did not use 64 trackers" >&2
    exit 3
  }
  grep -Fq 'stride_pref_degree 2' "$log" || {
    echo "[error] stride did not use degree 2" >&2
    exit 3
  }
  grep -Eq '^stride_pref_generated [1-9][0-9]*$' "$log" || {
    echo "[error] stride generated zero candidates" >&2
    exit 3
  }
  grep -Eq '^Core_0_L2C_prefetch_dropped 0$' "$log" || {
    echo "[error] stride dropped requests; captured candidate bank is incomplete" >&2
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
    --l2c_prefetcher_types=stride \
    --stride_num_trackers=64 --stride_pref_degree=2 \
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
    observed = sum(1 for _ in csv.DictReader(handle))
if observed != expected:
    raise SystemExit(
        "stride {} completed demand callbacks {} != simulator L2 loads {}".format(
            role, observed, expected
        )
    )
print("[PASS] stride {} demand callbacks={}".format(role, observed))
PY
}

collect() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  build
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
      --candidate-out "$STREAM_DIR/$TRACE.$POLICY.${role}_candidates.csv.gz"
    assert_collection_count "$role"
    input_files+=(
      "$TRACE.$POLICY.${role}_stream.csv.gz"
      "$TRACE.$POLICY.${role}_candidates.csv.gz"
    )
  done
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
  input_files+=("collection_manifest.json")
  ( cd "$STREAM_DIR" && sha256sum "${input_files[@]}" > SHA256SUMS )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "${input_files[@]}" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

validate_preserved_inputs() {
  local validated_manifest
  validated_manifest="$(mktemp "$RUN_DIR/.stride_collection_manifest.XXXXXX")"
  if ! python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$validated_manifest"; then
    rm -f "$validated_manifest"
    return 1
  fi
  # The archived collection_manifest.json intentionally records the historical
  # v9 collection semantics.  The current validator adds output-design
  # profiling, so compare bytes through the archived SHA256SUMS and use the
  # fresh manifest as a semantic validation result; do not rewrite reused input.
  rm -f "$validated_manifest"
  ( cd "$STREAM_DIR" && sha256sum -c SHA256SUMS )
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

# Active v24 validation is local to Stride and imports only torch-free modules,
# so the replay host does not need torch or numpy.
assert_model_metadata_v24() {
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
    live_stride_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=stride \
        --stride_num_trackers=64 --stride_pref_degree=2 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_policy "$log"
      ;;
    offline_stride)
      local list="$(colab_dir "$BASE_TAG")/offline_stride.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_natural_cardinality_stride_lstm_*)
      local tag="${method#offline_}"
      local list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
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
  prepare_colab_output_archive
  python3 "$INSTALL_COLAB_OUTPUT" \
    --archive "$RUN_DIR/$RUN_ID.colab_output.tar.gz" \
    --output-dir "$COLAB_ROOT" --model-tags "$MODEL_TAGS_CSV"
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_stride.replay.csv \
      offline_nn.replay.csv model.pt training_history.csv; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || {
        echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2
        exit 2
      }
    done
    assert_model_metadata_v24 "$(colab_dir "$tag")/run_metadata.json"
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
  run_method live_stride_reference
  run_method offline_stride
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
