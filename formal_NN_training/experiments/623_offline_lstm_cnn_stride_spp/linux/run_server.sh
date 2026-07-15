#!/usr/bin/env bash
# Linux stages for matched 623 stride/SPP LSTM-versus-sliding-CNN tracks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_cnn_stride_spp"
TRACE="623.xalancbmk_s-700B"
RUN_ID="${RUN_ID:-623_offline_lstm_cnn_stride_spp_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-stride_lstm_h4,stride_lstm_h8,stride_lstm_h15,stride_cnn_c8,stride_cnn_c16,stride_cnn_c32,spp_lstm_h4,spp_lstm_h8,spp_lstm_h15,spp_cnn_c8,spp_cnn_c16,spp_cnn_c32}"
STRIDE_BASE_TAG="${STRIDE_BASE_TAG:-stride_lstm_h4}"
SPP_BASE_TAG="${SPP_BASE_TAG:-spp_lstm_h4}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_stride_spp_cnn_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
VALIDATE_INPUTS="$EXP/python/validate_collected_inputs.py"
ANALYZE="$EXP/python/analyze_replay.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"

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
    echo "[build-lock] acquired by 623 stride/SPP run $RUN_ID"
    RESET_PATCH="${RESET_PATCH:-0}" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] 623 stride/SPP build complete"
  ) 9>"$BUILD_LOCK"
}

assert_live_stride() {
  local log="$1"
  grep -Eiq 'adding L2C_PREFETCHER:.*stride' "$log" || {
    echo "[error] live stride was not registered" >&2
    exit 3
  }
  grep -Fq "stride_num_trackers 64" "$log" || {
    echo "[error] stride did not use 64 trackers" >&2
    exit 3
  }
  grep -Fq "stride_pref_degree 2" "$log" || {
    echo "[error] stride did not use degree 2" >&2
    exit 3
  }
  grep -Eq '^stride_pref_generated [1-9][0-9]*$' "$log" || {
    echo "[error] stride generated zero candidates" >&2
    exit 3
  }
}

assert_live_spp() {
  local log="$1"
  grep -Eiq 'adding L2C_PREFETCHER:.*spp' "$log" || {
    echo "[error] live SPP was not registered" >&2
    exit 3
  }
  grep -Eq '^Core_0_L2C_prefetch_requested [1-9][0-9]*$' "$log" || {
    echo "[error] SPP generated zero candidates" >&2
    exit 3
  }
}

run_policy_events() {
  local policy="$1" role="$2" warmup="$3" simulation="$4"
  local raw="$EVENT_DIR/$TRACE.$policy.$role.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/$TRACE.$policy.$role.collect.log"
  if [[ "$FORCE" != 1 && -s "$gz" && -s "$log" ]] && gzip -t "$gz"; then
    echo "[skip] $policy $role"
    return
  fi
  rm -f "$raw" "$gz"
  case "$policy" in
    stride)
      DEMAND_EVENT_LOG="$raw" "$BIN" \
        --l2c_prefetcher_types=stride \
        --stride_num_trackers=64 --stride_pref_degree=2 \
        --warmup_instructions="$warmup" \
        --simulation_instructions="$simulation" \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_stride "$log"
      ;;
    spp)
      DEMAND_EVENT_LOG="$raw" "$BIN" \
        --l2c_prefetcher_types=spp_dev2 \
        --spp_dev2_fill_threshold=90 --spp_dev2_pf_threshold=40 \
        --warmup_instructions="$warmup" \
        --simulation_instructions="$simulation" \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_spp "$log"
      ;;
    *)
      echo "[error] unknown collection policy $policy" >&2
      exit 2
      ;;
  esac
  [[ -s "$raw" ]] || {
    echo "[error] missing event output for $policy $role" >&2
    exit 3
  }
  gzip -f "$raw"
  gzip -t "$gz"
}

assert_collection_count() {
  local policy="$1" role="$2"
  local log="$LOG_DIR/$TRACE.$policy.$role.collect.log"
  local stream="$STREAM_DIR/$TRACE.$policy.${role}_stream.csv.gz"
  python3 - "$log" "$stream" "$policy" "$role" <<'PY'
import csv
import gzip
import re
import sys

log_path, stream_path, policy, role = sys.argv[1:]
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
        "{} {} completed demand callbacks {} != simulator L2 loads {}".format(
            policy, role, observed, expected
        )
    )
print("[PASS] {} {} demand callbacks={}".format(policy, role, observed))
PY
}

collect() {
  [[ -s "$TRACE_FILE" ]] || {
    echo "[error] missing trace $TRACE_FILE" >&2
    exit 2
  }
  build
  local policy role warmup simulation
  local input_files=()
  for policy in stride spp; do
    for role in train guard eval; do
      case "$role" in
        train) warmup=0; simulation=20000000 ;;
        guard) warmup=20000000; simulation=5000000 ;;
        eval) warmup=25000000; simulation=25000000 ;;
      esac
      run_policy_events "$policy" "$role" "$warmup" "$simulation"
      python3 "$NORMALIZE" \
        --events "$EVENT_DIR/$TRACE.$policy.$role.events.csv.gz" \
        --policy "$policy" \
        --stream-out "$STREAM_DIR/$TRACE.$policy.${role}_stream.csv.gz" \
        --candidate-out "$STREAM_DIR/$TRACE.$policy.${role}_candidates.csv.gz"
      assert_collection_count "$policy" "$role"
      input_files+=(
        "$TRACE.$policy.${role}_stream.csv.gz"
        "$TRACE.$policy.${role}_candidates.csv.gz"
      )
    done
  done
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" \
    --manifest-out "$COLLECTION_MANIFEST"
  input_files+=("collection_manifest.json")
  (
    cd "$STREAM_DIR"
    sha256sum "${input_files[@]}" > SHA256SUMS
  )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "${input_files[@]}" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata() {
  python3 - "$1" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1]))
tag = metadata.get("model_tag", "")
policy = tag.split("_", 1)[0]
family = metadata.get("model_family")
common = {
    "trace": "623.xalancbmk_s-700B",
    "matched_normal_prefetcher": policy,
    "model_does_not_use_pc": True,
    "pc_is_replay_transport_only": True,
    "normal_candidate_bank_is_fixed": True,
    "nn_can_only_suppress_normal_candidates": True,
    "training_chunks_shuffled": False,
    "causal_no_future_self_test": "PASS",
    "cnn_architecture_self_test": "PASS",
    "candidate_rank_normalization": (
        "min(candidate_rank, 32) / 32; fixed before data collection"
    ),
    "event_logger_schema": "623_causal_trigger_v4",
    "candidate_attachment_mode": "explicit_trigger_event_id",
    "experiment_revision": "stride_spp_sliding_cnn_v4",
}
bad = {
    key: (metadata.get(key), expected)
    for key, expected in common.items()
    if metadata.get(key) != expected
}
if policy not in {"stride", "spp"}:
    bad["policy_from_tag"] = (policy, "stride_or_spp")
if family == "lstm":
    expected = {
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "cnn_temporal_layers": 0,
    }
elif family == "cnn":
    expected = {
        "training_state_mode": "three_event_causal_sliding_window",
        "training_state_carried_across_chunks": False,
        "training_state_detached_between_chunks": False,
        "cnn_temporal_layers": 1,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_dilation": 1,
        "cnn_receptive_field_events": 3,
        "training_left_context_overlap": 2,
    }
else:
    expected = {"model_family": "lstm_or_cnn"}
for key, value in expected.items():
    if metadata.get(key) != value:
        bad[key] = (metadata.get(key), value)
if bad:
    raise SystemExit("invalid 623 stride/SPP metadata: {}".format(bad))
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
      DEMAND_EVENT_LOG="$raw" "$BIN" \
        --l2c_prefetcher_types=none \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    live_stride_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" \
        --l2c_prefetcher_types=stride \
        --stride_num_trackers=64 --stride_pref_degree=2 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_stride "$log"
      ;;
    live_spp_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" \
        --l2c_prefetcher_types=spp_dev2 \
        --spp_dev2_fill_threshold=90 --spp_dev2_pf_threshold=40 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_spp "$log"
      ;;
    offline_stride)
      local list="$(colab_dir "$STRIDE_BASE_TAG")/offline_stride.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_spp)
      local list="$(colab_dir "$SPP_BASE_TAG")/offline_spp.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_stride_lstm_*|offline_stride_cnn_*|offline_spp_lstm_*|offline_spp_cnn_*)
      local tag="${method#offline_}"
      local list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *)
      echo "[error] unknown method $method" >&2
      exit 2
      ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || {
    echo "[error] missing final IPC for $method" >&2
    exit 3
  }
  [[ -s "$raw" ]] || {
    echo "[error] missing event output for $method" >&2
    exit 3
  }
  gzip -f "$raw"
  gzip -t "$gz"
}

require_colab_outputs() {
  local tag policy
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" \
    --manifest-out "$COLLECTION_MANIFEST"
  for tag in "${MODEL_TAGS[@]}"; do
    policy="${tag%%_*}"
    for name in run_metadata.json "offline_${policy}.replay.csv" \
      offline_nn.replay.csv model.pt policy_sweep.csv; do
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
    --input-dir "$STREAM_DIR" \
    --manifest-out "$COLLECTION_MANIFEST"
  python3 "$ANALYZE" \
    --run-dir "$RUN_DIR" \
    --model-tags "$MODEL_TAGS_CSV"
}

replay() {
  [[ -s "$TRACE_FILE" ]] || {
    echo "[error] missing trace $TRACE_FILE" >&2
    exit 2
  }
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_stride_reference
  run_method live_spp_reference
  run_method offline_stride
  run_method offline_spp
  local tag
  for tag in "${MODEL_TAGS[@]}"; do
    run_method "offline_$tag"
  done
  analyze
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
