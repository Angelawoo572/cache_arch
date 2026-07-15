#!/usr/bin/env bash
# Independent matched 623 stride track: normal stride versus LSTM/CNN gates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_cnn_stride"
TRACE="623.xalancbmk_s-700B"
POLICY="stride"
RUN_ID="${RUN_ID:-623_offline_lstm_cnn_stride_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-stride_lstm_h4,stride_lstm_h8,stride_lstm_h15,stride_cnn_c8,stride_cnn_c16,stride_cnn_c32}"
BASE_TAG="${BASE_TAG:-stride_lstm_h4}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_stride_cnn_replay}"
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

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata() {
  python3 - "$1" "$POLICY" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

metadata = json.load(open(sys.argv[1]))
policy = sys.argv[2]
tag = metadata.get("model_tag", "")
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
    "candidate_rank_normalization": "min(candidate_rank, 32) / 32; fixed before data collection",
    "event_logger_schema": "623_causal_trigger_v5",
    "candidate_attachment_mode": "explicit_trigger_event_id",
    "experiment_revision": "stride_sliding_cnn_v1",
    "neural_role": "stride_candidate_gate",
    "normal_policy_private_state_is_not_nn_input": True,
    "captured_fill_level_is_replay_action_metadata_not_nn_input": True,
}
bad = {key: (metadata.get(key), expected) for key, expected in common.items()
       if metadata.get(key) != expected}
if not tag.startswith(policy + "_"):
    bad["model_tag"] = (tag, policy + "_<family>_<size>")
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

def inspect_replay(path, allow_empty):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["pc", "line", "occ", "prefetch_addr"]:
            raise SystemExit("invalid stride replay header in {}".format(path))
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 4:
                raise SystemExit("invalid stride replay row {} in {}".format(line_number, path))
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occ = int(fields[2], 10)
                address = int(fields[3], 0)
            except ValueError as exc:
                raise SystemExit("invalid stride replay integer at {}: {}".format(line_number, exc))
            if min(pc, line, occ, address) < 0 or address % 64:
                raise SystemExit("unaligned/negative stride replay row {}".format(line_number))
            count += 1
    if count <= 0 and not allow_empty:
        raise SystemExit("empty stride replay list {}".format(path))
    return count, digest

root = Path(sys.argv[1]).parent
for name, count_key, hash_key, allow_empty in (
    ("offline_stride.replay.csv", "offline_normal_entries", "normal_list_sha256", False),
    ("offline_nn.replay.csv", "offline_nn_entries", "nn_list_sha256", True),
):
    path = root / name
    if not path.is_file():
        bad[name] = ("missing", "nonempty validated replay list")
        continue
    count, digest = inspect_replay(path, allow_empty)
    if metadata.get(count_key) != count:
        bad[count_key] = (metadata.get(count_key), count)
    if metadata.get(hash_key) != digest:
        bad[hash_key] = (metadata.get(hash_key), digest)
if bad:
    raise SystemExit("invalid 623 stride metadata: {}".format(bad))
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
    offline_stride_lstm_*|offline_stride_cnn_*)
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
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_stride.replay.csv \
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
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
  python3 "$ANALYZE" --run-dir "$RUN_DIR" --model-tags "$MODEL_TAGS_CSV"
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
  analyze
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
