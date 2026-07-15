#!/usr/bin/env bash
set -euo pipefail

# Repackage already-collected streams for the threshold-free neural runs.
# Every .csv.gz file is copied byte-for-byte and checked with cmp.  Only the
# generated manifest/SHA256SUMS and archive container are new.

usage() {
  cat <<'EOF'
usage: stage_direct_action_inputs.sh MODE SOURCE_RUN NEW_RUN

MODE is one of:
  602-stride  602-streamer  602-ampm

The split 623 tracks intentionally do not reuse the old combined inputs.
Collect once in each standalone LSTM policy directory, then use
stage_623_split_inputs.sh to create the byte-identical CNN input archive.

Example:
  bash formal_NN_training/experiments/stage_direct_action_inputs.sh \
    602-stride 602_offline_lstm_stride_stateful_v2_seed7 \
    602_offline_lstm_stride_threshold_free_v5_seed7
EOF
}

[[ $# -eq 3 ]] || { usage >&2; exit 2; }
MODE="$1"
SOURCE_RUN="$2"
NEW_RUN="$3"
CACHE_ROOT="${CACHE_ROOT:-$HOME/cache}"

case "$MODE" in
  602-stride)
    EXP_NAME="602_offline_lstm_stride"
    TRACE="602.gcc_s-734B"
    POLICY=""
    ROLES=(train eval)
    KIND="602"
    ;;
  602-streamer)
    EXP_NAME="602_offline_lstm_streamer"
    TRACE="602.gcc_s-734B"
    POLICY=""
    ROLES=(train eval)
    KIND="602"
    ;;
  602-ampm)
    EXP_NAME="602_offline_lstm_ampm"
    TRACE="602.gcc_s-734B"
    POLICY=""
    ROLES=(train guard eval)
    KIND="602"
    ;;
  *)
    echo "[error] unknown MODE: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

EXP="$CACHE_ROOT/formal_NN_training/experiments/$EXP_NAME"
SOURCE_DIR="$EXP/runs/$SOURCE_RUN/colab_input"
RUN_DIR="$EXP/runs/$NEW_RUN"
DEST_DIR="$RUN_DIR/colab_input"

[[ -d "$SOURCE_DIR" ]] || {
  echo "[error] source input directory does not exist: $SOURCE_DIR" >&2
  exit 1
}
mkdir -p "$DEST_DIR"

FILES=()
copy_exact() {
  local relative="$1"
  local source="$SOURCE_DIR/$relative"
  local destination="$DEST_DIR/$relative"
  [[ -s "$source" ]] || {
    echo "[error] missing source input: $source" >&2
    exit 1
  }
  cp -f "$source" "$destination"
  cmp -s "$source" "$destination" || {
    echo "[error] byte mismatch after copying $relative" >&2
    exit 1
  }
  FILES+=("$relative")
  echo "[unchanged] $(sha256sum "$destination")"
}

for role in "${ROLES[@]}"; do
  copy_exact "$TRACE.${role}_stream.csv.gz"
done

(
  cd "$DEST_DIR"
  sha256sum "${FILES[@]}" > SHA256SUMS
)

ARCHIVE="$RUN_DIR/$NEW_RUN.colab_input.tar.gz"
tar -C "$DEST_DIR" -czf "$ARCHIVE" "${FILES[@]}" SHA256SUMS
echo "[READY] $ARCHIVE"
echo "[contract] supported-track .csv.gz bytes are unchanged"
