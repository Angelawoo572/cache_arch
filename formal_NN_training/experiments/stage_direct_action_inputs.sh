#!/usr/bin/env bash
set -euo pipefail

# Repackage already-collected streams for the independent direct-action runs.
# Every .csv.gz file is copied byte-for-byte and checked with cmp.  Only the
# generated manifest/SHA256SUMS and archive container are new.

usage() {
  cat <<'EOF'
usage: stage_direct_action_inputs.sh MODE SOURCE_RUN NEW_RUN

MODE is one of:
  602-stride  602-streamer  602-ampm  623-stride  623-spp

Example:
  bash formal_NN_training/experiments/stage_direct_action_inputs.sh \
    602-stride 602_offline_lstm_stride_stateful_v2_seed7 \
    602_offline_lstm_stride_direct_v3_seed7
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
  623-stride)
    EXP_NAME="623_offline_lstm_cnn_stride"
    TRACE="623.xalancbmk_s-700B"
    POLICY="stride"
    ROLES=(train guard eval)
    KIND="623-stride"
    ;;
  623-spp)
    EXP_NAME="623_offline_lstm_cnn_spp"
    TRACE="623.xalancbmk_s-700B"
    POLICY="spp"
    ROLES=(train guard eval)
    KIND="623-spp"
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

if [[ "$KIND" == "602" ]]; then
  for role in "${ROLES[@]}"; do
    copy_exact "$TRACE.${role}_stream.csv.gz"
  done
elif [[ "$KIND" == "623-stride" ]]; then
  for role in "${ROLES[@]}"; do
    copy_exact "$TRACE.$POLICY.${role}_stream.csv.gz"
    copy_exact "$TRACE.$POLICY.${role}_candidates.csv.gz"
  done
  python3 "$EXP/python/validate_collected_inputs.py" \
    --input-dir "$DEST_DIR" \
    --manifest-out "$DEST_DIR/collection_manifest.json"
  FILES+=(collection_manifest.json)
else
  for role in "${ROLES[@]}"; do
    copy_exact "$TRACE.$POLICY.${role}_stream.csv.gz"
    copy_exact "$TRACE.$POLICY.${role}_teacher_actions.csv.gz"
  done
  if [[ -s "$SOURCE_DIR/spp_source_contract.json" ]]; then
    copy_exact spp_source_contract.json
  else
    cp -f "$EXP/data/spp_source_contract.json" "$DEST_DIR/spp_source_contract.json"
    FILES+=(spp_source_contract.json)
    echo "[metadata] installed audited SPP source contract"
  fi
  python3 "$EXP/python/validate_collected_inputs.py" \
    --input-dir "$DEST_DIR" \
    --manifest-out "$DEST_DIR/collection_manifest.json" \
    --source-contract "$DEST_DIR/spp_source_contract.json"
  FILES+=(collection_manifest.json)
fi

(
  cd "$DEST_DIR"
  sha256sum "${FILES[@]}" > SHA256SUMS
)

ARCHIVE="$RUN_DIR/$NEW_RUN.colab_input.tar.gz"
tar -C "$DEST_DIR" -czf "$ARCHIVE" "${FILES[@]}" SHA256SUMS
echo "[READY] $ARCHIVE"
echo "[contract] all collected .csv.gz bytes are unchanged; no recollection required"

