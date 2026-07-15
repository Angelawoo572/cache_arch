#!/usr/bin/env bash
set -euo pipefail

# Validate one freshly collected 623 policy stream, then copy every normalized
# input byte from its standalone LSTM run into its standalone CNN run.  The
# archive containers have different run names; their contained model inputs,
# labels, manifest, source contract (SPP), and SHA256SUMS are byte-identical.

usage() {
  cat <<'EOF'
usage: stage_623_split_inputs.sh stride|spp

Optional environment overrides:
  CACHE_ROOT  LSTM_RUN_ID  CNN_RUN_ID
EOF
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
POLICY="$1"
CACHE_ROOT="${CACHE_ROOT:-$HOME/cache}"
TRACE="623.xalancbmk_s-700B"

case "$POLICY" in
  stride)
    LSTM_EXP_NAME="623_offline_lstm_stride"
    CNN_EXP_NAME="623_offline_cnn_stride"
    LSTM_RUN_ID="${LSTM_RUN_ID:-623_offline_lstm_stride_threshold_free_v7_seed7}"
    CNN_RUN_ID="${CNN_RUN_ID:-623_offline_cnn_stride_threshold_free_v7_seed7}"
    SUFFIX="candidates"
    EXTRA_FILES=()
    ;;
  spp)
    LSTM_EXP_NAME="623_offline_lstm_spp"
    CNN_EXP_NAME="623_offline_cnn_spp"
    LSTM_RUN_ID="${LSTM_RUN_ID:-623_offline_lstm_spp_threshold_free_v9_seed7}"
    CNN_RUN_ID="${CNN_RUN_ID:-623_offline_cnn_spp_threshold_free_v9_seed7}"
    SUFFIX="teacher_actions"
    EXTRA_FILES=(spp_source_contract.json)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

LSTM_EXP="$CACHE_ROOT/formal_NN_training/experiments/$LSTM_EXP_NAME"
CNN_EXP="$CACHE_ROOT/formal_NN_training/experiments/$CNN_EXP_NAME"
SOURCE_DIR="$LSTM_EXP/runs/$LSTM_RUN_ID/colab_input"
DEST_RUN_DIR="$CNN_EXP/runs/$CNN_RUN_ID"
DEST_DIR="$DEST_RUN_DIR/colab_input"
SOURCE_MANIFEST="$SOURCE_DIR/collection_manifest.json"

[[ -d "$SOURCE_DIR" ]] || {
  echo "[error] collect the LSTM $POLICY track first: $SOURCE_DIR" >&2
  exit 1
}

VALIDATE_SOURCE=(
  python3 "$LSTM_EXP/python/validate_collected_inputs.py"
  --input-dir "$SOURCE_DIR"
  --manifest-out "$SOURCE_MANIFEST"
)
if [[ "$POLICY" == spp ]]; then
  VALIDATE_SOURCE+=(--source-contract "$SOURCE_DIR/spp_source_contract.json")
fi
"${VALIDATE_SOURCE[@]}"

FILES=()
for role in train guard eval; do
  FILES+=(
    "$TRACE.$POLICY.${role}_stream.csv.gz"
    "$TRACE.$POLICY.${role}_${SUFFIX}.csv.gz"
  )
done
FILES+=("${EXTRA_FILES[@]}" collection_manifest.json)

mkdir -p "$DEST_DIR"
for relative in "${FILES[@]}"; do
  source="$SOURCE_DIR/$relative"
  destination="$DEST_DIR/$relative"
  [[ -s "$source" ]] || {
    echo "[error] missing collected input: $source" >&2
    exit 1
  }
  cp -f "$source" "$destination"
  cmp -s "$source" "$destination" || {
    echo "[error] byte mismatch after copying $relative" >&2
    exit 1
  }
  echo "[identical] $(sha256sum "$destination")"
done

VALIDATE_DEST=(
  python3 "$CNN_EXP/python/validate_collected_inputs.py"
  --input-dir "$DEST_DIR"
  --manifest-out "$DEST_DIR/collection_manifest.json"
)
if [[ "$POLICY" == spp ]]; then
  VALIDATE_DEST+=(--source-contract "$DEST_DIR/spp_source_contract.json")
fi
"${VALIDATE_DEST[@]}"

cmp -s "$SOURCE_MANIFEST" "$DEST_DIR/collection_manifest.json" || {
  echo "[error] LSTM/CNN collection manifests differ" >&2
  exit 1
}

(
  cd "$SOURCE_DIR"
  sha256sum "${FILES[@]}" > SHA256SUMS
)
(
  cd "$DEST_DIR"
  sha256sum "${FILES[@]}" > SHA256SUMS
)
cmp -s "$SOURCE_DIR/SHA256SUMS" "$DEST_DIR/SHA256SUMS" || {
  echo "[error] LSTM/CNN SHA256SUMS differ" >&2
  exit 1
}

LSTM_ARCHIVE="$LSTM_EXP/runs/$LSTM_RUN_ID/$LSTM_RUN_ID.colab_input.tar.gz"
CNN_ARCHIVE="$DEST_RUN_DIR/$CNN_RUN_ID.colab_input.tar.gz"
tar -C "$SOURCE_DIR" -czf "$LSTM_ARCHIVE" "${FILES[@]}" SHA256SUMS
tar -C "$DEST_DIR" -czf "$CNN_ARCHIVE" "${FILES[@]}" SHA256SUMS

echo "[READY:LSTM] $LSTM_ARCHIVE"
echo "[READY:CNN]  $CNN_ARCHIVE"
echo "[PASS] $POLICY LSTM/CNN contained inputs are byte-identical"
