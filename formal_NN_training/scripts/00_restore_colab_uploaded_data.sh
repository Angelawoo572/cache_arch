#!/usr/bin/env bash
# 00_restore_colab_uploaded_data.sh
#
# Restore compressed/split training data after git pull in Colab.
#
# Expected upload layout:
#   formal_NN_training/data/upload/602/
#     lstm_events_602.gcc_s-734B.csv.gz.part_000 ...
#     candidate_table_602.gcc_s-734B.csv.gz.part_000 ...
#     spp_events_602.gcc_s-734B.csv.gz.part_000 ...
#
# or unsplit:
#   formal_NN_training/data/upload/602/<name>.csv.gz
#
# Usage from repo root or Colab:
#   TRACE=602.gcc_s-734B bash formal_NN_training/scripts/00_restore_colab_uploaded_data.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
UPLOAD_TAG="${UPLOAD_TAG:-${TRACE%%.*}}"
UPLOAD_DIR="$ROOT/formal_NN_training/data/upload/$UPLOAD_TAG"

OUT_LSTM="$ROOT/formal_NN_training/data/generated/lstm_events_${TRACE}.csv"
OUT_CAND="$ROOT/formal_NN_training/results/spp_trace_dump/candidate_table_${TRACE}.csv"
OUT_EVENTS="$ROOT/formal_NN_training/results/spp_trace_dump/events/spp_events_${TRACE}.csv"

mkdir -p "$(dirname "$OUT_LSTM")" "$(dirname "$OUT_CAND")" "$(dirname "$OUT_EVENTS")"

restore_one () {
  local base="$1"
  local out="$2"
  local gz="$UPLOAD_DIR/${base}.gz"
  local part_glob="$UPLOAD_DIR/${base}.gz.part_"'*'

  if [ -f "$gz" ]; then
    echo "[restore] $gz -> $out"
    gunzip -c "$gz" > "$out"
  elif compgen -G "$part_glob" > /dev/null; then
    echo "[restore] split parts $UPLOAD_DIR/${base}.gz.part_* -> $out"
    cat "$UPLOAD_DIR/${base}.gz.part_"* | gunzip -c > "$out"
  else
    echo "[skip] no compressed upload for $base under $UPLOAD_DIR"
    return 0
  fi

  if [ ! -s "$out" ]; then
    echo "[error] restored file is empty: $out"
    exit 1
  fi
  ls -lh "$out"
}

if [ ! -d "$UPLOAD_DIR" ]; then
  echo "[error] upload directory missing: $UPLOAD_DIR"
  echo "        Expected split .gz files under formal_NN_training/data/upload/$UPLOAD_TAG/"
  exit 1
fi

restore_one "lstm_events_${TRACE}.csv" "$OUT_LSTM"
restore_one "candidate_table_${TRACE}.csv" "$OUT_CAND"
restore_one "spp_events_${TRACE}.csv" "$OUT_EVENTS"

echo
echo "[done] restored available files for TRACE=$TRACE"
echo "  LSTM data      : $OUT_LSTM"
echo "  candidate table: $OUT_CAND"
echo "  SPP events     : $OUT_EVENTS"
