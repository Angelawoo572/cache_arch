#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
UPLOAD_TAG="${UPLOAD_TAG:-${TRACE%%.*}}"
SPLIT_SIZE="${SPLIT_SIZE:-90m}"

UPLOAD_DIR="$ROOT/formal_NN_training/data/upload/$UPLOAD_TAG"
mkdir -p "$UPLOAD_DIR"

pack_one () {
  local src="$1"
  local base="$2"

  if [ ! -s "$src" ]; then
    echo "[skip] missing/empty: $src"
    return 0
  fi

  local gz="$UPLOAD_DIR/${base}.gz"
  echo "[gzip] $src -> $gz"
  gzip -c "$src" > "$gz"

  echo "[split] $gz -> ${gz}.part_* size=$SPLIT_SIZE"
  rm -f "${gz}.part_"*
  split -b "$SPLIT_SIZE" -d -a 3 "$gz" "${gz}.part_"

  echo "[remove unsplit gz] $gz"
  rm -f "$gz"

  ls -lh "${gz}.part_"*
}

pack_one \
  "$ROOT/formal_NN_training/data/generated/lstm_events_${TRACE}.csv" \
  "lstm_events_${TRACE}.csv"

pack_one \
  "$ROOT/formal_NN_training/results/spp_trace_dump/candidate_table_${TRACE}.csv" \
  "candidate_table_${TRACE}.csv"

pack_one \
  "$ROOT/formal_NN_training/results/spp_trace_dump/events/spp_events_${TRACE}.csv" \
  "spp_events_${TRACE}.csv"

echo
echo "[done] split upload dir:"
echo "$UPLOAD_DIR"
