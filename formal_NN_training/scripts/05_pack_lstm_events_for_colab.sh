#!/usr/bin/env bash
# Pack generated lstm_events_<TRACE>.csv into gzip split parts for Colab upload.
# Run from repo root after 01_run_spp_trace_dump.sh has generated data/generated/lstm_events_<TRACE>.csv.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
TAG="${UPLOAD_TAG:-${TRACE%%.*}}"
SPLIT_SIZE="${SPLIT_SIZE:-90m}"

EVENTS="${EVENTS:-$ROOT/formal_NN_training/data/generated/lstm_events_${TRACE}.csv}"
OUT_DIR="${OUT_DIR:-$ROOT/formal_NN_training/data/upload/${TAG}}"
BASE="lstm_events_${TRACE}.csv.gz"
GZ="$OUT_DIR/$BASE"

if [ ! -s "$EVENTS" ]; then
  echo "[error] missing events CSV: $EVENTS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python3 - <<PY
import csv, sys
p = "$EVENTS"
limit = 100000
with open(p, newline="") as f:
    r = csv.DictReader(f)
    fields = r.fieldnames or []
    if "replay_access_idx" not in fields:
        raise SystemExit("[error] replay_access_idx column missing in %s" % p)
    n = blank = nonblank = 0
    for row in r:
        n += 1
        if row.get("replay_access_idx", "") == "":
            blank += 1
        else:
            nonblank += 1
        if n >= limit:
            break
print("[check] events=%s checked=%d blank_replay_access_idx=%d nonblank=%d" % (p, n, blank, nonblank))
if n == 0 or blank != 0:
    raise SystemExit("[error] replay_access_idx is not ready; do not upload this file to Colab")
PY

rm -f "$OUT_DIR"/"$BASE" "$OUT_DIR"/"$BASE".part_*

echo "[gzip] $EVENTS -> $GZ"
gzip -c "$EVENTS" > "$GZ"

echo "[split] $GZ -> $OUT_DIR/${BASE}.part_* size=$SPLIT_SIZE"
split -b "$SPLIT_SIZE" -d -a 3 "$GZ" "$OUT_DIR/${BASE}.part_"
rm -f "$GZ"

echo "[done] upload these files to Colab:"
ls -lh "$OUT_DIR"/"$BASE".part_*
