#!/usr/bin/env bash
# organize_results.sh
# Clean up generated experiment outputs while preserving stable script paths.
# Safe to run from the repo root.
#
# Stable tracked config files live under configs/.
# Large/generated experiment outputs stay local-only under results/.

set -uo pipefail

WORKDIR="$(pwd)"
RESULTS="$WORKDIR/results"
LOG_DIR="$RESULTS/logs"
RAW_DIR="$RESULTS/raw"
TMP_DIR="$RESULTS/tmp"
GEN_DIR="$RESULTS/generated"
PREFETCH_DIR="$GEN_DIR/prefetch_lists"
mkdir -p "$RESULTS" "$LOG_DIR" "$RAW_DIR" "$TMP_DIR" "$PREFETCH_DIR"

move_if_exists () {
  local src="$1"
  local dst_dir="$2"
  [ -e "$src" ] || return 0
  mkdir -p "$dst_dir"
  local base
  base="$(basename "$src")"
  if [ -e "$dst_dir/$base" ]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    mv "$src" "$dst_dir/${base}.${ts}"
  else
    mv "$src" "$dst_dir/"
  fi
}

# Logs belong in results/logs/.
for f in "$WORKDIR"/*.log "$RESULTS"/*.log "$RESULTS"/mlp_demo/*.log; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "$LOG_DIR"
done

# Model-generated prefetch lists are large/local outputs. Keep them out of git.
for f in "$WORKDIR"/prefetch_list*.txt "$WORKDIR"/colab_prefetch_result/prefetch_list*.txt; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "$PREFETCH_DIR"
done

# Keep large trace-dumper CSVs in results/ for compatibility with notebooks.
# We do not move access_trace.*.csv automatically because notebooks often expect it.

# Put ad-hoc result text outputs under tmp/ unless they are known tracked summaries.
for f in "$RESULTS"/*.txt; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "$TMP_DIR"
done

cat <<EOF
[organize] done
  logs          -> $LOG_DIR
  prefetch txt  -> $PREFETCH_DIR
  temp txt      -> $TMP_DIR

[status hint]
  git status --short
EOF
