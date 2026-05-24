#!/usr/bin/env bash
# organize_results.sh
# Clean up generated experiment outputs without changing the stable paths that
# the main scripts use. This is safe to run from the repo root.
#
# It does NOT move bypass_pc_list*.txt because run_bypass.sh defaults to the
# repo-root file path.

set -uo pipefail

WORKDIR="$(pwd)"
RESULTS="$WORKDIR/results"
LOG_DIR="$RESULTS/logs"
RAW_DIR="$RESULTS/raw"
TMP_DIR="$RESULTS/tmp"
mkdir -p "$RESULTS" "$LOG_DIR" "$RAW_DIR" "$TMP_DIR"

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
for f in "$RESULTS"/*.log "$RESULTS"/mlp_demo/*.log; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "$LOG_DIR"
done

# Keep large trace-dumper CSVs in results/ for compatibility with notebooks,
# but provide a raw/ location for manual archival if desired. We do not move
# access_trace.*.csv automatically because notebooks often expect this path.

# Put clearly temporary text outputs under tmp/ unless they are known summaries.
for f in "$RESULTS"/*.txt; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "$TMP_DIR"
done

cat <<EOF
[organize] done
  logs -> $LOG_DIR
  temp txt -> $TMP_DIR

[status hint]
  git status --short

[cleanup hint for generated files inside submodules]
  git submodule foreach --recursive 'git clean -fd'
EOF
