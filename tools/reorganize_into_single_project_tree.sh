#!/usr/bin/env bash
# Reorganize cache_arch into a single project tree.
#
# This script is intentionally written as one self-contained file so the repo can be
# reorganized reproducibly from main without manually moving files one by one.
#
# Target layout:
#
#   projects/
#     legacy_gru_prefetch/
#       scripts/
#       notebooks/
#       docs/
#       configs/
#       champsim_modules/
#       _cfg/
#       results/              # optional: moved when KEEP_RESULTS_AT_ROOT=0
#     post_prefetch_filter/
#       README.md
#       related_work.md
#       experiment_plan.md
#       scripts/
#
# By default, results/ remains at the repo root because old scripts and large output
# files are easier to inspect there. Set KEEP_RESULTS_AT_ROOT=0 to move it too.
#
# Usage:
#   bash tools/reorganize_into_single_project_tree.sh
#
# Optional:
#   KEEP_RESULTS_AT_ROOT=0 bash tools/reorganize_into_single_project_tree.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

LEGACY="projects/legacy_gru_prefetch"
NEW="projects/post_prefetch_filter"
KEEP_RESULTS_AT_ROOT="${KEEP_RESULTS_AT_ROOT:-1}"

mkdir -p "$LEGACY" "$NEW"
mkdir -p "$LEGACY/scripts" "$LEGACY/notebooks" "$LEGACY/docs" "$LEGACY/configs" "$LEGACY/champsim_modules" "$LEGACY/_cfg"

move_dir() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ]; then
      echo "[skip] $dst already exists; merging contents from $src"
      shopt -s dotglob nullglob
      for item in "$src"/*; do
        mv "$item" "$dst"/
      done
      rmdir "$src" 2>/dev/null || true
      shopt -u dotglob nullglob
    else
      echo "[move] $src -> $dst"
      mv "$src" "$dst"
    fi
  else
    echo "[skip] $src not found"
  fi
}

# Move old experiment folders into one legacy project folder.
move_dir "scripts" "$LEGACY/scripts"
move_dir "notebook" "$LEGACY/notebooks"
move_dir "notebooks" "$LEGACY/notebooks_extra"
move_dir "docs" "$LEGACY/docs"
move_dir "configs" "$LEGACY/configs"
move_dir "_cfg" "$LEGACY/_cfg"
move_dir "champsim_modules" "$LEGACY/champsim_modules"

if [ "$KEEP_RESULTS_AT_ROOT" = "0" ]; then
  move_dir "results" "$LEGACY/results"
else
  echo "[keep] results/ remains at repo root"
fi

# Root-level convenience directories after reorganization.
mkdir -p tools

# Rewrite references inside moved legacy scripts/docs/notebooks.
# Goal: old commands like `bash scripts/run_nn_replay.sh` become paths relative to repo root:
#       `bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh`.
# This is deliberately conservative: it only touches known top-level paths.
rewrite_file() {
  local f="$1"
  [ -f "$f" ] || return 0
  case "$f" in
    *.sh|*.py|*.md|*.tex|*.ipynb|*.json|*.txt|*.csv)
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])scripts/#projects/legacy_gru_prefetch/scripts/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])notebook/#projects/legacy_gru_prefetch/notebooks/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])notebooks/#projects/legacy_gru_prefetch/notebooks/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])docs/#projects/legacy_gru_prefetch/docs/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])configs/#projects/legacy_gru_prefetch/configs/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])_cfg/#projects/legacy_gru_prefetch/_cfg/#g' "$f"
      perl -0pi -e 's#(?<![A-Za-z0-9_./-])champsim_modules/#projects/legacy_gru_prefetch/champsim_modules/#g' "$f"
      ;;
  esac
}

while IFS= read -r -d '' f; do
  rewrite_file "$f"
done < <(find "$LEGACY" "$NEW" -type f -print0 2>/dev/null)

# Fix accidental double rewrites if the script is run twice.
while IFS= read -r -d '' f; do
  case "$f" in
    *.sh|*.py|*.md|*.tex|*.ipynb|*.json|*.txt|*.csv)
      perl -0pi -e 's#projects/legacy_gru_prefetch/projects/legacy_gru_prefetch/#projects/legacy_gru_prefetch/#g' "$f"
      ;;
  esac
done < <(find "$LEGACY" "$NEW" -type f -print0 2>/dev/null)

# Add a root README that explains the new layout, if none exists.
if [ ! -f README.md ]; then
  cat > README.md <<'EOF'
# cache_arch

This repository is organized around two project tracks.

```text
projects/
  legacy_gru_prefetch/      old GRU / NN replay / bypass experiments
  post_prefetch_filter/     new post-prefetch candidate utility filter idea
```

The old top-level folders were moved into `projects/legacy_gru_prefetch/`:

```text
scripts/           -> projects/legacy_gru_prefetch/scripts/
notebook/          -> projects/legacy_gru_prefetch/notebooks/
docs/              -> projects/legacy_gru_prefetch/docs/
configs/           -> projects/legacy_gru_prefetch/configs/
_cfg/              -> projects/legacy_gru_prefetch/_cfg/
champsim_modules/  -> projects/legacy_gru_prefetch/champsim_modules/
```

The new direction is in `projects/post_prefetch_filter/`.
EOF
fi

# Add a local helper after paths move, so the common command is obvious.
cat > "$LEGACY/README.md" <<'EOF'
# Legacy GRU Prefetch Experiments

This folder contains the previous GRU / NN replay / bypass experiments.

Important moved paths:

```text
scripts/          -> projects/legacy_gru_prefetch/scripts/
notebook/         -> projects/legacy_gru_prefetch/notebooks/
docs/             -> projects/legacy_gru_prefetch/docs/
configs/          -> projects/legacy_gru_prefetch/configs/
_cfg/             -> projects/legacy_gru_prefetch/_cfg/
champsim_modules/ -> projects/legacy_gru_prefetch/champsim_modules/
```

Example old command:

```bash
TRACE=602.gcc_s-734B bash scripts/run_gru_v9_decode_sweep.sh
```

New command:

```bash
TRACE=602.gcc_s-734B bash projects/legacy_gru_prefetch/scripts/run_gru_v9_decode_sweep.sh
```
EOF

# Show final tree summary.
echo
printf '[done] reorganized repo under %s\n' "$ROOT"
echo
find projects -maxdepth 3 -type d | sort

echo
echo '[next] review changes:'
echo '  git status --short'
echo '  git diff --stat'
