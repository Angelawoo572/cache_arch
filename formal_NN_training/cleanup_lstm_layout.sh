#!/usr/bin/env bash
set -euo pipefail

# Final formal_NN_training layout cleanup.
# Run from repo root:
#   bash formal_NN_training/cleanup_lstm_layout.sh
# Then inspect and commit:
#   git status --short
#
# Desired layout:
#   formal_NN_training/LSTM/notebooks/      # LSTM notebooks only
#   formal_NN_training/scripts/             # shared/common workflow scripts
#   formal_NN_training/*.md                 # LSTM notes stay visible at top level
#   formal_NN_training/results/LSTM/draft/  # old LSTM outputs kept as draft history

ROOT="formal_NN_training"
LSTM_DIR="$ROOT/LSTM"
RESULT_DIR="$ROOT/results/LSTM/draft"
SCRIPT_DIR="$ROOT/scripts"

mkdir -p "$LSTM_DIR/notebooks" "$SCRIPT_DIR" "$RESULT_DIR"

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    git mv "$src" "$dst"
    echo "[move] $src -> $dst"
  fi
}

remove_if_exists() {
  local p="$1"
  if [ -e "$p" ]; then
    git rm -r "$p"
    echo "[remove] $p"
  fi
}

# -------------------------------------------------------------------
# 1. Keep only notebooks inside LSTM/. No LSTM/docs and no LSTM/scripts.
# -------------------------------------------------------------------
move_if_exists "$ROOT/LSTM_cache_action_predictor.ipynb" \
               "$LSTM_DIR/notebooks/LSTM_cache_action_predictor.ipynb"
move_if_exists "$ROOT/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb" \
               "$LSTM_DIR/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb"

# User preference: keep these two .md files at formal_NN_training/ top level.
move_if_exists "$LSTM_DIR/docs/LSTM_cache_action_pipeline_story.md" \
               "$ROOT/LSTM_cache_action_pipeline_story.md"
move_if_exists "$LSTM_DIR/README_LSTM_cache_action_predictor.md" \
               "$ROOT/README_LSTM_cache_action_predictor.md"
move_if_exists "$ROOT/LSTM_cache_action_pipeline_story.md" \
               "$ROOT/LSTM_cache_action_pipeline_story.md"
move_if_exists "$ROOT/README_LSTM_cache_action_predictor.md" \
               "$ROOT/README_LSTM_cache_action_predictor.md"

# User preference: scripts are shared/common, so keep the whole scripts folder at root.
if [ -d "$LSTM_DIR/scripts" ]; then
  mkdir -p "$SCRIPT_DIR"
  shopt -s nullglob dotglob
  for p in "$LSTM_DIR/scripts"/*; do
    move_if_exists "$p" "$SCRIPT_DIR/$(basename "$p")"
  done
  shopt -u nullglob dotglob
fi

# Delete superseded duplicate helpers if they still exist.
remove_if_exists "$SCRIPT_DIR/03_run_lstm_replay.sh"
remove_if_exists "$SCRIPT_DIR/06_run_lstm_trace_replay.sh"
remove_if_exists "$SCRIPT_DIR/06_pack_split_for_colab.sh"

# profile_lstm_events_no_pandas.py is not needed as a separate file after the scout script is self-contained.
find "$ROOT" -type f -name 'profile_lstm_events_no_pandas.py' -print0 | while IFS= read -r -d '' f; do
  git rm "$f"
  echo "[remove] $f"
done

# -------------------------------------------------------------------
# 2. Old LSTM outputs stay under results/LSTM/draft.
# -------------------------------------------------------------------
move_if_exists "$ROOT/results/final_tables" "$RESULT_DIR/final_tables"
move_if_exists "$ROOT/results/replay_compare" "$RESULT_DIR/replay_compare"
move_if_exists "$ROOT/results/capacity_sweep" "$RESULT_DIR/capacity_sweep"
move_if_exists "$ROOT/artifacts" "$RESULT_DIR/artifacts"

# Delete explicitly unwanted stale bulky outputs.
remove_if_exists "$ROOT/backup_scout_623"
remove_if_exists "$ROOT/data/backup_scout_623"
remove_if_exists "$ROOT/data/generated/backup_scout_623"
remove_if_exists "$ROOT/data/upload/backup_scout_623"
find "$ROOT" -type d -name 'backup_scout_623' -print0 | while IFS= read -r -d '' d; do
  git rm -r "$d"
  echo "[remove] $d"
done
find "$ROOT" -type f \( -name 'lstm_events_602.gcc_s-734B.csv.gz' -o -name 'upload_605_620_for_colab.tar.gz' \) -print0 | while IFS= read -r -d '' f; do
  git rm "$f"
  echo "[remove] $f"
done

# -------------------------------------------------------------------
# 3. Rewrite path references to the final layout.
# -------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path

root = Path('formal_NN_training')
repls = {
    'formal_NN_training/LSTM/scripts/': 'formal_NN_training/scripts/',
    'formal_NN_training/LSTM/docs/LSTM_cache_action_pipeline_story.md': 'formal_NN_training/LSTM_cache_action_pipeline_story.md',
    'formal_NN_training/LSTM/README_LSTM_cache_action_predictor.md': 'formal_NN_training/README_LSTM_cache_action_predictor.md',
    'formal_NN_training/results/replay_compare/': 'formal_NN_training/results/LSTM/draft/replay_compare/',
    'formal_NN_training/results/final_tables/': 'formal_NN_training/results/LSTM/draft/final_tables/',
    'formal_NN_training/results/capacity_sweep/': 'formal_NN_training/results/LSTM/draft/capacity_sweep/',
    'formal_NN_training/artifacts/': 'formal_NN_training/results/LSTM/draft/artifacts/',
    'formal_NN_training/LSTM_cache_action_predictor.ipynb': 'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb',
    'formal_NN_training/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb': 'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb',
}

suffixes = {'.md', '.py', '.sh', '.ipynb', '.txt', '.json', '.csv'}
for p in root.rglob('*'):
    if not p.is_file() or p.suffix not in suffixes:
        continue
    try:
        s = p.read_text()
    except UnicodeDecodeError:
        continue
    old = s
    for a, b in repls.items():
        s = s.replace(a, b)
    if s != old:
        p.write_text(s)
        print('[rewrite]', p)
PY

# -------------------------------------------------------------------
# 4. Make 08_scout_candidate_traces.sh self-contained so it no longer needs
#    profile_lstm_events_no_pandas.py.
# -------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path
p = Path('formal_NN_training/scripts/08_scout_candidate_traces.sh')
if p.exists():
    s = p.read_text()
    old = '''  python3 formal_NN_training/scripts/profile_lstm_events_no_pandas.py \\
    --csv "$OUT" \\
    --trace "$T" \\
    --out "$SUMMARY" \\
    --append'''
    new = '''  python3 - "$OUT" "$T" "$SUMMARY" <<'PYSCOUT'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
trace = sys.argv[2]
summary = Path(sys.argv[3])

fields = [
    "trace", "rows", "useful", "duplicate", "issued",
    "useful_rate", "duplicate_rate", "issued_rate",
]
rows = useful = duplicate = issued = 0

with csv_path.open(newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows += 1
        useful += int(row.get("outcome_useful", "0") or 0)
        duplicate += int(row.get("outcome_duplicate", "0") or 0)
        issued += int(row.get("spp_issued", "0") or 0)

def rate(x):
    return "0.000000" if rows == 0 else f"{x / rows:.6f}"

summary.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary.exists() or summary.stat().st_size == 0
with summary.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
    w.writerow({
        "trace": trace,
        "rows": rows,
        "useful": useful,
        "duplicate": duplicate,
        "issued": issued,
        "useful_rate": rate(useful),
        "duplicate_rate": rate(duplicate),
        "issued_rate": rate(issued),
    })
PYSCOUT'''
    if old in s:
        s = s.replace(old, new)
        p.write_text(s)
        print('[rewrite scout self-contained]', p)
PY

# -------------------------------------------------------------------
# 5. Write compact indexes for the final layout.
# -------------------------------------------------------------------
cat > "$ROOT/README.md" <<'EOF'
# formal_NN_training

Organized by model family, with shared scripts at the top level.

```text
formal_NN_training/
  README_LSTM_cache_action_predictor.md
  LSTM_cache_action_pipeline_story.md
  scripts/                  # shared/common dump, pack, replay, parse scripts
  LSTM/
    notebooks/              # LSTM Colab notebooks only
  results/
    LSTM/
      draft/                # old LSTM artifacts/results; regenerate for new runs
```

Future NN families should add their own notebook/model folder, but reuse or extend `formal_NN_training/scripts/` when the workflow is shared.
EOF
git add "$ROOT/README.md"

cat > "$LSTM_DIR/README.md" <<'EOF'
# LSTM notebooks

This folder only keeps LSTM-specific notebooks.

Top-level LSTM notes stay in:

```text
formal_NN_training/README_LSTM_cache_action_predictor.md
formal_NN_training/LSTM_cache_action_pipeline_story.md
```

Shared scripts stay in:

```text
formal_NN_training/scripts/
```
EOF
git add "$LSTM_DIR/README.md"

find "$ROOT" -type d -empty -delete || true

echo
echo "[done] final formal_NN_training layout prepared. Inspect with:"
echo "  git status --short"
echo "  find formal_NN_training -maxdepth 4 -type f | sort"
