#!/usr/bin/env bash
set -euo pipefail

# Reorganize formal_NN_training so future NN families can live side-by-side.
# Run from repo root:
#   bash formal_NN_training/cleanup_lstm_layout.sh
# Then inspect:
#   git status --short

ROOT="formal_NN_training"
LSTM_DIR="$ROOT/LSTM"
RESULT_DIR="$ROOT/results/LSTM/draft"

mkdir -p \
  "$LSTM_DIR/notebooks" \
  "$LSTM_DIR/docs" \
  "$LSTM_DIR/scripts" \
  "$RESULT_DIR"

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
# LSTM notebooks / docs
# -------------------------------------------------------------------
move_if_exists "$ROOT/LSTM_cache_action_predictor.ipynb" \
               "$LSTM_DIR/notebooks/LSTM_cache_action_predictor.ipynb"
move_if_exists "$ROOT/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb" \
               "$LSTM_DIR/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb"
move_if_exists "$ROOT/LSTM_cache_action_pipeline_story.md" \
               "$LSTM_DIR/docs/LSTM_cache_action_pipeline_story.md"
move_if_exists "$ROOT/README_LSTM_cache_action_predictor.md" \
               "$LSTM_DIR/README_LSTM_cache_action_predictor.md"

# -------------------------------------------------------------------
# LSTM scripts kept for rerun / replay. Old duplicate helpers are removed.
# Preferred rerun path after cleanup:
#   01_run_spp_trace_dump.sh -> 05_pack_lstm_events_for_colab.sh -> notebook
#   -> 07_prepare_actions_for_replay.py -> 12_replay_trace_sweep.sh
# -------------------------------------------------------------------
for f in \
  00_restore_colab_uploaded_data.sh \
  01_run_spp_trace_dump.sh \
  02_actions_to_prefetch_list.py \
  04_eval_lstm_accuracy.py \
  05_pack_lstm_events_for_colab.sh \
  07_prepare_actions_for_replay.py \
  08_scout_candidate_traces.sh \
  09_compare_spp_lstm_accuracy.py \
  10_audit_all_outputs_no_pandas.py \
  11_run_trace_dump_pack_many.sh \
  12_replay_trace_sweep.sh \
  13_make_final_figures.py \
  14_run_capacity_sweep.sh \
  15_run_hybrid_replay_suite.sh \
  16_make_hybrid_replay_figures.py \
  local_build_capacity_bins.sh \
  local_parse_capacity_sweep.py \
  local_parse_capacity_sweep_no_pandas.py; do
  move_if_exists "$ROOT/scripts/$f" "$LSTM_DIR/scripts/$f"
done

# Delete superseded / duplicate LSTM helpers.
remove_if_exists "$ROOT/scripts/03_run_lstm_replay.sh"
remove_if_exists "$ROOT/scripts/06_run_lstm_trace_replay.sh"
remove_if_exists "$ROOT/scripts/06_pack_split_for_colab.sh"
remove_if_exists "$ROOT/scripts/README.md"

# -------------------------------------------------------------------
# Draft LSTM outputs. These are old experiment outputs; keep them under
# results/LSTM/draft so new LSTM reruns start cleanly.
# -------------------------------------------------------------------
move_if_exists "$ROOT/results/final_tables" "$RESULT_DIR/final_tables"
move_if_exists "$ROOT/results/replay_compare" "$RESULT_DIR/replay_compare"
move_if_exists "$ROOT/results/capacity_sweep" "$RESULT_DIR/capacity_sweep"
move_if_exists "$ROOT/artifacts" "$RESULT_DIR/artifacts"

# -------------------------------------------------------------------
# Update path references after moving files.
# -------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path

root = Path('formal_NN_training')
repls = {
    'formal_NN_training/LSTM/scripts/': 'formal_NN_training/LSTM/scripts/',
    'formal_NN_training/results/LSTM/draft/replay_compare/': 'formal_NN_training/results/LSTM/draft/replay_compare/',
    'formal_NN_training/results/LSTM/draft/final_tables/': 'formal_NN_training/results/LSTM/draft/final_tables/',
    'formal_NN_training/results/LSTM/draft/capacity_sweep/': 'formal_NN_training/results/LSTM/draft/capacity_sweep/',
    'formal_NN_training/results/LSTM/draft/artifacts/': 'formal_NN_training/results/LSTM/draft/artifacts/',
    'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb': 'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb',
    'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb': 'formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb',
    'formal_NN_training/LSTM/docs/LSTM_cache_action_pipeline_story.md': 'formal_NN_training/LSTM/docs/LSTM_cache_action_pipeline_story.md',
    'formal_NN_training/LSTM/README_LSTM_cache_action_predictor.md': 'formal_NN_training/LSTM/README_LSTM_cache_action_predictor.md',
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

# Add a compact index for the new LSTM area if it does not already exist.
if [ ! -f "$LSTM_DIR/README.md" ]; then
  cat > "$LSTM_DIR/README.md" <<'EOF'
# LSTM cache-action predictor

This folder contains the LSTM/SPP cache-action experiment family.

Layout:

```text
formal_NN_training/LSTM/
  notebooks/   # Colab training notebooks
  scripts/     # trace dump, pack, replay, parse, figure scripts
  docs/        # explanation / story notes

formal_NN_training/results/LSTM/draft/
  artifacts/        # old Colab/model/action outputs
  replay_compare/   # old replay summaries/log-derived CSVs
  capacity_sweep/   # old capacity-sweep outputs
  final_tables/     # old final comparison tables
```

The old result files are intentionally kept as `draft` outputs because the next LSTM run should regenerate fresh artifacts/results.
EOF
  git add "$LSTM_DIR/README.md"
fi

# Add a top-level index for multiple NN families.
cat > "$ROOT/README.md" <<'EOF'
# formal_NN_training

This directory is organized by neural-network family so multiple model ideas can coexist cleanly.

Current active family:

```text
LSTM/                    # LSTM + SPP cache-action predictor code/docs/notebooks
results/LSTM/draft/      # old LSTM results/artifacts kept only as draft history
```

Future model families should use the same pattern:

```text
formal_NN_training/<MODEL_NAME>/
  notebooks/
  scripts/
  docs/

formal_NN_training/results/<MODEL_NAME>/draft/
formal_NN_training/results/<MODEL_NAME>/final/
```
EOF
git add "$ROOT/README.md"

# Remove empty old directories if possible.
find "$ROOT" -type d -empty -delete || true

echo
echo "[done] LSTM files/results reorganized. Inspect with:"
echo "  git status --short"
echo "  find formal_NN_training -maxdepth 4 -type f | sort"
