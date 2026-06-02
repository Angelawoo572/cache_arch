# Outcome-aware SPP-assisted LSTM Cache Action Predictor

This folder implements the end-to-end flow in the project diagram:

```text
trace + ChampSim
  -> instrumented SPP candidate/context/outcome events
  -> outcome-aware LSTM training table
  -> multi-task LSTM cache-action learner
  -> action CSV / prefetch list
  -> ChampSim replay
  -> offline + system-level metrics
```

## Current framing

SPP is **not** the final policy and not merely a yes/no filter. At this stage SPP provides:

1. candidate address / delta,
2. context features such as confidence, hit/miss, MSHR pressure, and bandwidth pressure,
3. supervision signals such as `outcome_useful` and `outcome_duplicate`.

The LSTM learns cache actions:

```text
good-prefetch probability
bypass / low-priority probability
timing / urgency bucket
auxiliary candidate-delta/address head
```

The key label is now outcome-aware:

```text
good_prefetch = outcome_useful == 1 AND outcome_duplicate == 0
```

The older next-demand-line formulation remains useful as a diagnostic, but it should not be reported as useful-prefetch precision.

## Main files

```text
formal_NN_training/LSTM_cache_action_predictor.ipynb
formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
formal_NN_training/scripts/01_run_spp_trace_dump.sh
formal_NN_training/scripts/train_lstm_cache_action.py
formal_NN_training/scripts/02_actions_to_prefetch_list.py
formal_NN_training/scripts/03_run_lstm_replay.sh
formal_NN_training/scripts/04_eval_lstm_accuracy.py
formal_NN_training/scripts/05_eval_current_label_lstm_vs_spp.py
```

## Default first trace

Use `602.gcc_s-734B` first because it is the most general initial trace in this project: mixed / phase behavior, not purely streaming and not purely pointer-chasing.

Then repeat on:

```text
619.lbm_s-4268B      streaming
605.mcf_s-994B      pointer-chase / irregular
620.omnetpp_s-874B  indirect / graph-like
```

## End-to-end runbook from repo root

### 0. Pull latest code

```bash
cd /scratch/qianruw/cache
git pull --ff-only
```

### 1. Generate SPP candidate/outcome data on cluster

```bash
TRACE=602.gcc_s-734B \
WARMUP=25000000 \
SIM=25000000 \
RESET_SPP=1 \
BUILD=1 \
PATCH_SPP=1 \
bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Expected outputs:

```text
formal_NN_training/results/spp_trace_dump/events/spp_events_602.gcc_s-734B.csv
formal_NN_training/results/spp_trace_dump/candidate_table_602.gcc_s-734B.csv
formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv
```

### 2. Train outcome-aware LSTM

Fast smoke run:

```bash
python3 formal_NN_training/scripts/train_lstm_cache_action.py \
  --trace 602.gcc_s-734B \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --max-rows 500000 \
  --seq-len 32 \
  --epochs 2 \
  --batch-size 128 \
  --hidden-dim 96 \
  --emb-dim 32
```

Main run:

```bash
python3 formal_NN_training/scripts/train_lstm_cache_action.py \
  --trace 602.gcc_s-734B \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --max-rows 2000000 \
  --seq-len 64 \
  --epochs 8 \
  --batch-size 256 \
  --hidden-dim 128 \
  --emb-dim 32 \
  --good-threshold 0.50 \
  --bypass-threshold 0.60
```

Full-data run uses `--max-rows 0`.

Expected outputs:

```text
formal_NN_training/artifacts/outcome_lstm_cache_action_predictor.pt
formal_NN_training/artifacts/outcome_lstm_summary.json
formal_NN_training/artifacts/outcome_lstm_training_history.csv
formal_NN_training/artifacts/outcome_lstm_cache_actions.csv
formal_NN_training/artifacts/full_lstm_cache_actions.csv
```

`full_lstm_cache_actions.csv` is kept as the compatibility name expected by the replay script.

### 3. Offline outcome-aware evaluation

```bash
python3 formal_NN_training/scripts/04_eval_lstm_accuracy.py \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --actions formal_NN_training/artifacts/full_lstm_cache_actions.csv \
  --out formal_NN_training/results/lstm_outcome_accuracy_602.gcc_s-734B.csv \
  --policy action \
  --prefetch-threshold 0.50 \
  --bypass-threshold 0.60
```

Primary metrics to read first:

```text
lstm_good_precision
lstm_good_recall
lstm_good_f1
lstm_duplicate_rate
spp_good_precision
spp_duplicate_rate
```

### 4. Convert actions to list_replayer prefetch list

```bash
python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
  --actions formal_NN_training/artifacts/full_lstm_cache_actions.csv \
  --out formal_NN_training/results/replay_compare/prefetch_lists/prefetch_list_602_gcc_lstm.txt \
  --policy action \
  --prefetch-threshold 0.50 \
  --bypass-threshold 0.60
```

### 5. Replay in ChampSim and compare

```bash
TRACE=602.gcc_s-734B \
WARMUP=25000000 \
SIM=25000000 \
POLICY=action \
PREFETCH_THRESHOLD=0.50 \
BYPASS_THRESHOLD=0.60 \
bash formal_NN_training/scripts/03_run_lstm_replay.sh
```

Expected summary:

```text
formal_NN_training/results/replay_compare/summary_602.gcc_s-734B.csv
```

Compare:

```text
no_prefetch
spp
LSTM_<trace>_action_th0.50_bp0.60
```

## Colab data movement

To restore split input files in Colab:

```bash
TRACE=602.gcc_s-734B bash formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
```

Expected upload layout:

```text
formal_NN_training/data/upload/602/lstm_events_602.gcc_s-734B.csv.gz.part_000 ...
```

To split large outputs manually:

```bash
mkdir -p formal_NN_training/artifacts/packed/602
cd formal_NN_training/artifacts

gzip -c full_lstm_cache_actions.csv > packed/602/full_lstm_cache_actions.csv.gz
split -b 90m packed/602/full_lstm_cache_actions.csv.gz packed/602/full_lstm_cache_actions.csv.gz.part_
```

## What to report

Do not only report `delta_top1`. Report in this order:

1. `good_prefetch precision / recall / F1`,
2. duplicate reduction / duplicate rate,
3. useful prefetch rate and coverage,
4. cache hit rate / MPKI,
5. bandwidth / PQ / MSHR pressure,
6. IPC as the final system-level check.

## Important distinction

Old implemented objective:

```text
predict next demand-line delta / next demand-line address
```

New implemented objective:

```text
learn whether an SPP candidate/action is useful, duplicate, bypass-worthy, or timing-sensitive
using outcome_useful / outcome_duplicate labels
```

The auxiliary delta head remains useful, but it is not the main success criterion.
