# formal_NN_training/scripts

Supported scripts for the current SPP-assisted LSTM cache-action pipeline.

## End-to-end order per trace

```text
01_run_spp_trace_dump.sh
  -> 05_pack_lstm_events_for_colab.sh
  -> Colab / notebook training
  -> copy packed Colab output back to artifacts/packed/<tag>/
  -> 06_run_lstm_trace_replay.sh
```

Use `09_compare_spp_lstm_accuracy.py` separately when you only want to re-parse existing logs.

## Scripts

### 00_restore_colab_uploaded_data.sh

Use in Colab after uploading split `.csv.gz.part_*` input files.

```bash
TRACE=619.lbm_s-4268B UPLOAD_TAG=619 \
  bash formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
```

### 01_run_spp_trace_dump.sh

Run ChampSim with `spp_dev` candidate logging and create `lstm_events_<TRACE>.csv`.

Full run:

```bash
TRACE=619.lbm_s-4268B WARMUP=25000000 SIM=25000000 BUILD=0 PATCH_SPP=0 \
  bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Convert-only after fixing conversion code or reusing an existing event log:

```bash
TRACE=619.lbm_s-4268B CONVERT_ONLY=1 BUILD=0 PATCH_SPP=0 \
  bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Required check after this step:

```bash
python3 - <<'PY'
import csv
TRACE = "619.lbm_s-4268B"
P = f"formal_NN_training/data/generated/lstm_events_{TRACE}.csv"
with open(P, newline="") as f:
    r = csv.DictReader(f)
    n = blank = nonblank = 0
    for row in r:
        n += 1
        if row.get("replay_access_idx", "") == "": blank += 1
        else: nonblank += 1
        if n >= 100000: break
print("checked", n)
print("blank", blank)
print("nonblank", nonblank)
PY
```

`blank` must be `0`.

### 02_actions_to_prefetch_list.py

Convert `full_lstm_cache_actions.csv` to a ChampSim `list_replayer` prefetch list. It must print:

```text
[idx_col] replay_access_idx
```

If it prints `event_id`, the replay is invalid.

### 03_run_lstm_replay.sh

Run no-prefetch, SPP, and LSTM list replay on the same trace/window.

```bash
TRACE=619.lbm_s-4268B WARMUP=25000000 SIM=25000000 \
POLICY=threshold PREFETCH_THRESHOLD=0.20 BYPASS_THRESHOLD=1.00 \
REPL_BIN=/scratch/qianruw/cache/external/ChampSim/bin/champsim.l2_replayer \
MODEL_TAG=LSTM_lbm_s-4268B_L2_replayidx_hex_th0.20_bp1.00 \
  bash formal_NN_training/scripts/03_run_lstm_replay.sh
```

### 04_eval_lstm_accuracy.py

Offline candidate/action evaluation against `lstm_events_<TRACE>.csv`. Use this for offline good-prefetch precision/recall/F1, not for final ChampSim IPC.

### 05_pack_lstm_events_for_colab.sh

Pack `lstm_events_<TRACE>.csv` into gzip split parts for Colab upload. This script first verifies that `replay_access_idx` is nonblank.

```bash
TRACE=619.lbm_s-4268B UPLOAD_TAG=619 \
  bash formal_NN_training/scripts/05_pack_lstm_events_for_colab.sh
```

Upload the generated files from:

```text
formal_NN_training/data/upload/<tag>/
```

### 06_run_lstm_trace_replay.sh

Post-Colab one-command replay helper. Use this after copying the Colab output parts back to:

```text
formal_NN_training/artifacts/packed/<tag>/full_lstm_cache_actions.csv.gz.part_*
```

It runs `07_prepare_actions_for_replay.py`, `03_run_lstm_replay.sh`, and `09_compare_spp_lstm_accuracy.py`.

```bash
TRACE=619.lbm_s-4268B \
WARMUP=25000000 \
SIM=25000000 \
PREFETCH_THRESHOLD=0.20 \
BYPASS_THRESHOLD=1.00 \
REPL_BIN=/scratch/qianruw/cache/external/ChampSim/bin/champsim.l2_replayer \
  bash formal_NN_training/scripts/06_run_lstm_trace_replay.sh
```

### 07_prepare_actions_for_replay.py

Prepare Colab output for replay. It restores packed split outputs, validates the trace, merges missing `replay_access_idx`, and optionally copies the prepared CSV to the default replay path.

```bash
python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
  --trace 619.lbm_s-4268B \
  --copy-default
```

### 08_scout_candidate_traces.sh

Quick scout across traces to decide which traces are worth full training/replay.

### 09_compare_spp_lstm_accuracy.py

Final SPP-vs-LSTM replay metric parser. Use this after `03_run_lstm_replay.sh`.

619 example:

```bash
python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace 619.lbm_s-4268B \
  --include-lstm replayidx \
  --exclude-lstm aligned_hex \
  --out formal_NN_training/results/replay_compare/accuracy_compare_619.lbm_s-4268B.csv
```

602 example:

```bash
python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace 602.gcc_s-734B \
  --include-lstm L2_aligned_hex_th0.20 \
  --out formal_NN_training/results/replay_compare/accuracy_compare_602.gcc_s-734B.csv
```

## Removed obsolete script

`05_eval_current_label_lstm_vs_spp.py` was removed because it evaluated the old next-delta formulation, not the current outcome-aware cache-action objective.

## Metric rule

Use these definitions consistently:

```text
issued-prefetch precision = USEFUL / ISSUED
coverage                  = total USEFUL
performance               = IPC
useful-vs-evicted ratio   = USEFUL / (USEFUL + USELESS)
```

Do not report `USEFUL / (USEFUL + USELESS)` as issued-prefetch precision.
