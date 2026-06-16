# formal_NN_training/scripts

Supported scripts for the current SPP-assisted LSTM cache-action pipeline.

## Main workflow

```text
Cluster/Sacramento:
  01_run_spp_trace_dump.sh
  05_pack_lstm_events_for_colab.sh
  or the multi-trace wrapper:
  11_run_trace_dump_pack_many.sh

Colab:
  restore upload split files
  train notebook
  export packed action outputs

Cluster/Sacramento:
  07_prepare_actions_for_replay.py
  03_run_lstm_replay.sh for one configuration
  or 12_replay_trace_sweep.sh for multiple thresholds/traces
  13_make_final_figures.py
```

Use `13_make_final_figures.py` to regenerate final CSV/SVG tables from logs without rerunning ChampSim.

## Script list

### 00_restore_colab_uploaded_data.sh

Use in Colab after uploading split `.csv.gz.part_*` input files.

```bash
TRACE=623.xalancbmk_s-700B UPLOAD_TAG=623 \
  bash formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
```

### 01_run_spp_trace_dump.sh

Run ChampSim with SPP candidate logging and create:

```text
formal_NN_training/data/generated/lstm_events_<TRACE>.csv
```

Full run:

```bash
TRACE=623.xalancbmk_s-700B WARMUP=25000000 SIM=25000000 BUILD=0 PATCH_SPP=0 RESET_SPP=1 \
  bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Convert-only after fixing conversion code or reusing an existing event log:

```bash
TRACE=619.lbm_s-4268B CONVERT_ONLY=1 BUILD=0 PATCH_SPP=0 \
  bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Required check after this step: `replay_access_idx` must be nonblank.

### 02_actions_to_prefetch_list.py

Convert `full_lstm_cache_actions.csv` to a ChampSim `list_replayer` prefetch list.

The output must print:

```text
[idx_col] replay_access_idx
```

If it prints `event_id`, the replay is invalid.

Threshold example:

```bash
python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
  --actions formal_NN_training/artifacts/by_trace/623.xalancbmk_s-700B/full_lstm_cache_actions.csv \
  --out formal_NN_training/results/replay_compare/prefetch_lists/prefetch_list_623.txt \
  --policy threshold \
  --prefetch-threshold 0.0005 \
  --bypass-threshold 1.00 \
  --allow-bypass-prefetch
```

### 03_run_lstm_replay.sh

Run one no-prefetch / SPP / LSTM replay configuration for one trace.

```bash
TRACE=619.lbm_s-4268B \
WARMUP=25000000 \
SIM=25000000 \
POLICY=threshold \
PREFETCH_THRESHOLD=0.20 \
BYPASS_THRESHOLD=1.00 \
MODEL_TAG=LSTM_lbm_s-4268B_L2_replayidx_hex_th0.20_bp1.00 \
  bash formal_NN_training/scripts/03_run_lstm_replay.sh
```

### 04_eval_lstm_accuracy.py

Offline candidate/action evaluation against `lstm_events_<TRACE>.csv`. Use this for offline precision/recall/F1 diagnostics only, not for final ChampSim IPC.

### 05_pack_lstm_events_for_colab.sh

Pack `lstm_events_<TRACE>.csv` into gzip split parts for Colab upload. This script verifies that `replay_access_idx` is nonblank.

```bash
TRACE=623.xalancbmk_s-700B UPLOAD_TAG=623 \
  bash formal_NN_training/scripts/05_pack_lstm_events_for_colab.sh
```

Upload files from:

```text
formal_NN_training/data/upload/<tag>/
```

### 06_run_lstm_trace_replay.sh

Post-Colab helper for one trace and one threshold. It restores packed Colab output, prepares actions, runs one replay, then parses metrics.

```bash
TRACE=623.xalancbmk_s-700B \
WARMUP=25000000 \
SIM=25000000 \
PREFETCH_THRESHOLD=0.0005 \
BYPASS_THRESHOLD=1.00 \
ALLOW_BYPASS_PREFETCH=1 \
  bash formal_NN_training/scripts/06_run_lstm_trace_replay.sh
```

For many thresholds/traces, use `12_replay_trace_sweep.sh` instead.

### 07_prepare_actions_for_replay.py

Prepare Colab output for replay. It restores packed split outputs, validates the trace, merges missing `replay_access_idx` by `event_id`, and optionally copies the prepared CSV to the default replay path.

```bash
python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
  --trace 623.xalancbmk_s-700B \
  --restore-packed \
  --copy-default
```

Successful output should include:

```text
missing_event_id=0
addr_mismatch=0
[blank replay_access_idx] 0
[done] actions are ready
```

### 08_scout_candidate_traces.sh

Quick scout across traces to decide which traces are worth full training/replay.

### 09_compare_spp_lstm_accuracy.py

Parse replay logs into a per-trace CSV containing IPC, issued/useful/useless prefetch counts, and useful-per-issued.

```bash
python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace 623.xalancbmk_s-700B \
  --include-lstm LSTM \
  --out formal_NN_training/results/replay_compare/accuracy_compare_623.xalancbmk_s-700B.csv
```

### 10_audit_all_outputs_no_pandas.py

Audit existing outputs and logs without pandas.

### 11_run_trace_dump_pack_many.sh

Reusable multi-trace cluster-side dump and Colab-pack wrapper. Missing trace files are skipped.

```bash
TRACES="623.xalancbmk_s-700B 605.mcf_s-994B" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=2 \
FORCE_DUMP=0 \
  bash formal_NN_training/scripts/11_run_trace_dump_pack_many.sh
```

Use `FORCE_DUMP=1` only when you intentionally want to replace an existing `lstm_events_<TRACE>.csv`.

### 12_replay_trace_sweep.sh

Reusable post-Colab replay sweep for one or more traces and thresholds. This is the preferred script for new traces.

```bash
TRACES="623.xalancbmk_s-700B" \
THRESHOLDS="0p00001:0.00001 0p0001:0.0001 0p0005:0.0005 0p001:0.001" \
ALLOW_BYPASS_PREFETCH=1 \
MAX_JOBS=1 \
WARMUP=25000000 \
SIM=25000000 \
  bash formal_NN_training/scripts/12_replay_trace_sweep.sh
```

For multiple traces:

```bash
TRACES="605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" MAX_JOBS=2 \
  bash formal_NN_training/scripts/12_replay_trace_sweep.sh
```

### 13_make_final_figures.py

Regenerate final CSVs and SVG figures from replay logs. This does not rerun ChampSim.

```bash
TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
  python3 formal_NN_training/scripts/13_make_final_figures.py
```

Outputs:

```text
formal_NN_training/results/final_tables/accuracy_compare_all_traces.csv
formal_NN_training/results/final_tables/normal_best_by_trace.csv
formal_NN_training/results/final_tables/normal_lstm_candidates_filtered.csv
formal_NN_training/results/final_tables/normal_ipc_by_trace.svg
formal_NN_training/results/final_tables/normal_speedup_by_trace.svg
formal_NN_training/results/final_tables/normal_useful_per_issued_by_trace.svg
formal_NN_training/results/capacity_sweep/capacity_sweep_602_619.csv
formal_NN_training/results/capacity_sweep/capacity_sweep_speedup.svg
```

### 14_run_capacity_sweep.sh

Reusable capacity sweep helper, assuming capacity-specific ChampSim binaries already exist.

```bash
TRACES="602.gcc_s-734B 619.lbm_s-4268B" \
CAPS="256K 512K 1M 2M" \
MAX_JOBS=2 \
  bash formal_NN_training/scripts/14_run_capacity_sweep.sh
```

## Metric rule

Use these definitions consistently:

```text
issued-prefetch precision = USEFUL / ISSUED
coverage                  = total USEFUL
performance               = IPC
useful-vs-evicted ratio   = USEFUL / (USEFUL + USELESS)
```

Do not report `USEFUL / (USEFUL + USELESS)` as issued-prefetch precision.
