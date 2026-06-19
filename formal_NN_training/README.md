# formal_NN_training

This directory is organized by neural-network family while shared simulator scripts stay in the top-level `scripts/` folder.

Current active flow:

```text
scripts/                                      # common Pythia-based audit / run scripts
LSTM/                                         # old/reference LSTM notebooks and new LSTM notebooks
results/LSTM/                                 # old SPP/LSTM-specific outputs
results/base_prefetcher_zoo/                  # normal-prefetcher audit outputs for NN planning
results/base_prefetcher_zoo/oracle_event_table/ # LSTM-ready per-access teacher/oracle tables
```

Active scripts:

```text
formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/03_patch_pythia_residual_logger.sh
formal_NN_training/scripts/04_parse_residual_demand_audit.py
formal_NN_training/scripts/05_run_residual_demand_audit.sh
formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py
formal_NN_training/scripts/08_build_normal_prefetcher_oracle_table.py
```

Removed / merged:

```text
02_run_prefetch_behavior_audit.sh     # removed; 06 is the unified behavior runner
17_parse_prefetch_behavior_audit.py   # removed; duplicate parser replaced by 01
```

## Research direction

The new target is not an SPP-output filter. The target is a base-independent LSTM prefetcher:

```text
raw demand stream -> LSTM learns generalized prefetch policy
```

Normal prefetchers are used as:

```text
1. baselines
2. teachers / oracle labels
3. diagnostic categories
```

They should not be required runtime inputs to the final LSTM. The LSTM input should come from raw stream information such as pc/ip, address line, delta history, page/offset, hit/miss, access type, and later resource-pressure signals.

## Current normal-prefetcher goal

Before changing the LSTM input features or labels, collect the same kind of information for every working normal prefetcher:

```text
counter behavior: IPC, speedup, L2 miss rate, miss reduction, accuracy, nodup accuracy, timeliness, late rate, duplicate proxy
residual behavior: demand miss rate, covered-on-time count, coverage among original misses, late rate, residual miss/share, duplicate event rate
joined planning table: one row per trace/prefetcher with behavior_* and residual_* fields
oracle event table: one row per demand access with raw features and per-prefetcher teacher labels
```

Default stable normal prefetchers:

```text
no_pref stride streamer ampm spp ipcp sms sandbox power7
```

These are the prefetchers that completed useful 25M/25M runs in the current Pythia fork. Other names from the Pythia multi-L2 prefetcher file can still be tested manually by overriding `PREFETCHERS`, but they are not default because they produced failed/no-final-stat logs in this setup.

## Full metric and LSTM-oracle collection commands

Run these from the repo root on the cluster.

### 1. Counter-level behavior audit

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
PREFETCHERS="no_pref stride streamer ampm spp ipcp sms sandbox power7" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=6 \
BUILD=0 \
FORCE_REPLAY=0 \
NODUP=1 \
bash formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
```

Output:

```text
formal_NN_training/results/base_prefetcher_zoo/behavior_audit/logs/
formal_NN_training/results/base_prefetcher_zoo/behavior_audit/summary_nodup.csv
formal_NN_training/results/base_prefetcher_zoo/behavior_audit/RUN_INFO.txt
```

### 2. Demand-centric residual audit

```bash
cd ~/cache

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
PREFETCHERS="no_pref stride streamer ampm spp ipcp sms sandbox power7" \
OUT_ROOT=formal_NN_training/results/base_prefetcher_zoo/residual_audit \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=4 \
FORCE_REPLAY=0 \
BUILD=0 \
COMPRESS=1 \
bash formal_NN_training/scripts/05_run_residual_demand_audit.sh
```

Output:

```text
formal_NN_training/results/base_prefetcher_zoo/residual_audit/events/*.events.csv.gz  # large, ignored
formal_NN_training/results/base_prefetcher_zoo/residual_audit/logs/*.log              # ignored
formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv             # small, tracked
formal_NN_training/results/base_prefetcher_zoo/residual_audit/RUN_INFO.txt            # small, tracked
```

### 3. Join behavior + residual summaries

```bash
python3 formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py \
  --behavior formal_NN_training/results/base_prefetcher_zoo/behavior_audit/summary_nodup.csv \
  --residual formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv \
  --out formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv
```

Final aggregate output for research planning:

```text
formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv
```

### 4. Build LSTM oracle event tables

```bash
python3 formal_NN_training/scripts/08_build_normal_prefetcher_oracle_table.py \
  --event-root formal_NN_training/results/base_prefetcher_zoo/residual_audit/events \
  --out-root formal_NN_training/results/base_prefetcher_zoo/oracle_event_table \
  --summary-out formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/summary.csv \
  --traces "602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
  --prefetchers "stride streamer ampm spp ipcp sms sandbox power7" \
  --max-lookahead 128 \
  --compressed
```

Final LSTM-ready output:

```text
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/602.gcc_s-734B.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/619.lbm_s-4268B.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/605.mcf_s-994B.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/620.omnetpp_s-874B.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/623.xalancbmk_s-700B.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/summary.csv
```

Each oracle row is a demand access from the no-prefetch stream with:

```text
raw features: demand_idx, cycle, pc, addr, line, page, page_offset, delta, no_pref_hit/miss
per-prefetcher teacher labels: <pf>_hit, <pf>_miss, <pf>_covered_on_time, <pf>_late, <pf>_mismatch
combined teacher labels: covered_by_any_normal, cover_count, teacher_prefetcher_class, residual_after_all_normal, late_by_any_normal
future labels: future_target_idx, future_distance, future_line, future_delta, future_pc, future_covered_by_any_normal, future_teacher_prefetcher_class, future_residual_after_all_normal
```

This table is the right input for the base-independent LSTM replacement direction. It uses normal prefetchers as teacher/oracle labels, not as required runtime inputs.

## Current interpretation from the first behavior zoo audit

The first 25M/25M counter-level sweep showed that SPP is not uniformly the best classical base prefetcher.

```text
602.gcc_s-734B:
  best observed base: sandbox
  sandbox speedup ≈ 1.1855, SPP speedup ≈ 1.1640
  Interpretation: SPP is strong, but sandbox/streamer/ampm/power7 are stronger classical baselines. A replacement LSTM must be compared against the best base, not only against SPP.

619.lbm_s-4268B:
  best observed base: sms
  sms speedup ≈ 1.1834, SPP speedup ≈ 1.1780
  Interpretation: SPP+LSTM is still useful as a timing/duplicate case, but SMS is the strongest classical base from the current sweep.

605.mcf_s-994B:
  best observed bases: ampm and spp around 1.0304 speedup
  AMPM has much better miss reduction than SPP, while IPC gain is similar.
  Interpretation: SPP-only residual failure is not purely an LSTM failure; the base prefetcher matters.

620.omnetpp_s-874B:
  best observed base: sms
  sms speedup ≈ 1.0398, SPP speedup ≈ 1.0050
  Interpretation: SPP is the wrong base for this trace if the goal is best classical prefetcher + NN.

623.xalancbmk_s-700B:
  best observed base: spp, but only about 1.0020 speedup
  aggressive prefetchers often hurt IPC.
  Interpretation: this is a conservative/gating trace; naive combination is risky.
```

High-level conclusion:

```text
Use normal prefetchers as teachers and baselines.
Do not hard-code the new LSTM notebook to SPP output.
Train the LSTM from raw access stream features toward oracle/future labels.
```

## Planned matrix

```text
normal prefetcher: trace-specific best base first; SPP retained for comparison
NN:                LSTM first, tiny Transformer later
size/#params:      small / medium / large
seq_len:           64 / 128 / 256
metrics:           accuracy, nodup accuracy, timeliness, demand coverage, residual share, IPC speedup
```
