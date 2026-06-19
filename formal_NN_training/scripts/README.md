# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
01_parse_prefetch_behavior_audit.py   # parse counter-level behavior logs; flags failed runs
03_patch_pythia_residual_logger.sh    # patch local Pythia for demand-centric event logging
04_parse_residual_demand_audit.py     # parse demand-centric residual event CSVs
05_run_residual_demand_audit.sh       # residual demand-audit runner for arbitrary base prefetchers
06_run_base_prefetcher_zoo_audit.sh   # main behavior-audit runner for normal prefetchers
07_join_normal_prefetcher_metrics.py  # join behavior + residual summaries into one table
08_build_normal_prefetcher_oracle_table.py  # build per-access LSTM teacher/oracle tables
```

Removed / merged:

```text
02_run_prefetch_behavior_audit.sh     # removed; superseded by 06_run_base_prefetcher_zoo_audit.sh
17_parse_prefetch_behavior_audit.py   # removed; duplicate of 01_parse_prefetch_behavior_audit.py
```

Default stable normal prefetchers:

```text
no_pref stride streamer ampm spp ipcp sms sandbox power7
```

These are the prefetchers that completed useful 25M/25M behavior runs in the current Pythia fork. Other names from `prefetcher/multi.l2c_pref` can still be tested by overriding `PREFETCHERS`, but are not default because they produced failed/no-final-stat logs in the current setup.

## Full normal-prefetcher metric collection

Run from the repo root on the cluster.

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

This joined file contains one row per trace/prefetcher with:

```text
behavior_*: IPC, speedup, L2 miss rate/reduction, pf issued/useful/useless/late, accuracy, nodup accuracy, timeliness, failure flags
residual_*: demand miss rate, covered-on-time, coverage among original misses, late rate, residual share, duplicate event rate, top residual PCs/deltas, event file path
```

### 4. Build LSTM oracle event tables

Use this after Step 2 has produced residual event files for all working normal prefetchers. The oracle table uses normal prefetchers as teacher labels/diagnostics, not as required runtime model inputs.

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

Output:

```text
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/<trace>.oracle.csv.gz
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table/summary.csv
```

Each oracle row is a demand access from the no-prefetch stream with raw stream features plus normal-prefetcher teacher labels:

```text
raw features: demand_idx, cycle, pc, addr, line, page, page_offset, delta, no_pref_hit/miss
per-prefetcher teacher labels: <pf>_hit, <pf>_miss, <pf>_covered_on_time, <pf>_late, <pf>_mismatch
combined teacher labels: covered_by_any_normal, cover_count, teacher_prefetcher_class, residual_after_all_normal, late_by_any_normal
future labels: future_target_idx, future_distance, future_line, future_delta, future_pc, future_covered_by_any_normal, future_teacher_prefetcher_class, future_residual_after_all_normal
```

This is the LSTM-ready table for a base-independent replacement direction.

Quick view by best speedup:

```bash
python3 - <<'PY'
import csv
from collections import defaultdict
path="formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv"
rows=list(csv.DictReader(open(path)))
by=defaultdict(list)
for r in rows:
    if r.get("behavior_run_failed") == "1":
        continue
    by[r["trace"]].append(r)
for tr, rs in by.items():
    rs=sorted(rs, key=lambda r: float(r.get("behavior_speedup_vs_no_pref") or 0), reverse=True)
    print("\n==", tr, "==")
    for r in rs[:8]:
        print(f'{r["prefetcher"]:10s} speedup={float(r.get("behavior_speedup_vs_no_pref") or 0):.4f} '
              f'miss_red={float(r.get("behavior_miss_reduction_vs_no_pref") or 0):.4f} '
              f'nodup_acc={float(r.get("behavior_nodup_accuracy") or 0):.4f} '
              f'time={float(r.get("behavior_timeliness") or 0):.4f} '
              f'residual_share={float(r.get("residual_residual_share_of_misses") or 0):.4f} '
              f'cov={float(r.get("residual_coverage_among_misses") or 0):.4f}')
PY
```

Note: `05_run_residual_demand_audit.sh` can reuse an already patched Pythia binary with `BUILD=0`. A fresh rebuild uses `03_patch_pythia_residual_logger.sh`.
