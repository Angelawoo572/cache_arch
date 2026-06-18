# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
01_parse_prefetch_behavior_audit.py   # parse counter-level behavior logs; flags failed runs
03_patch_pythia_residual_logger.sh    # patch local Pythia for demand-centric event logging
04_parse_residual_demand_audit.py     # parse demand-centric residual event CSVs
05_run_residual_demand_audit.sh       # residual demand-audit runner for arbitrary base prefetchers
06_run_base_prefetcher_zoo_audit.sh   # main behavior-audit runner for normal prefetchers
07_join_normal_prefetcher_metrics.py  # join behavior + residual summaries into one table
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

Final output for NN planning:

```text
formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv
```

This joined file contains one row per trace/prefetcher with:

```text
behavior_*: IPC, speedup, L2 miss rate/reduction, pf issued/useful/useless/late, accuracy, nodup accuracy, timeliness, failure flags
residual_*: demand miss rate, covered-on-time, coverage among original misses, late rate, residual share, duplicate event rate, event file path
```

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
