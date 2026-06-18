# formal_NN_training

This directory is organized by neural-network family while shared simulator scripts stay in the top-level `scripts/` folder.

Current active flow:

```text
scripts/                                      # common Pythia-based audit / run scripts
LSTM/                                         # old/reference LSTM notebooks and new LSTM notebooks
results/LSTM/                                 # old SPP/LSTM-specific outputs
results/base_prefetcher_zoo/                  # normal-prefetcher audit outputs for NN planning
```

Active scripts:

```text
formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/03_patch_pythia_residual_logger.sh
formal_NN_training/scripts/04_parse_residual_demand_audit.py
formal_NN_training/scripts/05_run_residual_demand_audit.sh
formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py
```

Removed / merged:

```text
02_run_prefetch_behavior_audit.sh     # removed; 06 is the unified behavior runner
17_parse_prefetch_behavior_audit.py   # removed; duplicate parser replaced by 01
```

## Current normal-prefetcher goal

Before changing the LSTM input features or labels, collect the same kind of information for every working normal prefetcher:

```text
counter behavior: IPC, speedup, L2 miss rate, miss reduction, accuracy, nodup accuracy, timeliness, late rate, duplicate proxy
residual behavior: demand miss rate, covered-on-time count, coverage among original misses, late rate, residual miss/share, duplicate event rate
joined planning table: one row per trace/prefetcher with behavior_* and residual_* fields
```

Default stable normal prefetchers:

```text
no_pref stride streamer ampm spp ipcp sms sandbox power7
```

These are the prefetchers that completed useful 25M/25M runs in the current Pythia fork. Other names from the Pythia multi-L2 prefetcher file can still be tested manually by overriding `PREFETCHERS`, but they are not default because they produced failed/no-final-stat logs in this setup.

## Full metric collection commands

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

Final output for LSTM planning:

```text
formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv
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
              f'cov={float(r.get("residual_coverage_among_misses") or 0):.4f} '
              f'residual_share={float(r.get("residual_residual_share_of_misses") or 0):.4f}')
PY
```

## Current interpretation from the first behavior zoo audit

The first 25M/25M counter-level sweep showed that SPP is not uniformly the best classical base prefetcher.

```text
602.gcc_s-734B:
  best observed base: sandbox
  sandbox speedup ≈ 1.1855, SPP speedup ≈ 1.1640
  Interpretation: SPP is strong, but sandbox/streamer/ampm/power7 are stronger classical baselines. A booster should be compared against the best base, not only against SPP.

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
Use the normal-prefetcher metrics table to choose trace-specific base prefetchers.
Do not hard-code the new LSTM notebook to SPP only.
However, keep 619 SPP+LSTM as the first replay candidate because the offline booster signal was strongest there.
```

## First LSTM residual-booster target

Old LSTM notebooks are kept as reference. New residual-booster notebooks should be added separately.

Recommended first notebook:

```text
formal_NN_training/LSTM/notebooks/LSTM_residual_booster_spp.ipynb
```

Next notebook change:

```text
BASE_PREFETCHER = "spp" / "sms" / "ampm" / "sandbox" / ...
```

First model scope:

```text
input:  recent demand stream + selected base prefetcher request/output context
output: residual useful prefetch / residual delta / timing bin
seq_len: 64 / 128 / 256, not 2048 by default
metrics: demand miss reduction, duplicate rate, late rate, nodup accuracy, IPC speedup
```

## Planned matrix

```text
normal prefetcher: trace-specific best base first; SPP retained for comparison
NN:                LSTM first, tiny Transformer later
size/#params:      small / medium / large
seq_len:           64 / 128 / 256
metrics:           accuracy, nodup accuracy, timeliness, demand coverage, residual share, IPC speedup
```
