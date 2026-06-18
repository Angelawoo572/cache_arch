# formal_NN_training

This directory is organized by model family, while shared Pythia/ChampSim scripts stay in the top-level `scripts/` folder.

Current active flow:

```text
scripts/                                      # common Pythia-based audit / run scripts
results/base_prefetcher_zoo/                  # all normal-prefetcher behavior + residual metrics
results/base_prefetcher_zoo/residual_audit/   # demand-centric coverage/residual data for working bases
LSTM/                                         # old/reference LSTM notebooks and new booster notebooks
results/LSTM/                                 # old/reference LSTM artifacts/results
```

Active scripts:

```text
formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/02_run_prefetch_behavior_audit.sh
formal_NN_training/scripts/03_patch_pythia_residual_logger.sh
formal_NN_training/scripts/04_parse_residual_demand_audit.py
formal_NN_training/scripts/05_run_residual_demand_audit.sh
formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py
```

Legacy scripts that depended on the previous ChampSim `config.sh`, `spp_dev` patching, `champsim.l2_replayer`, or `PFETCH_LIST_PATH` replay flow were removed after switching `external/ChampSim` to the Pythia fork.

## Goal now

Before changing the LSTM notebook again, collect the same information for every working normal prefetcher:

```text
behavior-side:
  IPC, speedup_vs_no_pref, miss_reduction_vs_no_pref,
  accuracy, nodup_accuracy, timeliness,
  late_per_issued, useless_per_issued, duplicate proxy, failure flag

residual-side:
  demand_miss_rate, covered_on_time, coverage_among_misses,
  late_rate_among_misses, residual_share_of_misses,
  pf_duplicate_rate, event file
```

Final joined table:

```text
formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_all_metrics.csv
```

This table should drive the next LSTM changes. The notebook should become base-aware instead of hard-coded to SPP.

## Step 1: counter-level base-prefetcher zoo audit

Run from the repo root on the cluster.

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=6 \
BUILD=0 \
FORCE_REPLAY=0 \
NODUP=1 \
bash formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
```

Default zoo:

```text
no_pref next_line stride streamer ampm bop spp ipcp sms bingo mlop sandbox scooby dspatch power7
```

Output:

```text
formal_NN_training/results/base_prefetcher_zoo/logs/
formal_NN_training/results/base_prefetcher_zoo/summary_nodup.csv
formal_NN_training/results/base_prefetcher_zoo/RUN_INFO.txt
```

View top prefetchers per trace:

```bash
python3 - <<'PY'
import csv
from collections import defaultdict
p="formal_NN_training/results/base_prefetcher_zoo/summary_nodup.csv"
rows=list(csv.DictReader(open(p)))
by=defaultdict(list)
for r in rows:
    if r.get("run_failed") == "1":
        continue
    by[r["trace"]].append(r)
for tr, rs in by.items():
    rs=sorted(rs, key=lambda r: float(r.get("speedup_vs_no_pref") or 0), reverse=True)
    print("\n==", tr, "==")
    for r in rs[:8]:
        print(f'{r["prefetcher"]:12s} speedup={float(r["speedup_vs_no_pref"]):.4f} '
              f'miss_red={float(r["miss_reduction_vs_no_pref"]):.4f} '
              f'nodup_acc={float(r["nodup_accuracy"]):.4f} '
              f'time={float(r["timeliness"]):.4f}')
PY
```

Important: some Pythia prefetchers may fail in the current build/config. `01_parse_prefetch_behavior_audit.py` now marks those rows with `run_failed=1`, so do not treat IPC=0 rows as real results.

## Step 2: demand-centric residual audit for working normal prefetchers

Run residual audit only for prefetchers that produced valid counter-level logs. Current working set from the first zoo run:

```text
no_pref stride streamer ampm spp ipcp sms sandbox power7
```

Run:

```bash
cd ~/cache
git pull

WORKING_PREFS="no_pref stride streamer ampm spp ipcp sms sandbox power7"

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
PREFETCHERS="$WORKING_PREFS" \
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
formal_NN_training/results/base_prefetcher_zoo/residual_audit/events/*.events.csv.gz
formal_NN_training/results/base_prefetcher_zoo/residual_audit/logs/*.log
formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv
formal_NN_training/results/base_prefetcher_zoo/residual_audit/RUN_INFO.txt
```

Main residual-audit metrics:

```text
demand_miss_rate          # direct L2 demand-load miss rate under this prefetcher
covered_on_time           # original miss-pool accesses converted to prefetched hits
coverage_among_misses     # covered_on_time / original_miss_pool
late_rate_among_misses    # demand miss merged with in-flight prefetch
pf_duplicate_rate         # duplicate/merged prefetch-request proxy
residual_share_of_misses  # current residual miss pool / original miss pool
```

## Step 3: join all normal-prefetcher information

```bash
python3 formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py \
  --behavior formal_NN_training/results/base_prefetcher_zoo/summary_nodup.csv \
  --residual formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv \
  --out formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_all_metrics.csv
```

View final table:

```bash
column -t -s, formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_all_metrics.csv | less -S
```

## Current interpretation from the first zoo run

```text
602.gcc_s-734B:
  best observed base: sandbox
  also strong: streamer, power7, ampm, spp
  interpretation: SPP is not the strongest base here; protect strong classical baselines.

619.lbm_s-4268B:
  best observed base: sms
  SPP remains close and has clear timing/duplicate weakness.
  interpretation: still useful for LSTM+SPP timing/gating story, but SMS is the stronger classical baseline.

605.mcf_s-994B:
  best observed bases: ampm / spp by IPC, sms by miss reduction
  interpretation: SPP is weak by coverage; try AMPM/SMS residual data before changing model architecture.

620.omnetpp_s-874B:
  best observed base: sms
  also useful: sandbox, power7, streamer
  interpretation: SPP is a poor base; switch base before blaming LSTM.

623.xalancbmk_s-700B:
  best observed base: spp, but gain is tiny
  many aggressive prefetchers hurt IPC
  interpretation: combination-sensitive / pollution-sensitive; use conservative gating or no-prefetch detector.
```

High-level conclusion:

```text
SPP-only story is no longer enough.
The next notebook should be base-aware:
  BASE_PREFETCHER in {spp, sms, ampm, sandbox, streamer, power7, stride}

First keep 619 SPP+LSTM as the demonstrable neural signal.
For 605/620, first switch the base prefetcher, then train the residual booster.
For 623, be conservative; aggressive prefetchers mostly hurt.
```

## Planned matrix

```text
normal prefetcher: trace-selected base first; SPP kept as comparison
NN:                LSTM first, tiny Transformer later
size/#params:      small / medium / large
seq_len:           64 / 128 / 256, not 2048 by default
metrics:           accuracy, nodup accuracy, timeliness, coverage/residual share, miss reduction, IPC speedup
```

Old LSTM notebooks are kept as reference. New residual-booster notebooks should be added separately instead of overwriting the old ones.
