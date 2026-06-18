# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
01_parse_prefetch_behavior_audit.py   # parse Pythia/ChampSim counter logs; flags failed runs
02_run_prefetch_behavior_audit.sh     # behavior-audit runner for selected base prefetchers
03_patch_pythia_residual_logger.sh    # patch local Pythia for demand-centric event logging
04_parse_residual_demand_audit.py     # parse residual event CSVs, fixed on-time coverage accounting
05_run_residual_demand_audit.sh       # residual demand-audit runner; supports arbitrary base prefetchers
06_run_base_prefetcher_zoo_audit.sh   # broad sweep over available Pythia L2 prefetchers
07_join_normal_prefetcher_metrics.py  # join behavior + residual summaries into one NN-planning table
```

Legacy scripts that depended on the old ChampSim `config.sh`, `spp_dev` patching, `champsim.l2_replayer`, or `PFETCH_LIST_PATH` replay flow were removed after switching `external/ChampSim` to the Pythia fork.

## 1. Counter-level base-prefetcher zoo audit

Run this first. It collects IPC/speedup, miss reduction, accuracy, nodup accuracy, timeliness, late rate, and duplicate proxy for each normal prefetcher.

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

After the run, failed/unsupported prefetchers are flagged by `run_failed=1` in `summary_nodup.csv`. Treat IPC=0 rows as failed unless the log proves otherwise.

## 2. Demand-centric residual audit for working normal prefetchers

Run this after the zoo audit. It collects demand coverage and residual-pool metrics for the working base prefetchers. The default working set excludes prefetchers that failed in the current Pythia build.

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
formal_NN_training/results/base_prefetcher_zoo/residual_audit/events/
formal_NN_training/results/base_prefetcher_zoo/residual_audit/logs/
formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv
formal_NN_training/results/base_prefetcher_zoo/residual_audit/RUN_INFO.txt
```

Regenerate an existing residual summary without rerunning ChampSim:

```bash
python3 formal_NN_training/scripts/04_parse_residual_demand_audit.py \
  --event-root formal_NN_training/results/base_prefetcher_zoo/residual_audit/events \
  --out formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv \
  --traces "602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
  --prefetchers "$WORKING_PREFS" \
  --compressed
```

## 3. Join behavior + residual metrics for NN planning

```bash
python3 formal_NN_training/scripts/07_join_normal_prefetcher_metrics.py \
  --behavior formal_NN_training/results/base_prefetcher_zoo/summary_nodup.csv \
  --residual formal_NN_training/results/base_prefetcher_zoo/residual_audit/summary.csv \
  --out formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_all_metrics.csv
```

Final table:

```text
formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_all_metrics.csv
```

This is the table to use before changing the LSTM notebook. It contains behavior-side accuracy/timeliness/speedup and residual-side coverage/residual-pool metrics in one place.

Note: `05_run_residual_demand_audit.sh` can reuse an already patched Pythia binary with `BUILD=0`. A fresh rebuild uses `03_patch_pythia_residual_logger.sh`.
