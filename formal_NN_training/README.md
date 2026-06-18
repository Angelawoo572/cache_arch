# formal_NN_training

This directory is organized by neural-network family while shared simulator scripts stay in the top-level `scripts/` folder.

Current active flow:

```text
scripts/                  # common Pythia-based audit / run scripts
LSTM/                     # old/reference LSTM notebooks
results/LSTM/behavior_audit/
results/LSTM/residual_audit/
results/LSTM/draft/       # old LSTM artifacts/results kept only as draft history
```

Active audit scripts:

```text
formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/02_run_prefetch_behavior_audit.sh
formal_NN_training/scripts/03_patch_pythia_residual_logger.sh
formal_NN_training/scripts/04_parse_residual_demand_audit.py
formal_NN_training/scripts/05_run_residual_demand_audit.sh
```

Note: old scripts that depended on the previous ChampSim `config.sh`, `spp_dev` patching, `champsim.l2_replayer`, or `PFETCH_LIST_PATH` replay flow are no longer the current Pythia workflow.

## Step 1: prefetcher behavior audit

Run from the repo root on the cluster.

Important: use `FORCE_REPLAY=1` when changing configs/prefetcher sets, otherwise old logs will be reused.

### SPP-only first pass

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=3 \
NODUP=1 \
FORCE_REPLAY=1 \
bash formal_NN_training/scripts/02_run_prefetch_behavior_audit.sh
```

### SPP + IPCP + combined audit

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
PREFETCHERS="no_pref spp ipcp spp_ipcp" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=3 \
NODUP=1 \
BUILD=0 \
FORCE_REPLAY=1 \
bash formal_NN_training/scripts/02_run_prefetch_behavior_audit.sh
```

Output:

```text
formal_NN_training/results/LSTM/behavior_audit/logs/
formal_NN_training/results/LSTM/behavior_audit/summary_nodup.csv
```

View the summary:

```bash
column -t -s, formal_NN_training/results/LSTM/behavior_audit/summary_nodup.csv
```

Sanity check that each log selected the intended prefetcher only once:

```bash
grep -R "adding L2C_PREFETCHER" formal_NN_training/results/LSTM/behavior_audit/logs/*.log
```

Expected patterns:

```text
no_pref:  no IPCP/SPP line
spp:      one SPP_dev2 line
ipcp:     one IPCP line
spp_ipcp: one SPP_dev2 line + one IPCP line
```

Main audit metrics:

```text
speedup_vs_no_pref        # IPC ratio vs no-prefetch
miss_reduction_vs_no_pref # L2 demand-load miss reduction vs no-prefetch
accuracy                  # pf_useful / pf_issued
nodup_accuracy            # pf_useful / (pf_issued - pq_merged_duplicate_proxy)
timeliness                # pf_useful / (pf_useful + pf_late)
```

Important: `coverage_vs_no_pref_l2_miss` in the counter-level audit is only a rough useful-prefetch proxy. The demand-centric residual audit below is the better source for residual labels.

## Step 2: demand-centric residual audit

The counter audit is not enough to build final NN labels. Step 2 logs one row per L2C demand LOAD access plus every L2C prefetch request, then summarizes where each base prefetcher failed.

### Full 5-trace SPP/IPCP/combined residual audit

```bash
cd ~/cache
git pull

T620=$(basename "$(ls traces/620*.champsimtrace.xz | head -1)" .champsimtrace.xz)
T623=$(basename "$(ls traces/623*.champsimtrace.xz | head -1)" .champsimtrace.xz)

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B $T620 $T623" \
PREFETCHERS="no_pref spp ipcp spp_ipcp" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=4 \
FORCE_REPLAY=0 \
BUILD=0 \
COMPRESS=1 \
bash formal_NN_training/scripts/05_run_residual_demand_audit.sh
```

If the event files already exist and only the summary needs to be regenerated:

```bash
python3 formal_NN_training/scripts/04_parse_residual_demand_audit.py \
  --event-root formal_NN_training/results/LSTM/residual_audit/events \
  --out formal_NN_training/results/LSTM/residual_audit/summary.csv \
  --traces "602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B $T620 $T623" \
  --prefetchers "no_pref spp ipcp spp_ipcp" \
  --compressed
```

Output:

```text
formal_NN_training/results/LSTM/residual_audit/events/*.events.csv.gz  # large, ignored
formal_NN_training/results/LSTM/residual_audit/logs/*.log              # ignored
formal_NN_training/results/LSTM/residual_audit/summary.csv             # small, tracked
formal_NN_training/results/LSTM/residual_audit/RUN_INFO.txt            # small, tracked
```

View the summary:

```bash
column -t -s, formal_NN_training/results/LSTM/residual_audit/summary.csv
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

Parser note: use `04_parse_residual_demand_audit.py`. The coverage attribution is now nonzero and consistent with the miss-rate reductions for SPP-strong traces. Small differences between `original_miss_pool` and the exact no-prefetch miss count are expected because each prefetcher run can slightly perturb timing/path behavior.

## Current interpretation from 5-trace residual audit

Latest 25M/25M residual audit with `PREFETCHERS="no_pref spp ipcp spp_ipcp"`:

```text
602.gcc_s-734B:
  no_pref miss_rate      ≈ 0.5025
  SPP miss_rate          ≈ 0.1753
  SPP coverage           ≈ 65.1% of original miss pool
  SPP residual share     ≈ 34.9%
  SPP late_rate          ≈ 0.54%
  SPP duplicate          ≈ 0.08%
  Interpretation: SPP is very strong on this trace. NN should be conservative: avoid hurting SPP, maybe learn low-risk residual/gating only.

619.lbm_s-4268B:
  no_pref miss_rate      ≈ 0.9987
  SPP miss_rate          ≈ 0.7655
  SPP coverage           ≈ 23.3%
  SPP residual share     ≈ 76.7%
  SPP late_rate          ≈ 28.0%
  SPP duplicate          ≈ 33.2%
  Interpretation: SPP helps but has serious timeliness and duplicate problems. This is the best trace for timing-aware NN/gating.

605.mcf_s-994B:
  no_pref miss_rate      ≈ 0.7513
  SPP miss_rate          ≈ 0.7464
  SPP coverage           ≈ 0.7%
  SPP residual share     ≈ 99.3%
  SPP late_rate          ≈ 0.15%
  SPP duplicate          ≈ 0.95%
  Interpretation: SPP barely helps. This is a residual/blind-spot trace, but it may also be intrinsically hard/pointer-chase-like.

620.omnetpp_s-874B:
  no_pref miss_rate      ≈ 0.7013
  SPP miss_rate          ≈ 0.7008
  IPCP miss_rate         ≈ 0.6981
  SPP+IPCP miss_rate     ≈ 0.6978
  SPP coverage           ≈ 0.02%
  IPCP coverage          ≈ 0.68%
  SPP+IPCP coverage      ≈ 0.69%
  Interpretation: SPP is almost useless; IPCP is slightly better but still weak. This is another residual/blind-spot trace.

623.xalancbmk_s-700B:
  no_pref miss_rate      ≈ 0.3696
  SPP miss_rate          ≈ 0.3608
  IPCP miss_rate         ≈ 0.3785
  SPP+IPCP miss_rate     ≈ 0.3697
  SPP coverage           ≈ 0.26%
  IPCP coverage          ≈ 0.62% but worse miss rate
  SPP+IPCP coverage      ≈ 0.92% but no net benefit over no_pref
  Interpretation: SPP gives a small improvement; IPCP hurts; SPP+IPCP loses SPP's benefit. Treat as medium/weak SPP case and avoid naive prefetcher combining.
```

High-level conclusion:

```text
SPP-strong / protect-SPP trace:
  602

SPP-timing/duplicate problem trace:
  619

SPP-weak / residual-blind-spot traces:
  605, 620

SPP-small-gain / combination-sensitive trace:
  623

IPCP status:
  IPCP is not a strong baseline in the current configuration. Keep it in the matrix as a comparison/ablation, but the first NN notebook should focus on LSTM + SPP.
```

## First LSTM residual-booster target

Do not overwrite the old LSTM notebooks. Keep them as reference and create a new clean notebook for the residual-booster flow.

Recommended first notebook:

```text
formal_NN_training/LSTM/notebooks/LSTM_residual_booster_spp.ipynb
```

First label concept:

```text
base = SPP
residual_label = demand miss under SPP run
```

Better label when using the no-prefetch original miss pool:

```text
residual_label = original no-prefetch miss that SPP did not cover in time
```

First model scope:

```text
input:  recent demand stream + SPP request/output context
output: residual useful prefetch / residual delta / timing bin
seq_len: 64 / 128 / 256, not 2048 by default
metrics: demand miss reduction, duplicate rate, late rate, nodup accuracy, IPC speedup
```

## Planned matrix, after the audit is stable

```text
normal prefetcher: SPP first, IPCP later
NN:                LSTM first, tiny Transformer later
size/#params:      small / medium / large
seq_len:           64 / 128 / 256, not 2048 by default
metrics:           accuracy, nodup accuracy, timeliness, coverage/miss reduction, IPC speedup
```

Old LSTM notebooks are kept as reference. New residual-booster notebooks should be added separately instead of overwriting the old ones.

Future model families should use the same pattern:

```text
formal_NN_training/<MODEL_NAME>/
  notebooks/

formal_NN_training/results/<MODEL_NAME>/draft/
formal_NN_training/results/<MODEL_NAME>/final/
```
