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
formal_NN_training/scripts/17_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/run_prefetch_behavior_audit.sh
formal_NN_training/scripts/19_parse_residual_demand_audit.py
formal_NN_training/scripts/20_patch_pythia_residual_logger.sh
formal_NN_training/scripts/run_residual_demand_audit.sh
```

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
bash formal_NN_training/scripts/run_prefetch_behavior_audit.sh
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
bash formal_NN_training/scripts/run_prefetch_behavior_audit.sh
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

Important: `coverage_vs_no_pref_l2_miss` is only a rough useful-prefetch proxy from ChampSim counters and can be misleading. Use `miss_reduction_vs_no_pref` plus IPC as the safer first-pass coverage signal. True residual labels need the Step 2 demand-centric table.

## Current interpretation from the 3-trace audit

Latest FORCE_REPLAY audit with `PREFETCHERS="no_pref spp ipcp spp_ipcp"`, 25M/25M, `NODUP=1`:

```text
602.gcc_s-734B:
  SPP speedup ≈ 1.16x, timeliness ≈ 0.998, nodup accuracy ≈ 0.059.
  IPCP alone ≈ no-prefetch; SPP+IPCP ≈ SPP.
  Interpretation: SPP is doing useful work, but precision is low. First NN target should be nodup/resource gating and maybe residual misses, not replacing SPP.

619.lbm_s-4268B:
  SPP speedup ≈ 1.18x, timeliness ≈ 0.745, nodup accuracy ≈ 0.082.
  IPCP alone ≈ no-prefetch; SPP+IPCP ≈ SPP.
  Interpretation: SPP helps, but timeliness is the obvious weakness. First NN target should be timing-aware gating / residual timing, not aggressive extra prefetching.

605.mcf_s-994B:
  SPP speedup ≈ 1.03x, low useful-prefetch proxy coverage, nodup accuracy ≈ 0.020.
  IPCP alone ≈ no-prefetch; SPP+IPCP ≈ SPP.
  Interpretation: SPP barely helps. This is the best first trace for residual NN: learn demand misses SPP did not cover.
```

## Step 2: demand-centric residual audit

The counter audit above is not enough to build final NN labels. Step 2 patches local Pythia to log one row per L2C demand LOAD access plus every L2C prefetch request, then summarizes where each base prefetcher failed.

### Full SPP/IPCP/combined residual audit

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
PREFETCHERS="no_pref spp ipcp spp_ipcp" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=3 \
FORCE_REPLAY=1 \
RESET_PATCH=1 \
bash formal_NN_training/scripts/run_residual_demand_audit.sh
```

Smoke test before the full run:

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
PREFETCHERS="no_pref spp ipcp spp_ipcp" \
WARMUP=1000000 \
SIM=1000000 \
MAX_JOBS=3 \
FORCE_REPLAY=1 \
RESET_PATCH=1 \
bash formal_NN_training/scripts/run_residual_demand_audit.sh
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
covered_on_time_rate       # demand hit on a prefetched cache line
late_rate_among_misses     # demand miss merged with in-flight prefetch
residual_share_of_misses   # demand miss not covered and not late; residual NN target pool
pf_duplicate_rate          # prefetch requests merged/duplicated in PQ
```

The first LSTM residual label should be:

```text
residual_label = demand miss AND not covered in time by SPP
```

In the residual audit summary, this is approximated by `residual_miss` for the SPP run.

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
