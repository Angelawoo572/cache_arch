# formal_NN_training

This directory is organized by neural-network family while shared simulator scripts stay in the top-level `scripts/` folder.

Current active flow:

```text
scripts/                  # common Pythia-based audit / run scripts
LSTM/                     # old/reference LSTM notebooks
results/LSTM/behavior_audit/
results/LSTM/draft/       # old LSTM artifacts/results kept only as draft history
```

Active scripts:

```text
formal_NN_training/scripts/17_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
```

## Step 1: SPP behavior audit

Run from the repo root on the cluster.

Important: after the 2026-06-17 config fix, rerun with `FORCE_REPLAY=1` once. The earlier logs used both `--config` and `--l2c_prefetcher_types`, which duplicated single prefetchers and made `spp_ipcp` appear as one unsupported string. The fixed script uses config files only.

```bash
cd ~/cache
git pull

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=3 \
NODUP=1 \
FORCE_REPLAY=1 \
bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
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

For debugging or quick smoke tests:

```bash
cd ~/cache

TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
WARMUP=1000000 \
SIM=1000000 \
MAX_JOBS=3 \
NODUP=1 \
FORCE_REPLAY=1 \
bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
```

Optional IPCP / combined audit:

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
bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
```

Sanity check that each log selected the intended prefetcher only once:

```bash
grep -R "adding L2C_PREFETCHER" formal_NN_training/results/LSTM/behavior_audit/logs/*.log
```

Expected patterns:

```text
no_pref: no IPCP/SPP line
spp:     one SPP_dev2 line
ipcp:    one IPCP line
spp_ipcp: one SPP_dev2 line + one IPCP line
```

If the IPCP run exits before writing a new summary, inspect the logs:

```bash
ls -lh formal_NN_training/results/LSTM/behavior_audit/logs/*ipcp*.log

grep -RniE "error|unsupported|assert|segmentation|abort|cannot|missing|adding L2C_PREFETCHER" \
  formal_NN_training/results/LSTM/behavior_audit/logs/*ipcp*.log | head -80

tail -80 formal_NN_training/results/LSTM/behavior_audit/logs/602.gcc_s-734B.ipcp.log
```

Main audit metrics:

```text
speedup_vs_no_pref       # IPC ratio vs no-prefetch
miss_reduction_vs_no_pref# L2 demand-load miss reduction vs no-prefetch
accuracy                 # pf_useful / pf_issued
nodup_accuracy           # pf_useful / (pf_issued - pq_merged_duplicate_proxy)
timeliness               # pf_useful / (pf_useful + pf_late)
```

Important: `coverage_vs_no_pref_l2_miss` is only a rough useful-prefetch proxy from ChampSim counters. Use `miss_reduction_vs_no_pref` plus IPC as the safer first-pass coverage signal. True residual labels need a later demand-centric table.

## Current interpretation from the first 3-trace SPP audit

The first audit was useful for debugging, but rerun after the config fix before treating the numbers as final. The qualitative direction is still the working hypothesis:

```text
602.gcc_s-734B:
  SPP likely helps, but duplicate/resource pressure needs checking with the fixed single-SPP run.
  First NN target: nodup/resource gating and maybe residual misses, not replacing SPP.

619.lbm_s-4268B:
  SPP likely helps, but timeliness and duplicate/resource pressure need checking with the fixed run.
  First NN target: timeliness + duplicate/resource gating.

605.mcf_s-994B:
  SPP likely gives little IPC gain.
  First NN target: residual demand misses. This is the best trace to test whether NN can catch what SPP misses.
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
