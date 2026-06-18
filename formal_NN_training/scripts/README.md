# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
17_parse_prefetch_behavior_audit.py   # parse Pythia/ChampSim counter logs into behavior metrics
18_run_prefetch_behavior_audit.sh     # older numbered behavior-audit runner
run_prefetch_behavior_audit.sh        # current behavior-audit runner wrapper
19_parse_residual_demand_audit.py     # parse demand-centric residual event CSVs
run_residual_demand_audit.sh          # current demand-centric residual audit runner
```

Legacy scripts that depended on the old ChampSim `config.sh`, `spp_dev` patching, `champsim.l2_replayer`, or `PFETCH_LIST_PATH` replay flow were removed after switching `external/ChampSim` to the Pythia fork.

Recommended first residual audit:

```bash
cd ~/cache
git pull
TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
PREFETCHERS="no_pref spp ipcp spp_ipcp" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=4 \
FORCE_REPLAY=0 \
BUILD=0 \
COMPRESS=1 \
bash formal_NN_training/scripts/run_residual_demand_audit.sh
```

Output:

```text
formal_NN_training/results/LSTM/residual_audit/events/
formal_NN_training/results/LSTM/residual_audit/logs/
formal_NN_training/results/LSTM/residual_audit/summary.csv
```

Note: `run_residual_demand_audit.sh` can reuse an already patched Pythia binary with `BUILD=0`. A fresh rebuild requires the residual logger patch script to be present.
