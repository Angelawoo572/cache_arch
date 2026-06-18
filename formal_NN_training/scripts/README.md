# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
01_parse_prefetch_behavior_audit.py   # parse Pythia/ChampSim counter logs
04_parse_residual_demand_audit.py     # parse demand-centric residual event CSVs, fixed on-time coverage accounting
run_prefetch_behavior_audit.sh        # behavior-audit runner
run_residual_demand_audit.sh          # residual demand-audit runner
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

Regenerate an existing residual summary without rerunning ChampSim:

```bash
python3 formal_NN_training/scripts/04_parse_residual_demand_audit.py \
  --event-root formal_NN_training/results/LSTM/residual_audit/events \
  --out formal_NN_training/results/LSTM/residual_audit/summary.csv \
  --traces "602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
  --prefetchers "no_pref spp ipcp spp_ipcp" \
  --compressed
```

Output:

```text
formal_NN_training/results/LSTM/residual_audit/events/
formal_NN_training/results/LSTM/residual_audit/logs/
formal_NN_training/results/LSTM/residual_audit/summary.csv
```

Note: `run_residual_demand_audit.sh` can reuse an already patched Pythia binary with `BUILD=0`. A fresh rebuild still requires the local residual logger patch script to be present.
