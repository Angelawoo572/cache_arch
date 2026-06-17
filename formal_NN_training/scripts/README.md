# formal_NN_training/scripts

Active scripts here are for the current Pythia-based workflow.

```text
17_parse_prefetch_behavior_audit.py   # parse Pythia/ChampSim logs into behavior metrics
18_run_prefetch_behavior_audit.sh     # build/run Pythia audit for no_pref, SPP, IPCP, SPP+IPCP
```

Legacy scripts that depended on the old ChampSim `config.sh`, `spp_dev` patching, `champsim.l2_replayer`, or `PFETCH_LIST_PATH` replay flow were removed after switching `external/ChampSim` to the Pythia fork.

Default first audit:

```bash
cd ~/cache
TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
WARMUP=25000000 \
SIM=25000000 \
MAX_JOBS=3 \
NODUP=1 \
bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
```

Output:

```text
formal_NN_training/results/LSTM/behavior_audit/logs/
formal_NN_training/results/LSTM/behavior_audit/summary_nodup.csv
```
