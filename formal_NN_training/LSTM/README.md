# LSTM cache-action predictor

This folder keeps the older LSTM/SPP cache-action notebooks as reference material while the active simulator flow has moved to the Pythia fork.

Current layout:

```text
formal_NN_training/LSTM/
  notebooks/              # old/reference LSTM Colab notebooks

formal_NN_training/scripts/
  17_parse_prefetch_behavior_audit.py
  18_run_prefetch_behavior_audit.sh

formal_NN_training/results/LSTM/
  behavior_audit/         # new Pythia SPP/IPCP behavior-audit outputs
  draft/                  # old LSTM artifacts/results kept as draft history
```

Legacy old-ChampSim scripts were removed because the repo now uses Pythia under `external/ChampSim`.
