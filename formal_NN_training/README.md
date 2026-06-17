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

Future model families should use the same pattern:

```text
formal_NN_training/<MODEL_NAME>/
  notebooks/

formal_NN_training/results/<MODEL_NAME>/draft/
formal_NN_training/results/<MODEL_NAME>/final/
```
