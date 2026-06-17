# formal_NN_training

This directory is organized by neural-network family so multiple model ideas can coexist cleanly.

Current active family:

```text
LSTM/                    # LSTM + SPP cache-action predictor code/docs/notebooks
results/LSTM/draft/      # old LSTM results/artifacts kept only as draft history
```

Future model families should use the same pattern:

```text
formal_NN_training/<MODEL_NAME>/
  notebooks/

formal_NN_training/results/<MODEL_NAME>/draft/
formal_NN_training/results/<MODEL_NAME>/final/
```
