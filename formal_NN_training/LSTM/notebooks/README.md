# Notebook status

## Current

`v4_1_run.ipynb` is the current notebook launcher. It materializes and runs the
v4.1 standalone analysis notebook through the active `formal_NN_training/scripts`
pipeline.

## Legacy — do not run as current experiments

```text
LSTM_cache_action_predictor.ipynb
LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb
```

These notebooks belong to the retired SPP/action-predictor workflow. Their old
script references are intentionally historical; they must not be repaired or
interpreted as callers of the current standalone no-prefetch design.
