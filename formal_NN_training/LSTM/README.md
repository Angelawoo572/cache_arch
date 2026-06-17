# LSTM cache-action predictor

This folder contains the LSTM/SPP cache-action experiment family.

Layout:

```text
formal_NN_training/LSTM/
  notebooks/   # Colab training notebooks
  scripts/     # trace dump, pack, replay, parse, figure scripts
  docs/        # explanation / story notes

formal_NN_training/results/LSTM/draft/
  artifacts/        # old Colab/model/action outputs
  replay_compare/   # old replay summaries/log-derived CSVs
  capacity_sweep/   # old capacity-sweep outputs
  final_tables/     # old final comparison tables
```

The old result files are intentionally kept as `draft` outputs because the next LSTM run should regenerate fresh artifacts/results.
