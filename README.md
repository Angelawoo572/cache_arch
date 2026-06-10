# cache_arch

This repository is organized around cache-prefetch research tracks.

## Active track

```text
formal_NN_training/
```

Current active pipeline:

```text
SPP candidate logging
  -> outcome-aware LSTM cache-action training
  -> list_replayer replay
  -> SPP / LSTM / no-prefetch comparison
```

Main docs:

```text
formal_NN_training/README_LSTM_cache_action_predictor.md   # current results + commands
formal_NN_training/LSTM_cache_action_pipeline_story.md      # story / diagram explanation
formal_NN_training/scripts/README.md                        # script usage
```

Current result summary:

```text
LSTM wins issued-prefetch precision on 602.gcc and 619.lbm.
SPP still wins final IPC on both traces.
Coverage is trace-dependent, so useful count alone should not be treated as IPC.
```

## Paused / legacy tracks

```text
projects/
  legacy_gru_prefetch/      old GRU / NN replay / bypass experiments
  post_prefetch_filter/     PAUSED WIP: earlier SPP candidate utility-filter experiment
```

`projects/post_prefetch_filter/` is kept because it contains reusable SPP candidate-logging and candidate-table conversion pieces, but it is not the active training/replay pipeline anymore.

The old top-level folders were moved into `projects/legacy_gru_prefetch/`:

```text
scripts/           -> projects/legacy_gru_prefetch/scripts/
notebook/          -> projects/legacy_gru_prefetch/notebooks/
docs/              -> projects/legacy_gru_prefetch/docs/
configs/           -> projects/legacy_gru_prefetch/configs/
_cfg/              -> projects/legacy_gru_prefetch/_cfg/
champsim_modules/  -> projects/legacy_gru_prefetch/champsim_modules/
```
