# cache_arch

This repository is organized around project tracks. The current state is:

```text
projects/
  legacy_gru_prefetch/      old GRU / NN replay / bypass experiments
  post_prefetch_filter/     PAUSED WIP: SPP candidate utility-filter experiment
```

The old top-level folders were moved into `projects/legacy_gru_prefetch/`:

```text
scripts/           -> projects/legacy_gru_prefetch/scripts/
notebook/          -> projects/legacy_gru_prefetch/notebooks/
docs/              -> projects/legacy_gru_prefetch/docs/
configs/           -> projects/legacy_gru_prefetch/configs/
_cfg/              -> projects/legacy_gru_prefetch/_cfg/
champsim_modules/  -> projects/legacy_gru_prefetch/champsim_modules/
```

`projects/post_prefetch_filter/` is kept as a paused research snapshot. It contains useful SPP candidate-logging, candidate-table conversion, and 602/605 offline-analysis artifacts, but it is not the active training/replay pipeline anymore. A future clean restart should use a new project directory instead of continuing to pile experiments into this paused folder.
