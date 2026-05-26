# Legacy GRU Prefetch Experiments

This folder contains the previous GRU / NN replay / bypass experiments.

Important moved paths:

```text
scripts/          -> projects/legacy_gru_prefetch/scripts/
notebook/         -> projects/legacy_gru_prefetch/notebooks/
docs/             -> projects/legacy_gru_prefetch/docs/
configs/          -> projects/legacy_gru_prefetch/configs/
_cfg/             -> projects/legacy_gru_prefetch/_cfg/
champsim_modules/ -> projects/legacy_gru_prefetch/champsim_modules/
```

Example old command:

```bash
TRACE=602.gcc_s-734B bash scripts/run_gru_v9_decode_sweep.sh
```

New command:

```bash
TRACE=602.gcc_s-734B bash projects/legacy_gru_prefetch/scripts/run_gru_v9_decode_sweep.sh
```
