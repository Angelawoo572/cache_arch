# LSTM Cache Action Predictor

This file documents the LSTM notebook added under `formal_NN_training/` without replacing the existing SPP schema README.

## Main file

- `LSTM_cache_action_predictor.ipynb`

## Purpose

The LSTM is **not only an SPP filter**. It predicts cache/memory actions directly from access-pattern history, time/context, hit/miss behavior, and optional SPP/cache-state signals.

The intended outputs are:

- next useful delta class
- future hit/miss tendency
- cache bypass or low-priority insertion decision
- timing / reuse-distance bucket

## Expected input

Put ChampSim-derived event CSV/Parquet files under:

```text
formal_NN_training/data/
formal_NN_training/data/generated/
```

Minimum columns:

```text
pc, addr
```

Recommended columns:

```text
trace, event_id, cycle, pc, addr, hit, is_store,
spp_delta, spp_conf,
mshr_occupancy, l2_occupancy, bandwidth_pressure,
semantic_class
```

## Main output

After training, the notebook writes artifacts under:

```text
formal_NN_training/artifacts/
```

Expected important files:

```text
lstm_cache_action_predictor.pt
lstm_training_history.csv
lstm_cache_action_table.csv
```

The action table is the file to connect next to the ChampSim replay / `.sh` flow.

## Next step

The next engineering step is to generate real ChampSim event data, train this notebook, then write a replay script that consumes the predicted action table and compares against:

```text
no prefetch
SPP
previous GRU flow
LSTM cache-action predictor
future Transformer / TCN baselines
```

Final claims should use IPC / MPKI / accuracy / coverage / timeliness / pollution / bandwidth pressure, not accuracy alone.
