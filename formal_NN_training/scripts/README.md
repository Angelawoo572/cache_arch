# `formal_NN_training/scripts`

This directory contains one **standalone, base-independent NN pipeline**.

The NN is not residual. It does not receive SPP/AMPM/SMS output as input, it is not trained only on normal-prefetcher misses, and it does not issue a base-plus-NN union policy. Normal prefetchers are comparison baselines only.

## Active scripts

```text
01_parse_prefetch_behavior_audit.py       Parse Pythia counter logs.
02_patch_pythia_demand_logger.sh          Patch Pythia to log raw no-prefetch L2 demand events.
03_collect_no_pref_demand_events.sh       Run no-prefetch traces and collect the raw stream.
04_run_normal_prefetcher_sweep.sh         Run normal-prefetcher baseline comparisons.
05_build_standalone_oracle_dataset.py     Convert no-prefetch events into the NN dataset.
06_install_keyed_listreplayer.sh          Build the PC-line-occurrence keyed ListReplayer.
07_prepare_keyed_replay_input.py          Convert a frozen NN export into replay keys.
08_run_standalone_lstm_replay.sh          Replay a frozen standalone LSTM policy.
09_parse_standalone_lstm_replay.py        Write the replay summary.
```

The dataset required before writing the notebook is:

```text
formal_NN_training/results/standalone_nn_data/oracle/<trace>.oracle.csv.gz
```

Each row contains only no-prefetch demand-stream fields:

```text
trace, demand_idx, cycle, pc, addr, line, page, page_offset, delta,
no_pref_hit, no_pref_miss, pc_line_occ
```

The notebook derives multi-horizon future-miss labels from `no_pref_miss`. It must not read normal-prefetcher coverage, teacher, proposal, union, or residual fields.

The separate normal-baseline table is:

```text
formal_NN_training/results/prefetcher_baselines/summary.csv
```

It is used only for final comparison against no-prefetch and the best normal prefetcher.

Execution order: `04` baseline sweep, `03` no-prefetch event collection, `05` standalone dataset build, then notebook training; after export run `06`, then `08`.
