# `formal_NN_training/scripts`

Standalone, base-independent NN workflow. Normal prefetchers are comparison baselines only.

## Active scripts

```text
01_parse_prefetch_behavior_audit.py       Parse Pythia counter logs.
02_patch_pythia_demand_logger.sh          Patch Pythia to log raw no-prefetch L2 demand events.
03_collect_no_pref_demand_events.sh       Run no-prefetch traces and collect the raw stream.
04_run_normal_prefetcher_sweep.sh         Run the canonical normal-prefetcher baseline sweep.
05_build_standalone_oracle_dataset.py     Convert no-prefetch events into the NN dataset.
06_install_keyed_listreplayer.sh          Verify and build the keyed ListReplayer.
07_prepare_keyed_replay_input.py          Convert a frozen NN export into replay keys.
08_run_standalone_lstm_replay.sh          Run a same-binary no-pref control and keyed NN replay.
09_parse_standalone_lstm_replay.py        Write current-normal and same-binary comparisons.
```

The notebook input is:

```text
formal_NN_training/results/standalone_nn_data/oracle/<trace>.oracle.csv.gz
```

Each row contains:

```text
trace, demand_idx, cycle, pc, addr, line, page, page_offset, delta,
no_pref_hit, no_pref_miss, pc_line_occ
```

The notebook derives future-miss labels only from `no_pref_miss`; it does not read normal-prefetcher coverage or teacher fields.

Execution order: 04 baseline sweep, 03 raw collection, 05 dataset build, notebook training, 06 keyed replayer build, then 08 replay.