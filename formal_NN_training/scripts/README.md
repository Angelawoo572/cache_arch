# `formal_NN_training/scripts`

This directory contains the active Pythia-based normal-prefetcher baseline, oracle-table, and standalone LSTM replay workflow.

```text
01_parse_prefetch_behavior_audit.py          # parse Pythia counter logs
03_patch_pythia_residual_logger.sh           # local demand-event logger patch
04_parse_residual_demand_audit.py            # parse demand-event CSVs
05_run_residual_demand_audit.sh              # collect residual event streams
06_run_base_prefetcher_zoo_audit.sh          # normal-prefetcher behavior sweep
07_join_normal_prefetcher_metrics.py         # join behavior and residual metrics
08_build_normal_prefetcher_oracle_table.py   # build LSTM oracle tables
09_run_oracle_replacer_replay_parallel.sh    # validated LSTM replay and summary
10_prepare_oracle_replacer_replay_input.py   # rich list to strict replay input
11_install_oracle_l2_replayer.sh             # build Pythia ListReplayer binary
12_parse_oracle_replacer_replay.py           # summarize replay, baseline, offline metrics
```

Normal prefetchers are baselines and teacher diagnostics. The standalone LSTM uses raw no-prefetch demand-stream features at runtime.

## Baseline and oracle-table pipeline

Run 06 for normal behavior logs, 05 for residual event logs, 07 to join the summaries, and 08 to build the `pc_line_occ` oracle tables. The current LSTM source directory is:

```text
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table_pc_line_occ/
```

## Standalone LSTM replay

Put one immutable notebook export into its own artifact directory, for example:

```text
formal_NN_training/artifacts/oracle_replacer/thr010/
```

That directory contains `prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv` plus `oracle_replacer_sweep.csv`. Script 09 converts the notebook rich CSV into the strict index list, signature-validates every runtime callback, runs Pythia, and writes a simulator `summary.csv`.

Detailed setup and commands: [`ORACLE_REPLAYER.md`](ORACLE_REPLAYER.md).
