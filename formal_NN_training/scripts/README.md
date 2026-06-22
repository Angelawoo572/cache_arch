# `formal_NN_training/scripts`

This directory contains the active Pythia normal-prefetcher baseline, oracle-table, and standalone base-independent LSTM replay workflow.

```text
01_parse_prefetch_behavior_audit.py          # parse Pythia counter logs
03_patch_pythia_residual_logger.sh           # local demand-event logger patch
04_parse_residual_demand_audit.py            # parse demand-event CSVs
05_run_residual_demand_audit.sh              # collect residual event streams
06_run_base_prefetcher_zoo_audit.sh          # normal-prefetcher behavior sweep
07_join_normal_prefetcher_metrics.py         # join behavior and residual metrics
08_build_normal_prefetcher_oracle_table.py   # build LSTM oracle tables
09_run_oracle_replacer_replay_parallel.sh    # strict validated LSTM replay driver
10_prepare_oracle_replacer_replay_input.py   # rich list -> strict replay input
11_install_oracle_l2_replayer.sh             # build Pythia ListReplayer binary
12_parse_oracle_replacer_replay.py           # the only replay result parser
```

Normal prefetchers are baselines and teacher diagnostics. The standalone LSTM runtime inputs are raw no-prefetch demand-stream features, not normal-prefetcher predictions.

## Baseline and oracle-table pipeline

Run 06 for normal behavior logs, 05 for residual event logs, 07 to join the summaries, and 08 to build the `pc_line_occ` oracle tables. The notebook source directory is:

```text
formal_NN_training/results/base_prefetcher_zoo/oracle_event_table_pc_line_occ/
```

## Standalone LSTM replay

Keep every notebook export in an immutable artifact directory. The current lead-1 export is named:

```text
oracle_replacer_sweep_lead1_addrconf_lru2048.csv
prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv
```

Script 09 calls Script 10 to build strict ROI-L2-load index inputs, runs Pythia ListReplayer with PC/line signature validation, and then calls Script 12 to write the simulator summary. Do not run a second legacy summary script.

Detailed setup, preflight, build, replay, monitoring, and interpretation commands: [`ORACLE_REPLAYER.md`](ORACLE_REPLAYER.md).