# `formal_NN_training/scripts`

All Sacramento-side Python utilities are Python 3.6 compatible and use only the standard library.

## Active pipeline

```text
01_parse_prefetch_behavior_audit.py       Parse normal simulator runs into summary.csv.
02_patch_pythia_demand_logger.sh          Enable per-event L2C logging when needed.
03_collect_no_pref_demand_events.sh       Collect raw no-prefetch demand events.
05_build_standalone_oracle_dataset.py     Build standalone oracle data.
06_install_keyed_listreplayer.sh          Build keyed ListReplayer.
07_prepare_keyed_replay_input.py          Convert rich lists to keyed replay input.
08_run_standalone_lstm_replay.sh          Guarded standalone replay.
09_parse_standalone_lstm_replay.py        Parse replay results.
11_run_prefetch_event_attribution.sh      Unified normal baseline, standalone event, and evidence runner.
12_analyze_prefetch_event_attribution.py  Classify normal/standalone event outcomes.
13_build_cache_capacity_variant.sh        Build one capacity-specific binary.
14_run_cache_capacity_sweep.sh            Frozen-list capacity sensitivity control.
15_summarize_prefetch_evidence.py         Merge evidence into a report.
```

`04_run_normal_prefetcher_sweep.sh` was merged into script 11. Use
`MODE=normal COLLECT_EVENT_LOGS=0` with script 11 for counter-only normal runs.

## Optional research builders

```text
10_profile_champsim_trace.py               Dynamic trace profile used by evidence reports.
14_build_base_candidate_table.py           Optional normal-proposal table for base-aware research.
16_build_trace_dependency_features.py      Dependency profile and edge vocabulary builder.
17_prepare_v3_9_605_dependency_sidecar.sh  605 wrapper around script 16.
19_build_oracle_ceiling_lists.py           Ceiling-list builder.
20_run_oracle_ceiling_replay.sh            Ceiling replay driver.
21_join_decision_ledger_attribution.py     Event-attribution to full-ledger join.
22_resource_summary.py                     PQ/MSHR/request-pressure summary.
25_build_v4_1_notebook.py                  Colab-only v4.1 materializer.
replay/resolve_replay_plan.py              Shared replay-plan resolver.
replay/verify_same_binary_no_pref.py       Same-binary no-pref IPC guard.
```

Removed duplicate or pure-wrapper scripts:

```text
04_run_normal_prefetcher_sweep.sh
10_verify_same_binary_no_pref.py
16_verify_same_binary_no_pref.py
v4/run_oracle_ceiling_replay.sh
```
