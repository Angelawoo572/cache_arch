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

The old `04_run_normal_prefetcher_sweep.sh` was deleted after its normal-run functionality was merged into script 11. For a counter-only baseline use:

```bash
MODE=normal COLLECT_EVENT_LOGS=0 \
  bash formal_NN_training/scripts/11_run_prefetch_event_attribution.sh
```

For causal event evidence use `MODE=both COLLECT_EVENT_LOGS=1` with the same driver.

## Specialized builders retained deliberately

```text
10_profile_champsim_trace.py               Dynamic trace profile; distinct output contract.
14_build_base_candidate_table.py           Base-proposal table; not used by standalone v4, retained for base-aware research.
16_build_trace_dependency_features.py      Generic dependency profile/edge vocabulary builder.
17_prepare_v3_9_605_dependency_sidecar.sh  Reproducible 605 packaging/validation wrapper around script 16.
19_build_oracle_ceiling_lists.py           Pure ceiling-list builder.
20_run_oracle_ceiling_replay.sh            Ceiling build + replay driver.
21_join_decision_ledger_attribution.py     Exact event audit-to-ledger join.
22_resource_summary.py                     PQ/MSHR/request-pressure summary.
25_build_v4_1_notebook.py                  Colab-only v4.1 materializer.
replay/verify_same_binary_no_pref.py       Same-binary no-pref IPC guard.
```

These are not duplicates. In particular, 19 creates a rich list while 20 invokes the simulator; 21 consumes event attribution plus one full decision ledger, while 22 consumes event logs only. Combining either pair would couple unrelated input contracts and hide failure modes.

Removed as duplicate or redundant:

```text
04_run_normal_prefetcher_sweep.sh
10_verify_same_binary_no_pref.py
16_verify_same_binary_no_pref.py
v4/run_oracle_ceiling_replay.sh
```
