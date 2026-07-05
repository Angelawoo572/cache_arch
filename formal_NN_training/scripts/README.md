# `formal_NN_training/scripts`

All Sacramento-side utilities are Python 3.6 compatible and use only the
standard library unless a script explicitly invokes an existing ChampSim build
tool. Numeric names are retained for path stability.

## Current standalone raw-stream pipeline

```text
03 collect no-prefetch L2C demand events
  -> 05 build stable oracle (pc,line,pc_line_occ)
  -> Colab tiny/other standalone notebook writes:
       rich prefetch list + full decision ledger + replay_plan.csv
  -> 06 build keyed ListReplayer
  -> 07 verify/convert rich list to keyed replay input
  -> 08 guarded replay + same-binary no-prefetch validation
  -> 11 full normal matrix and/or event-logging standalone replay
  -> 09 summarize plan-mode replay logs and winners
  -> 12 demand-event overlap attribution
  -> 22 PQ/MSHR resource summary
  -> 15 consolidated evidence and cache-miss comparison
```

The standalone training source is always the no-prefetch oracle. Normal
prefetcher outputs are replay baselines and evidence only; they do not create
NN labels.

## Active general scripts

| Script | Responsibility |
|---|---|
| `01_parse_prefetch_behavior_audit.py` | Parse counter logs into IPC, L2 misses, issue/useful/useless/late, accuracy, coverage, and timeliness. |
| `02_patch_pythia_demand_logger.sh` | Idempotently add instrumentation-only L2C demand/prefetch logging. |
| `03_collect_no_pref_demand_events.sh` | Safely collect canonical no-prefetch demand streams. |
| `05_build_standalone_oracle_dataset.py` | Convert a raw demand stream into the stable standalone oracle. |
| `06_install_keyed_listreplayer.sh` | Build the PC-line-occurrence keyed ListReplayer. |
| `07_prepare_keyed_replay_input.py` | Strictly map rich exports to oracle keys; direct indices require matching PC and line. |
| `08_run_standalone_lstm_replay.sh` | Guarded keyed replay with optional same-binary no-prefetch verification. |
| `09_parse_standalone_lstm_replay.py` | Produce plan-mode candidate/winner summaries from either the standard replay layout or `11`'s event-root layout. |
| `10_profile_champsim_trace.py` | Dynamic trace profile; no opcode/source inference claim is made. |
| `11_run_prefetch_event_attribution.sh` | Unified normal-matrix and event-logging standalone replay driver. |
| `12_analyze_prefetch_event_attribution.py` | L2C no-prefetch-miss outcome and normal-vs-standalone overlap analysis. |
| `15_summarize_prefetch_evidence.py` | Merge measured evidence; include explicit normal-vs-NN cache-miss comparison. |
| `22_resource_summary.py` | Summarize PQ/MSHR pressure, PF acceptance, and duplicate rate from event logs. |

`04_run_normal_prefetcher_sweep.sh` was intentionally merged into `11`. For a
counter-only normal matrix use:

```bash
MODE=normal COLLECT_EVENT_LOGS=0 \
  bash formal_NN_training/scripts/11_run_prefetch_event_attribution.sh
```

For all normal metrics plus event-level evidence, use
`MODE=normal COLLECT_EVENT_LOGS=1`.

## Shared replay-plan contract

A replay plan requires `tag`, `trace`, and `source_rel`. `08`, `09`, `11`, and
`12` share `replay/resolve_replay_plan.py`; no consumer maintains its own
path-resolution convention.

A rich list needs at least `demand_idx` or `replay_idx`, `pc`, `line`, and
`prefetch_addr`. Optional policy fields such as source, score, predicted lead,
and reject ledger are preserved as research evidence but are not required by
the replayer.

## Specialized, not default pipeline

- `13_build_cache_capacity_variant.sh` and
  `14_run_cache_capacity_sweep.sh`: frozen-list capacity sensitivity control.
  It is not a capacity-trained model experiment.
- `14_build_base_candidate_table.py`: optional normal-proposal bridge for
  future base-aware research only.
- `16_build_trace_dependency_features.py` and
  `17_prepare_v3_9_605_dependency_sidecar.sh`: static training-prefix
  dependency sidecars for 605.
- `19_build_oracle_ceiling_lists.py` and
  `20_run_oracle_ceiling_replay.sh`: offline oracle ceilings, not NN
  baselines/results.
- `21_join_decision_ledger_attribution.py`: v4 aggregate-ledger diagnostic;
  its schema requirement is deliberate.

## Retired material

`formal_NN_training/LSTM/draft/` and the historical
`LSTM_cache_action_predictor*.ipynb` notebooks use the retired action-predictor
workflow. Do not run them for the current standalone no-prefetch pipeline.

`25_build_v4_1_notebook.py` was removed because its referenced extension
payload is absent. The explicit per-trace 07/05 notebooks supersede it.
