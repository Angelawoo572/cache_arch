# `formal_NN_training/scripts`

The repository has two separate workflows: a pure standalone NN workflow built only from the raw no-prefetch L2C demand stream, and an analysis workflow that uses normal-prefetcher outcomes only as evidence and comparison data. No active Python script uses pandas.

## Script order

```text
01_parse_prefetch_behavior_audit.py       Parse normal-prefetcher counter logs.
02_patch_pythia_demand_logger.sh          Enable L2C demand-event logging.
03_collect_no_pref_demand_events.sh       Collect raw no-prefetch L2C demand events.
04_run_normal_prefetcher_sweep.sh         Run a counter-only normal baseline sweep.
05_build_standalone_oracle_dataset.py     Build raw standalone NN input data.
06_install_keyed_listreplayer.sh          Build keyed PC-line-occurrence ListReplayer.
07_prepare_keyed_replay_input.py          Convert frozen NN exports to replay inputs.
08_run_standalone_lstm_replay.sh          Run keyed standalone replays.
09_parse_standalone_lstm_replay.py        Parse replay outputs.
10_profile_champsim_trace.py               Profile dynamic trace PC/branch/register/memory behavior.
11_run_prefetch_event_attribution.sh       Collect normal and standalone L2C event evidence and a normal summary.
12_analyze_prefetch_event_attribution.py   Compare timely, late, and residual demand outcomes.
13_build_cache_capacity_variant.sh         Build one reversible L1D/L2C/LLC capacity-specific binary.
15_summarize_prefetch_evidence.py          Merge trace, baseline, attribution, and replay evidence.
```

## Evidence first

Before modifying the notebook, run 10 → 11 → 12 → 15. Script 11 now writes `normal/summary.csv` from the same normal runs that produced the per-demand logs, so running script 04 as well is unnecessary for an evidence campaign. Use script 04 only when a quick counter-only baseline refresh is needed.

The resulting CSV/Markdown evidence identifies trace composition, normal prefetcher traffic/accuracy/timeliness, residual PC-delta-offset contexts, and the normal-only versus standalone-only timely demand misses.

The standard ChampSim input trace has dynamic PCs, branch flags, register IDs, and memory addresses. It has no opcode bytes. Script 10 can attach assembly only when an original benchmark executable is supplied with `--binary` and its address space matches trace PCs.

The standalone oracle remains:

```text
formal_NN_training/results/standalone_nn_data/oracle/<trace>.oracle.csv.gz
```

It is derived only from no-prefetch L2C demand events. Normal-prefetcher results never become standalone NN labels or inputs.

A frozen LSTM export records selected actions only. Therefore `no_earlier_selected_standalone_export` is not proof of candidate-bank absence. A future notebook must export a full per-candidate decision ledger with source, score, timing probabilities, eligibility, dedup result, and reject reason.

## Capacity

The previous automatic frozen-list capacity sweep was removed. It would have mixed a baseline-capacity frozen L2C policy with changed cache configurations, which is only a sensitivity control and is not a capacity-trained NN result.

Keep `13_build_cache_capacity_variant.sh` for a later, explicitly designed experiment. A real capacity-trained NN point requires a new capacity-specific no-prefetch collection, oracle, Colab training/export, and matching replay binary.
