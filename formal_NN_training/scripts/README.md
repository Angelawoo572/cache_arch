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
10_profile_champsim_trace.py               Profile trace-level dynamic PC/branch/memory information.
11_run_prefetch_event_attribution.sh       Collect normal and standalone per-event L2C evidence.
12_analyze_prefetch_event_attribution.py   Compare normal vs standalone timely/residual demand outcomes.
13_build_cache_capacity_variant.sh         Reversibly build L1D/L2C/LLC capacity-specific binaries.
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

## Explainability and capacity workflow

Scripts 10–12 are analysis-only. Script 11 re-runs normal prefetchers and
frozen standalone exports with the existing L2C demand-event logger, and script
12 produces per-run timely/late/residual summaries plus normal-vs-standalone
attribution. Existing frozen lists record only selected candidates. To prove
whether a standalone target was absent from the candidate bank versus rejected
by threshold/ranking/timing/LRU dedup, the next Colab notebook must export a
full decision ledger for every candidate.

The standard ChampSim `input_instr` trace record does not contain opcode bytes.
Script 10 reports the PC, branch, register, and memory-address information that
exists in the trace; source-level assembly requires the original executable and
matching address map.

Script 13 only builds a cache-capacity variant. A capacity-aware model experiment
requires capacity-specific no-prefetch collection, oracle construction, Colab
training, frozen export, and replay. Replaying a baseline-capacity frozen L2C
list under a new capacity is useful as a system-sensitivity control, not as a
capacity-trained model result.

Execution order: 04 baseline sweep, 03 raw collection, 05 dataset build, notebook training, 06 keyed replayer build, then 08 replay. For the analysis branch: 10 trace profiling, 11 event collection, then 12 attribution analysis.
