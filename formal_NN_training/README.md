# formal_NN_training

## Active research direction

The current project is a **standalone, base-independent neural prefetcher**.

```text
raw no-prefetch demand stream -> standalone LSTM / tiny Transformer -> keyed replay
```

This is **not** a residual/booster model:

- normal-prefetcher outputs are not neural inputs;
- normal-prefetcher coverage is not a neural label or loss weight;
- the experiment is not a `base + NN` union;
- normal prefetchers are baselines used only for final comparison.

The normal-baseline axis is still important: for each trace, compare the standalone NN against no-prefetch and every stable normal prefetcher, then report the best normal separately.

## Clean pipeline

```text
1. Normal baselines
   MODE=normal COLLECT_EVENT_LOGS=0 \
     bash scripts/11_run_prefetch_event_attribution.sh
   -> results/.../normal/summary.csv

2. Raw standalone NN data
   scripts/03_collect_no_pref_demand_events.sh
   scripts/05_build_standalone_oracle_dataset.py
   -> results/standalone_nn_data/oracle/<trace>.oracle.csv.gz

3. Model notebook
   LSTM/notebooks/LSTM_base_independent_multihorizon_policy_prefetcher.ipynb
   -> artifacts/standalone_multihorizon_lstm/

4. Valid keyed replay
   scripts/06_install_keyed_listreplayer.sh
   scripts/08_run_standalone_lstm_replay.sh
   -> results/standalone_lstm_replay/<run_tag>/summary.csv

5. Causal event evidence, only when needed
   MODE=both COLLECT_EVENT_LOGS=1 \
     bash scripts/11_run_prefetch_event_attribution.sh
   scripts/12_analyze_prefetch_event_attribution.py
```

## Dataset contract before the notebook starts

One row equals one no-prefetch L2 demand access. The required fields are:

```text
trace, demand_idx, cycle, pc, addr, line, page, page_offset, delta,
no_pref_hit, no_pref_miss, pc_line_occ
```

The notebook computes future-miss targets from this stream itself. The no-prefetch event stream is the only training data. `prefetcher_baselines/summary.csv` is evaluation metadata only.

## Evaluation

For every trace/model-size setting, report:

```text
offline: candidate-bank reachability, precision, recall, emitted requests/event
replay: IPC, L2 miss rate, useful/issued, timeliness, late requests,
        useless requests, dropped requests, speedup vs no-pref,
        speedup vs best normal, replay_transport_ok
```

A replay point is valid only when `replay_transport_ok=1`. The keyed replay uses `(pc,line,occ)` rather than a global L2 callback index, because the latter changes after prefetching changes memory timing.

See `scripts/README.md` for the exact active script list and execution order.
