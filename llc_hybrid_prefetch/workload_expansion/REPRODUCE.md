# Reproduce the workload expansion

Run from `llc_hybrid_prefetch/workload_expansion/` in the experiment branch. Commands below use Bash or Zsh, Python 3.9+ with NumPy, pandas and Matplotlib, curl, a C++14 compiler and the existing Pythia 1.3 source/build. The primary simulator is the project's instrumented two-core build with one active core. A clean installation first follows the [parent simulator instructions](../README.md) and [conventional controls build](../opportunity_study/conventional/conventional_baselines.md). Local and Sacramento archives keep the large prefixes, histories and native logs in `raw/`; the branch contains compact measurements and source. A missing local cached SPEC/GAP prefix can be fetched by URL using the registered discovery source. Reserve rows have no payload path.

## Input and simulator

The default intake is 801,024 complete 64-byte records per discovery source, including a trailing decoder guard. The long supplement uses 5,001,024 records in separate Google discovery files. Caches are bounded; the experiment does not require downloading entire multi-gigabyte source objects. On the project Mac source the environment file in the cache root first, to keep tool caches under the project.

```sh
python3 -B intake/register.py --refresh-manifest
python3 -B intake/fetch.py --jobs 4
python3 -B intake/audit.py
python3 -B intake/verify_reader.py
python3 -B intake/fetch_long.py --jobs 4
python3 -B intake/audit.py --long-google
python3 -B intake/verify_reader.py --long-google
python3 -B ../opportunity_study/conventional/build.py --jobs 6
```

## Native windows and demand identity

```sh
google=(gg_arizona gg_charlie gg_delta gg_merced gg_tahoe gg_tango gg_yankee)
python3 -B screen/run.py --jobs 4
python3 -B screen/history.py --jobs 4
python3 -B screen/measure.py --jobs 4
python3 -B screen/pairwise.py
python3 -B screen/producers.py

python3 -B screen/run.py --skip 450000 --output raw/replication --jobs 4
python3 -B screen/history.py --output raw/replication --jobs 4
python3 -B screen/measure.py --output raw/replication --summaries transfer --jobs 4
python3 -B screen/pairwise.py --output raw/replication --summaries transfer
python3 -B screen/producers.py --output raw/replication --summaries transfer --case gg_delta

python3 -B screen/run.py --skip 2000000 --warmup 1000000 --roi 2000000 \
  --prefix-dir "$PWD/raw/intake/long" --output raw/long_google --case "${google[@]}" --jobs 4
python3 -B screen/history.py --output raw/long_google --case "${google[@]}" --jobs 4
python3 -B screen/measure.py --output raw/long_google --summaries transfer/long_google --case "${google[@]}" --jobs 4
python3 -B screen/pairwise.py --output raw/long_google --summaries transfer/long_google --case "${google[@]}"
python3 -B screen/producers.py --output raw/long_google --summaries transfer/long_google --case gg_delta
```

The runners write per-arm input/geometry metadata and completion markers. The analyzers verify the entire measured input load sequence, per-core retirement window and physical-page mapping. Paired cycles always use differences inside one execution. Merges, ordinary cache hits, late tracked requests, direct timely first uses, and demands filtered before the LLC are separated. `cache_events.csv`, `rejections.csv`, `identity_audit.csv` and `performance.csv` beside the corresponding summaries retain the denominators. Archive `paired_outcomes.csv` records contingent outcome counts, and `producer_pairs.csv` links the PCs generating paired physical requests.

## Pattern discovery, controls and checks

```sh
python3 -B patterns/discovery.py
python3 -B patterns/controls.py
python3 -B patterns/window.py
python3 -B patterns/verify.py
for script in discovery controls window; do
  python3 -B "patterns/$script.py" --history-root raw/long_google/history \
    --output patterns/long_google --case "${google[@]}"
done
python3 -B patterns/verify.py --history-root raw/long_google/history --output patterns/long_google

python3 -B transfer/evaluate.py --jobs 4
python3 -B transfer/evaluate.py --test-root raw/long_google --output transfer/long_google --case "${google[@]}" --jobs 4
python3 -B transfer/verify.py
python3 -B screen/transfer_witnesses.py
python3 -B screen/charlie_probe.py
python3 -B temporal/study.py --jobs 4
python3 -B temporal/verify.py
```

To test transfer from the parent study, reuse the two LBM tables in `../transition_study/raw/predictability/`. If absent, reproduce the [donor histories and exports](../transition_study/replay/README.md) first, including `python3 -B ../transition_study/replay/prepare_tables.py`. The base donor weights and original published grant function are reused unchanged; an experiment-local copy resets only start/stop indices to score the discovery ROI.

```sh
python3 -B transfer/frozen_lbm.py
python3 -B transfer/tahoe_alias.py
python3 -B figures/plot.py
```

Each workload family has one role in this round. None of the commands evaluates a held-out workload. Independent native replay uses the three `screen/run.py` commands above with new output paths `raw/remote-repeat/A`, `raw/remote-repeat/B`, and `raw/remote-repeat/C` on Sacramento. After the primary raw runs are in place, run `python3 -B screen/verify_repeat.py` to compare core retirement, cycle counts and all recorded system statistics in all 156 configurations. Native equality establishes reproducibility across the two builds; the diagnostic tables still describe hypotheses about future prefetches.
