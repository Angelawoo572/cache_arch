# Conventional LLC baselines

[Sacramento results](RESULTS.md)

- Run NoPF, Next-Line, Stride, Streamer, AMPM, and SPP through one shared LLC issuer. All six use the same CPU, L1D, L2, LLC, DRAM, and replacement configuration. L1D and L2 prefetching are disabled. Each core has independent predictor state.
- Reuse Pythia v1.3's Stride, Streamer, and AMPM candidate generators. The LLC-only SPP port reuses the native signature and pattern tables, limits lookahead to four stages, and updates confidence from timely LLC use. All policies train their state during warmup and start issuing at the ROI.
- The first pass uses a limit of one accepted prefetch per L2D demand miss admitted to the LLC read queue. The common issuer skips cross-page, resident, and in-flight lines. The policies can still differ in issued requests and DRAM traffic.

## Reproduce the two-core 602 run

Run from `llc_lstm_research` on Sacramento. Use the same trace on both cores and skip the first 10,000,000 trace records on each. Set the warmup target to 200,000 retired instructions and the ROI target to 1,000,000 per core. The simulator maps the two copies to separate physical pages.

```sh
JOBS=16 ./conventional/build.sh 2 conventional/runs/build-2core
python3 conventional/run.py \
  --binary conventional/runs/build-2core/pythia/bin/champsim \
  --output conventional/runs/602_dual_10m \
  --skip 10000000 --warmup 200000 --instructions 1000000 --degree 1 \
  ../traces/602.gcc_s-734B.champsimtrace.xz \
  ../traces/602.gcc_s-734B.champsimtrace.xz
```

- `summary.csv` has one row per policy. `summary.json` also contains per-core IPC, request counts, and runtime. Each policy directory stores the command, results, and raw simulator log. The run folder is kept on Sacramento and in the local cache.
- Pass `--events` to log ingress order and cycle, completed lookup order and cycle, core, PC, physical line, MSHR occupancy, and hit/miss. A miss can wait for an MSHR or DRAM slot before lookup completes. `rq_full` counts failed enqueue attempts and can include retries. Use NoPF events from a development region for labels. Reserve the 602 segment above for final comparison after model and thresholds are chosen.
- `net_miss_reduction` is relative to NoPF LLC MPKI. The event IDs in these logs are per-run queue IDs. A stable cross-policy demand ID is needed to measure causal miss coverage.
- A demand that merges into a prefetch already in the LLC MSHR counts as late. A prefetch that loses the race while still in the prefetch queue is redundant when it reaches the occupied line or MSHR. Timely slack measures cycles from fill to the first qualifying demand hit.
- Upstream Pythia keeps cores running until the last core finishes. Per-core instruction counts, IPC, and LLC demand lookups stop at each core's ROI target. Traffic and shared-cache activity can include activity from a core that reached its target early.
