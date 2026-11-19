# Reproduce the complete-recording evaluation

Run from the repository root on the experimental branch. On the original Mac start with `source .runtime/project_env.sh` to place temporary files and caches under the cache project. Requirements include Git, CMake, Make, a C++17 compiler, zlib, curl, gzip, xz, Python 3.10 or later, NumPy and Matplotlib. The native scripts use the original Pythia two-core source and the project's existing conventional controls. Recordings and raw simulator products live under this study's `raw/` folder. The branch stores source, small machine-readable measurements, and reproduction instructions.

1. Follow [intake](intake/README.md) to download and verify compressed sources and create the registered native views. Use `intake/prepare.py pr_heldout_entire` for the additional complete PageRank view. Validation must finish before using a newly materialized input.
2. Follow [native build and validation](native/README.md) to reproduce baseline compatibility and the far-window placement control. The following commands run the main windows and the independent full-recording extension:

```sh
p=llc_hybrid_prefetch/full_workload
python3 -B "$p/native/batch.py" --jobs 6 --policies nopf stride multi spp temporal
python3 -B "$p/native/batch.py" --jobs 4 --cases bfs_discovery_far mcf17_discovery_far --policies nopf stride multi spp temporal
python3 -B "$p/native/entire_recording.py" --jobs 3
python3 -B "$p/native/summarize.py"
python3 -B "$p/native/summarize.py" --cases pr_heldout_entire --out "$p/native/entire"
python3 -B "$p/streams/measure.py"
python3 -B "$p/interference/run.py" --jobs 2
python3 -B "$p/interference/measure.py"
```

3. Reproduce the fitted rule exports, matched interventions and causal analysis using the commands in [discovery commands](discovery/README.md) and [the registered protocol](discovery/protocol.md). They use the earlier published donor intervals. The prospective rule and degree/grant schedules are fixed. The supplement documents post hoc controls separately. Verify the generating address, request lifecycle, reference-arm outcome, first use since the start of the recording, and the cost of displaced offers.

```sh
python3 -B "$p/streams/address_relations.py"
python3 -B "$p/streams/audit_first_reference.py"
python3 -B "$p/figures/plot.py"
```

4. For an independent platform repeat, build and run on Linux, then compare configuration, pages, quarters, native statistics and all dynamic outcome words using [the cross-host procedure](verification/README.md). Host execution time can change with CPU or job concurrency; the comparison uses simulated cycles and events.

The separate matched arithmetic intervention uses the same native source, with one additional causal candidate generator. Run the following after native build and registry validation:

```sh
python3 -B "$p/streams/replay.py" --build --cases bfs_discovery bfs_transfer pr_heldout soplex06_heldout --jobs 2
python3 -B "$p/streams/score_arithmetic.py"
```


Repeat the two matched increment interventions for all three 200M-instruction MCF regions, including request-level attribution to the 418 discovery targets, using the [MCF transfer procedure](streams/mcf_transfer/README.md).

Three independent checks supplement those summaries. The [continuous-ingress study](continuous/README.md) repeats complete 200M-instruction Stride runs and checks unchanged machine outcomes. The [held-out lookahead comparison](long_lookahead/README.md) tests the inherited two-stride action at full scale. The [single-address experiment](single_address/README.md) changes one earlier candidate in two otherwise normal 200M runs. Finally run:

```sh
python3 -B "$p/verification/check_tables.py"
```
