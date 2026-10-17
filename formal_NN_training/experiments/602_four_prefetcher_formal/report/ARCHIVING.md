# Preserve the complete v2 results

- Run the archiver only after the final local analysis has finished. It accepts `PASS` with exactly64 expected/completed rows (56 test +8 validation), all explicit checks passing, strict full callback verification, and generated PNG/SVG figures. A partial analysis or a superseded v1 analysis is rejected.

- From the healthy checkout root:

```sh
/opt/anaconda3/bin/python -B \
  formal_NN_training/experiments/602_four_prefetcher_formal/report/archive_results.py \
  --analysis /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2/analysis_final \
  --raw /Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261017/raw_v2
```

- `--destination` optionally selects a fresh full-archive directory.
- Its default is this experiment's `results/formal_20261017`.
- The four per-policy destinations are always the new `results/formal_20261017` in the Streamer, AMPM and SPP live siblings and the Stride dynamic sibling.
- All five final directories must be absent; the script never overwrites an existing archive.
- It stages all copies and subsets before publishing them.
- Any preserved staging directory after a failure contains recovery evidence and is not silently deleted or reused.

- The full archive copies every lightweight analysis CSV/PNG/SVG/Markdown/JSON/ HTML artifact byte for byte, excluding `raw_snapshots.json`.
- Individual files over64MiB are rejected rather than silently omitted.
- Actual raw `build.json`, `correctness_gate.json`, `selection.json`, the built source snapshot's `config.json`, and the source route manifest are copied unchanged under `metadata/`.
- Sync the actual build/source snapshot before archiving; a current config with no actual-build evidence is not a substitute.
- Models, executables, traces, raw callback/snapshot streams and large logs stay at the complete raw root and are excluded from the lightweight Git archive.

- Each policy subset keeps its own rows plus each exact matched NoPF control. It writes `per_run.csv`, both phase/full-ROI action-histogram tables, `phase_progress.csv`, `progress_snapshots.csv`, and `classified_requests.csv` with the same original columns and field values. These are row filters, with no metric recomputation, averages, prefix extrapolation or old-protocol rows. A README links to the full tables/figures/definitions, and a subset manifest records every retained run ID. Sparse event tables may have no NoPF rows.

- The full archive certificate records copied-byte equality, the actual build, the nine required full student callback logs, and all four subset manifests. It does not claim finite-cost IPC, new independent seeds or a stateful novelty result. Original negative and mixed results remain present.
