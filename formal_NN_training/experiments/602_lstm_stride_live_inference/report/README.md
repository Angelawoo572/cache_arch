# Offline-primary report build

The tracked TeX source is organized around the original 602 offline
keyed-replay protocol. It contains no hardcoded sweep measurements.
`compare_offline_live.py` regenerates the measured same-hidden-size conclusion
JSON/CSV/TeX, key-budget tables, auxiliary behavior differences, and optional
live appendix from ignored run artifacts.

Before compiling, run both read-only validators. Their TeX summaries state
whether the fairness audit passed and whether 20M regression used full old
artifacts or only historical metric anchors. The report never converts an IPC
anchor match into an exact-artifact-parity claim.

`generate_overleaf_figures.py` writes three required offline-primary figures
as PGFPlots TeX fragments below `report/`. It uses only Python's standard
library and does not need matplotlib. Qualifying live rows remain in the
secondary generated table. No old PNG plot is read, deleted, or overwritten.

After validation and aggregation, generate the figure fragments and package a
self-contained Overleaf project from the repository root:

~~~bash
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_DIR=$EXP/runs/602_gcc_stride_prefix_seed7
python3 "$EXP/python/generate_overleaf_figures.py" --run-dir "$RUN_DIR"
RUN_DIR="$RUN_DIR" FORCE=1 bash "$EXP/linux/package_overleaf_report.sh"
~~~

The archive includes `main.tex` plus ten generated `report/*.tex` inputs,
including the three primary figures. It contains no PNG/PDF files. Checkpoints,
streams, lists, logs, binaries, JSON/CSV, and raw results are excluded.

Upload the ZIP to Overleaf and select `main.tex` as the main document. Overleaf
renders PGFPlots and the final PDF; do not install matplotlib or run
`pdflatex` on Sacramento.

If live results are absent, the PDF says so and leaves the primary offline
conclusion intact. Missing live points are not interpolated.
