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

`plot_results.py` writes three required offline-primary figures below
`report_plots/`. If qualifying live rows exist, it adds two secondary figures.
It does not delete or overwrite the older 14 diagnostic plots below `plots/`.

After validation, aggregation, and plotting, package a self-contained Overleaf
project from the repository root:

~~~bash
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_DIR=$EXP/runs/602_gcc_stride_prefix_seed7
RUN_DIR="$RUN_DIR" bash "$EXP/linux/package_overleaf_report.sh"
~~~

The archive includes `main.tex`, all generated `report/*.tex` inputs, three
required offline figures, and zero or two live appendix figures. Checkpoints,
streams, lists, logs, binaries, and raw results are excluded.

If live results are absent, the PDF says so and leaves the primary offline
conclusion intact. Missing live points are not interpolated.
