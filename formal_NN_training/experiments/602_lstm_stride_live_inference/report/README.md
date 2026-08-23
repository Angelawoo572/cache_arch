# Report build

The tracked TeX file makes the original offline keyed-replay comparison the
primary result and contains no invented measurements. Every h8 point is
compared with h8-i20m and every h16 point with h16-i20m.
`compare_offline_live.py` writes the actual conclusion
CSV, JSON, and TeX into the ignored run directory. It distinguishes one-sided
performance sufficiency, two-sided equivalence to 20M, and a stable plateau
over all subsequent observed budgets. The plateau IPC band is 0.5%; coverage,
request pressure, student act rate, timeliness, and replay actions are
reported separately and do not silently redefine the IPC plateau. Functional
live inference is a secondary section with zero modeled NN latency.

After aggregation and plotting, package a self-contained Overleaf project from
the repository root:

~~~bash
EXP=formal_NN_training/experiments/602_lstm_stride_live_inference
RUN_DIR=$EXP/runs/602_gcc_stride_prefix_seed7
RUN_DIR="$RUN_DIR" bash "$EXP/linux/package_overleaf_report.sh"
~~~

If live results are absent, the PDF explicitly reports that execution is still
required rather than filling unavailable metrics with zero.
