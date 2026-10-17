# Checkout, source and raw-data recovery

The original `/Users/angelawoo/Documents/CMU/ece/cache` directory and its earlier
results were preserved. Its pre-existing Git index/refs were damaged. This study
uses an independent healthy checkout on `experiment/602-gate-mechanism`, based on
`experiment/602-stride-dynamic-inference`, with origin
`https://github.com/Angelawoo572/cache_arch.git`.

The starting source was the code that actually produced the completed
`frozen_formal_20261016_v2` results, not an older script inferred from a branch
name. The 21 archived active source files were compared byte for byte with that
snapshot before bringing in the already validated, uncommitted correctness
fixes. The recovered source retains independent seed loading, finite FIFO and
nonlinear controls, full cache warmup, completion-only state publication,
per-cycle service and arrival-time shadow-teacher snapshots. The separate
canonical h/c-history supplement was not transplanted into this study.

All new full formal executions use isolated build `build_20261016_213702` and
ChampSim base `838be17e`, with its consistent dependency setup. New ChampSim
instrumentation is installed by the existing dynamic experiment's
[installer](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/patches/install.py).
The runner checks actual frozen source/config/binary/table equality before
resuming outputs. It preserves incompatible or unfinished attempts rather than
declaring them complete. No source checksum ledger or performance pilot was
introduced.

## Server paths

- Source: `/home/qianruw/cache_gate_mechanism_20261016`.
- Full new raw results, logs, configurations, models used for fitting, snapshots,
  isolated build and certificates:
  `/home/qianruw/cache_gate_mechanism_20261016_runs`.
- Existing compatible reference raw data and frozen checkpoints:
  `/home/qianruw/cache_formal_20261016_runs_v2`.
- Frozen NN models:
  `models/seed{7,17,27}/balanced/export/model.bin` under that reference root.
- Actual fitted tables:
  `pc_gate/seed{7,17,27}/pc_gate.tsv` under the new root. Copies with provenance
  are committed in [pc_gate_fit](pc_gate_fit/).

Authentication was reused through the already established Sacramento connection.
No credential, permission, authentication, host-verification or security setting
was changed. Raw traces, large raw logs, model binaries and build products are
excluded from this commit.

## Reproduce or resume

Use the [run guide](../../formal_NN_training/experiments/602_gate_mechanism/RUN_GUIDE.md).
`run_gate.py pipeline` resumes completed compatible outputs and limits execution
to eight concurrent full simulations. `run.json`, `summary.json` and completion
markers distinguish submitted/running/complete states. A complete runner result
still requires the independent analysis/reference compatibility certificate.
All 71 logical rows and their source protocols remain in the final audit.

After copying the raw roots locally, generate a fresh analysis directory:

```bash
EXP=formal_NN_training/experiments/602_gate_mechanism
python3 "$EXP/analysis_gate.py" --raw NEW_RAW --prior-raw ORIGINAL_V2_RAW --out NEW_ANALYSIS
python3 "$EXP/summarize_gate.py" --raw NEW_RAW --analysis NEW_ANALYSIS --out NEW_SUMMARY
```

Plotting requires Matplotlib. Every aggregate retains the individual seed rows;
three training seeds are not interpreted as independent workload samples or as
confidence intervals. Startup curves use saved snapshots, while mechanism
histograms accumulate the entire ROI.
