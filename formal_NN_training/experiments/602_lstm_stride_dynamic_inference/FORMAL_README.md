# Frozen L2 teacher-action study, 2026-10-16

**Archived main-v2 protocol record.** This branch adds the separate
[gate/payload follow-up](../602_gate_mechanism/RUN_GUIDE.md), with a new raw root
and measurement-hook protocol. Use that guide for the current study. Reproduce
the original completed v2 study from its archived actual source/build; do not
install this branch's new hooks over a completed old simulator or overwrite
the old raw root. Old results and model files remain unchanged references.

The recorded main protocol is `frozen_formal_20261016_v2` in `formal_config.json`.
This is a live-inference study. Existing replay, historical live and old
cold-suffix finite-cost results remain scientific references, not inputs to
the new result tables. No pilot is run by the formal entry point.

## Fixed protocol

- One Sacramento machine, one archived ChampSim base and one isolated built
  executable for all formal rows; no changes to the active old simulator.
- Training: trace-start 0-1M records, original NoPF L2 LOAD stream, 12 epochs,
  h8, chunk256, PC batch128, Adam lr0.002. Seeds7/17/27 are independent actual
  training runs. Balanced/unweighted loss pairs share exact initial tensors,
  data and optimizer-step budget; saved tensors prove the pairing directly.
  The shared NoPF training-stream path retains an old `seed7` directory name;
  it is dataset provenance, not a shared neural checkpoint. Actual model loads
  use `models/seed7`, `models/seed17` and `models/seed27` independently.
- Validation: warm caches on records1-5M; measure records5-10M.
- Test ROI0: warm caches on20-25M; measure25-50M.
- Test ROI1: warm caches on70-75M; measure75-100M.
- Trace coordinates are nominal. Actual warmup retirement overshoot and
  `trace_origin_records` are retained; no counters are silently shifted.
- Cache warmup issues no prefetches for every method. Conventional Stride,
  diagnostic shadow Stride and SPP observe the full same warmup. Their states
  are retained at ROI. NN h/c observes only the selected zero/short/long
  warmup history; pending NN work is discarded at ROI.
- NN weights stay frozen. Only exact-PC h/c changes at modeled completion.
  One serial engine has II=service time,4MAC/cycle,16B/cycle weight/scratch
  bandwidth, one4-cycle nonlinear unit, input FIFO16, output FIFO32, decoder
  cap32 and64PC state entries. Zero-cost NN uses the same capacities.
- Same-page4KiB policy and64B lines apply throughout. Legacy Stride includes
  the current line; remove-current/forward1 remove it; forward2 adds a second
  lookahead. SPP is the existing native `spp_dev2`, configured by repeating
  the prefetcher option, rather than comma separation.
- Each workload's common cumulative request ceiling comes only from its
  validation forward1 request rate. All capped policies use that same ceiling.
  Actual counts can differ; reports retain realized gaps rather than claiming
  that equal ceilings force equal request counts.
- 605.mcf and623.xalancbmk use the602 checkpoint: these are cross-workload
  transfer tests, not workload-specific retraining.

## Run and recover

Actual SSH endpoint: `qianruw@sacramento.ece.local.cmu.edu`.
Existing authenticated control socket: `/tmp/cache602-dynamic-ssh`.
Source: `/home/qianruw/cache_formal_20261016`.
Raw outputs/models/logs: `/home/qianruw/cache_formal_20261016_runs_v2`.
The superseded v1 validation and its failed launch/trainer logs are preserved
in `/home/qianruw/cache_formal_20261016_runs`, outside all v2 tables.

On Sacramento, from the source root:

```sh
bash formal_NN_training/experiments/602_lstm_stride_dynamic_inference/linux/run_formal_study.sh
```

This verifies the current build and source bytes, runs correctness tests,
resumes completed compatible formal rows, trains missing models, verifies
training identity and real-checkpoint Python/C++ parity, selects budgets on
validation and executes the stored test matrix. It refuses incompatible
active jobs and stale sources/models. Failed attempts are moved to preserved
directories; prior launcher logs are copied before reuse. To intentionally
change engine source, first run `formal.py build` to create another isolated
build. Training changes require a new immutable protocol/raw root.

```sh
/home/qianruw/venvs/cache-few-nn-cpu/bin/python formal_NN_training/experiments/602_lstm_stride_dynamic_inference/formal.py status
```

Only a `complete` run with a verified final ROI snapshot enters analysis.
`pipeline_result.json` is written only after all stored test rows complete.
All config/model paths and child environments are recorded per row. Source
and model byte copies prevent stale reuse without requiring a SHA ledger.

## Interpretation and metrics

The arrival-time shadow teacher observes even dropped NN inputs. Its action
list is copied into the event before admission, never queried at completion.
The teacher-silent ablation suppresses output only; compute, admission and
h/c updates are preserved. Request category counts are descriptive. IPC
differences from the actual paired intervention support attribution.

Use LOAD-only timely/late request cohorts and matched NoPF denominators.
Keep generated, budget/page suppressed, cache new/merged/rejected, duplicates,
evicted-unused, resident-unused and pending outcomes distinct. Evicted-unused
is not a counterfactual pollution estimate. Histograms count all events;
bounded example logs are not used as distribution denominators.

DRAM counters instrument actual controller queue events, bus starts and modeled
bus completions; accepted payload and completed bus bytes are separate.
Initial/end queues and in-progress transfers are retained. NN weights are one
resident FP32 copy; internal weight reads are separate from modeled zero
weight DRAM traffic. No DRAM metric is estimated from prefetches issued.

The finite engine is an explicit architectural cost model, not synthesized
hardware. Standard ChampSim cache warmup uses zero-latency warmup behavior;
actual completed per-PC NN history is reported. The history study does not
infer a minimum from the first cumulative IPC crossing.

Final review found that the original v2 history arms also inherit different
PC tags/LRU/occupancy, which changes the table-control service term. They are
retained as operational history treatments, not a pure h/c-value intervention.
A separate full-length `hc_control` supplement fixes the initial PC metadata
across zero/short/long histories and reruns its own long-history control and
NoPF references in one isolated build. See the final study index for that
supplement's metadata equality certificate and results; do not substitute the
old v2 long row as its pure-history control.

The original local `.git` was already corrupt (empty index/refs). It is
preserved. Installed source edits are uncommitted; installation manifests
link per-file backups. Do not reset that repository to recover this study.
