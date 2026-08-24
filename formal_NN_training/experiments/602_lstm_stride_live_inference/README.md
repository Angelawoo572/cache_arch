# 602 Stride training-prefix sufficiency

The primary study in this directory is the h8/h16 training-prefix sweep under
the **original 602 offline keyed-replay protocol**. The completed sibling
experiment [`602_offline_lstm_stride`](../602_offline_lstm_stride/) remains
unchanged and is the authority for the model, labels, Python inference, keyed
replay transport, ChampSim setup, and metric definitions.

## Primary scientific question

Holding the existing model and protocol constant, what is the smallest
trace-start prefix `N` whose offline keyed-replay system result is equivalent
to the corresponding 20M model?

The comparisons are always:

- h8-`N` versus h8-i20m;
- h16-`N` versus h16-i20m.

h8 is never compared with h16-i20m to decide training-size sufficiency.

## Original offline protocol

- trace: `602.gcc_s-734B`;
- training prefix: trace start, warmup 0, measured instructions `N`;
- held-out evaluation: one fixed stream;
- system measurement: 25M ChampSim warmup + 25M measured instructions;
- inference: Python causal offline inference;
- transport: one keyed replay list per model through the same list-replayer;
- references: the same no-prefetch and offline Stride artifacts;
- labels: conventional Stride actions, never NN inputs;
- external NN inputs: current PC and cache-line address only.

Only `N` changes within each hidden-size curve. Seed, model revision, encoder,
state router, 12 epochs, TBPTT length 256, PC batch size 128, Adam, learning
rate 0.002, teacher, class-balance formula, held-out stream, replay format,
ChampSim binary/configuration, and metric parser stay fixed.

Class weights are recomputed from each prefix's own labels:

```text
weight[c] = decision_rows / (2 * prefix_label_frequency[c])
```

The 20M class frequencies are not reused. Fixed epochs mean larger prefixes
receive more optimizer steps, so this study measures required trace and
training work under the current recipe; it does not isolate data diversity
from optimizer-step count.

## Interpretation policy

Three concepts remain separate:

1. **Minimum reaching at least 99/99.5/99.9% of 20M IPC** is one-sided
   performance sufficiency. A point above 20M may satisfy it.
2. **Minimum within +/-1.0/0.5/0.1% of 20M IPC** uses two-sided relative
   error.
3. **Stable 0.5%-IPC plateau** requires a candidate and every subsequent
   observed same-hidden-size budget through 20M to remain within +/-0.5% of
   same-H i20m.

Coverage, L2 miss rate, request pressure, student act rate, timeliness, and
replay-action count are reported separately. Close IPC/cache outcomes may be
called system-performance equivalent. Policy equivalence is not claimed
unless action and traffic behavior are also explicitly close.

The current seed-7 data place the stable offline IPC plateau at approximately
i1m for both h8 and h16. h8-i250k and h16-i100k remain real aggressive
high-performing candidates, but their different traffic/coverage/act behavior
prevents calling them 20M-equivalent policies.

This conclusion is limited to `602.gcc_s-734B`, seed 7, and the original
offline keyed-replay protocol until multi-seed evidence exists.

## Audit outputs

`validation/validate_offline_fairness.py` writes a point-by-point JSON/CSV/TeX
audit of all fixed controls, source hashes, prefix-local class weights, shared
evaluation identity, replay format, keyed-replayer binary, references, and
metric parser.

`validation/validate_20m_regression.py` compares h8/h16 i20m against
`602_offline_lstm_stride_compact_hurdle_v9_seed7`. Exact artifact identity and
metric regression are separate. When the old run tree is incomplete, the
output says `historical-metric regression only` and does not claim exact
artifact parity.

ChampSim revisions may print either a 25M measurement-only final instruction
counter or a 50M cumulative warmup-plus-measurement counter. The fairness
validator records that counter scope explicitly while independently checking
the run-script contract of 25M warmup + 25M measured instructions. A final
superscalar retirement batch may overshoot either counter by at most four
instructions; the observed 25,000,003 is therefore recorded as a +3 boundary
overshoot rather than a protocol failure. It never mislabels a cumulative 50M
counter as the measured window.

The old-versus-new validator also keeps reference-log identity separate from
metric regression. A tolerance-based offline-Stride cache-metric PASS does not
establish exact log, replay-list, or artifact parity.
Offline-Stride coverage uses the corresponding no-prefetch L2-load-miss count
as its denominator. A genuinely unavailable old metric is reported as
`UNAVAILABLE` and cannot by itself fail the regression; it also cannot be used
to claim exact artifact parity.

## LaTeX-only Overleaf report

Sacramento does not need matplotlib or a local TeX installation. After the
JSON analysis is regenerated, `python/generate_overleaf_figures.py` writes the
three primary figures as PGFPlots `.tex` fragments. The packaging script
creates a source-only Overleaf ZIP containing `main.tex` and generated TeX
inputs; it excludes PNG/PDF files and every raw experiment artifact. Overleaf
renders the figures and final PDF.

## Secondary functional live inference

Live validation is secondary deployment evidence. Frozen C++ inference runs
inside the ChampSim callback with no replay list, teacher, optimizer,
gradient, or weight update. Current live results have **zero modeled NN
inference latency**. Host wall-clock nanoseconds are not simulated CPU cycles.

The default recommended live set is:

- primary: h8-i1m, h8-i20m, h16-i1m, h16-i20m;
- aggressive candidates: h8-i250k, h16-i100k.

Every live row has two comparisons: live-`N` versus offline-`N` for
same-checkpoint implementation parity, and live-`N` versus same-H live-i20m
for live training-size sufficiency. Missing live points are never
interpolated. `LIVE_ALL_VALID=1` remains available but is not required for the
primary offline conclusion. A full optional curve may be split into one h8
process and one h16 process because those selectors write disjoint point
directories; see `COMMANDS.md` for the tested two-process form.

See [COMMANDS.md](COMMANDS.md) for the analysis-only, LaTeX/Overleaf,
live-validation, SCP, and source-only Git workflow. Completed training and
keyed replay are not rerun by those analysis commands.
