# Frozen SPP student: the retained checkpoint in live inference

## Retained real weights and current formal output

This sibling carries the real small [original checkpoint](models/original_checkpoint.pt)
(16,790 bytes), byte-identical [source training metadata](models/source_training_metadata.json),
and [portable FP32 model](models/model.bin) (12,024 file bytes). The
portable model contains 2,865 parameters, or 11,460 resident FP32
weight bytes; file headers and tensor names account for the larger file size.
These are trained weights, not placeholders, a random reinitialization, a trace,
or a replay list. The retained checkpoint and metadata match the original
source run byte for byte; the portable binary matches the previously verified
export byte for byte. All copies are in cache.

The current formal destination is
`/home/qianruw/cache/formal_study_20261017/raw_v2/models/spp/model.bin`,
with an identical `immutable_model.bin` beside it. The copied model is frozen;
v2 changes callback-log flushing only and does not retrain or select weights.
Original source run paths above remain provenance. The retained files can be
used directly as checkpoint/metadata inputs to this sibling's exporter and parity
script, in a fresh output directory. Follow the
[guarded model-restoration and run commands](../602_four_prefetcher_formal/README.md)
before `build`: the builder does not automatically export or stage these files.
That guide restores absent files, compares existing bytes, and never overwrites
a different model. The final main table is `raw_v2/analysis_final`; v1 callback
logs have a known unflushed tail and remain excluded historical evidence.

The portable files make this model's numerical reproduction self-contained in
the checkout. Full formal IPC still needs the preserved 602 trace, consistent
ChampSim/libbf dependency, recorded machine, and a fresh build/certificate;
a standalone model or prefix parity is not a completed ROI experiment.

This sibling exports and checks the actual SPP empirical-prior student from
`602_offline_lstm_spp`. Its executable state is one chronological global h/c
pair. It is not the Stride checkpoint, keyed replay, a per-PC state table, or
online weight training. The common C++ implementation is
`formal_NN_training/common/live602_runtime`.

The retained seed7/h8 checkpoint under the cache root is:

`formal_NN_training/experiments/602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/colab_output/independent_delta_spp_lstm_h8/model.pt`

`python/export_and_validate.py` requires the known
`compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2` revision, LSTM model
family, 65 runtime features, four mixture components and fill levels `[2,4]`.
It rejects a different tensor set. `LV602R1` export preserves the actual FP32
tensors and decoder schema; the scalar runtime validates finite parameters,
shapes and the complete model size before use.

## Inputs, state and complete actions

An event encodes the aligned byte address (`line << 6`) as 64 LSB-first bits,
plus demand/fill kind. There is no PC, hit/miss, cycle, teacher action or private
teacher table in the neural feature vector. Demand events use the L2 LOAD
address. Fill events use the **evicted address** from the actual raw CACHE
L2-fill callback. Both raw fill call sites, including demand/writeback fills,
advance the student's global recurrent state. A fill event emits no action.

For a demand, the learned hurdle gate chooses emit/silent; the positive count
uses the original exponentiation and rounding rule. The autoregressive GRU
starts from the encoded context. Each slot chooses the first argmax mixture
logit and its selected mean, applies signed `expm1` and ties-even rounding,
then wraps the line address modulo `2^58`. Its learned fill head chooses
`FILL_L2=2` or `FILL_LLC=4`. The unrounded selected coordinate and fill choice
feed the next GRU step. Runtime guards reject invalid count/delta domains;
they do not silently clip actions into another model.

The new formal integration is MAC0 synchronous inference: each callback runs
the complete forward and publishes h/c before the next callback. It adds no
input FIFO, admission loss, output queue, per-PC state capacity, finite
inference latency, or decoder32 truncation to this global student. That is a
functional live-inference result; finite hardware service is not established.
The Stride arm retains its separate older per-PC engine and limits.

## Native SPP feedback is different

The original training logger observes **all raw L2-fill callbacks**. The
original native registry invokes SPP FILTER feedback only for prefetch fills.
The formal adapter preserves both contracts: the student consumes all raw
fill events, and native SPP receives `cache_fill` only when `prefetch==1`.
They share actual callback opportunities, but their effective state-update
inputs differ. A native/student IPC difference cannot be attributed solely to
learned payload under an assertion of identical effective inputs.

Native confidence-based L2/LLC fill destinations and its internal same-page
rule remain intact. The new driver explicitly sets request threshold40 and
L2-fill threshold90 to match the original teacher-label configuration. The
first formal study used the dependency's request-threshold25 default; that
old native SPP IPC must not be pooled as a new same-protocol reference. The student retains its learned fill output and complete
action list. Any common external page/request control is a separate experiment
intervention. In particular, an external `cross_page` setting does not remove
native SPP's intrinsic same-page rule. L2-target and LLC-target lifecycle
cohorts must be reported separately; see
[measurement scope](../602_four_prefetcher_formal/MEASUREMENT_SCOPE.md).

## Data coordinates and correctness

The old training prefix is records0..20M. The existing guard20..25M becomes
this study's budget/template validation split, with cache warmup15..20M.
The prior25..50M evaluation was already inspected: ROI0 is a replication ROI,
not a new unseen result. ROI1 measures75..100M after cache warmup70..75M and is
separate from the original train/guard/evaluation coordinates. Coordinates
are nominal trace records; retain actual retirement overshoot and the saved
trace origin. No test output is used to select a cap or template.

The parity script imports the real PyTorch source model and compares every
h/c vector, count, ordered address and fill action for a chronological causal
fixture, followed by explicit uint64 boundary inputs. Fill callbacks are
included. Exact action equality and maximum h/c error below `2e-5` are
required. This correctness fixture supplies no IPC or shortened formal ROI.
Exported models belong under the current study's `raw_v2/models`; the actual
fresh parity fixtures/reports and model/probe/C++ byte snapshots belong in
unique `raw_v2/certifications/<id>/` directories. Each probe is compiled from the
actual simulator runtime with the same `-O3 -ffp-contract=off` flags; each
certificate reruns the real checkpoint fixture. SPP re-export occurs only in
that certificate directory and must match the formal model bytes exactly.
The build-bound accepted gate is `raw_v2/correctness_gate.json`. Preserve these
alongside the actual binary and source evidence.

Use the unified [formal run guide](../602_four_prefetcher_formal/README.md)
for `build`, `certify`, `validation`, `select`, `test`, and `status`. Its
certificate gate must pass before validation/test runs. All new sources,
models, compiler fixtures and temporary files stay inside local/server cache;
reuse the existing Sacramento authentication. No additional seed, workload,
threshold sweep or online update is implied by exporting this one checkpoint.
