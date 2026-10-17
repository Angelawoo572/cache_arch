# Frozen Streamer student: real checkpoint to live inference

## Completed width/template supplement

The [new complete results](results/width_templates_20261017/README.md) preserve this route's own frozen model and explicitly labeled compatible references. See the [three-question report](../../../docs/602_width_templates/REPORT.md) and [new run/recovery guide](../602_four_prefetcher_formal/width_templates/README.md).


## Retained real weights and current formal output

This sibling carries the real small [original checkpoint](models/original_checkpoint.pt)
(16,209 bytes), byte-identical [source training metadata](models/source_training_metadata.json),
and [portable FP32 model](models/model.bin) (11,798 file bytes). The
portable model contains 2,821 parameters, or 11,284 resident FP32
weight bytes; file headers and tensor names account for the larger file size.
These are trained weights, not placeholders, a random reinitialization, a trace,
or a replay list. The retained checkpoint and metadata match the original
source run byte for byte; the portable binary matches the previously verified
export byte for byte. All copies are in cache.

The current formal destination is
`/home/qianruw/cache/formal_study/archive/20261017/raw_v2/models/streamer/model.bin`,
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

This sibling exports the actual address-only Streamer v7 student, not a Stride
checkpoint or a replay lookup table. The shared C++ backend is
`formal_NN_training/common/live602_runtime`; its source and ABI are owned
separately. No shared source, branch or commit is changed by these scripts.

The main existing checkpoint is under the original cache root:
`formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_output/h8/model.pt`,
with adjacent `run_metadata.json`. It has 2,821 finite FP32 parameters
(11,284 resident weight bytes), 64 input features, H=8, a global 64-byte h/c,
and 14 tensors. Its checkpoint declares policy=streamer and revision
source_input_variable_delta_free_running_v7. The seed is proven only by its
original external run metadata; exporting this legacy file requires the
explicit `--allow-legacy-metadata` flag. The old 545-parameter h8 sweep/v2
models are different models. Sizes h16/32/64/128 exist but are not a sweep
requested in this study, and are not additional training seeds.

The exact input is the LSB-first 64-bit encoding of the line-aligned byte
address (line << 6), with no PC. All eligible inputs in the chronological
stream share one h/c. Using per-PC states would change the trained model.
The source has an unprojected LSTM, Poisson mode floor(softplus(count)), a
tanh decoder initial state, and a four-component signed-log-delta mixture.
At every slot, choose the first argmax mixture logit, transform its mean with
signed expm1 and nearest/ties-even integer rounding, then add to the original
base line modulo 2^58. The unrounded selected mean is fed into the GRU for
the next action. The model has no learned fill head: its interface requests
L2. A formal common page/request gate is outside these model semantics and
must be reported separately.

Training was the chronological 0..20M instruction prefix (325,389 eligible
rows). The old evaluation was 25..50M (404,219 rows), independently zero
initialized and already inspected. It is a replication ROI, not a newly
unseen evaluation. The future75..100M ROI is distinct from training and the
legacy evaluation. The prior Stride validation5..10M overlaps this training;
it must not be reused as held-out validation for Streamer. Any proposed
20..25M budget validation must be recorded by the parent protocol, including
its overlap with cache warmup. This exporter does not select a budget, train
weights, or choose a model using test outcomes.

Run through the existing Sacramento Torch environment. All new artifacts go
under /home/qianruw/cache/formal_study/archive/20261017/; set TMPDIR and Python cache
paths inside its runtime directory. Local sources likewise remain inside
the authorized cache directory; no /tmp output is created.

```sh
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/qianruw/cache/formal_study/archive/20261017/runtime_v2/tmp \
/home/qianruw/venvs/cache-few-nn-cpu/bin/python \
  python/export_live_model.py \
  --checkpoint /home/qianruw/cache/formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_output/h8/model.pt \
  --run-metadata /home/qianruw/cache/formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_output/h8/run_metadata.json \
  --seed 7 --allow-legacy-metadata \
  --out-dir /home/qianruw/cache/formal_study/archive/20261017/reproduction/exports/streamer
```

The export checks original policy, source revision, seed metadata, input
contract, tensor shape/dtype/finite values and exact FP32 roundtrip through
LV602R1. It preserves all requested model actions without a hidden degree
cap. The backend operation counters, finite-service costs, actual output
limits, callback admission and queue policy are separate from the literal
network inventory. Model storage is not whole-process RSS or measured PPA.
The training-only mixture-scale outputs do not select mode actions; any
backend elimination of those unused computations must be stated explicitly.
