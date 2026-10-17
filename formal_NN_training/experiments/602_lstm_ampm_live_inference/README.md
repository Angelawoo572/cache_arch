# Real AMPM student: offline checkpoint to live inference

## Completed width/template supplement

The [new complete results](results/width_templates_20261017/README.md) preserve this route's own frozen model and explicitly labeled compatible references. See the [three-question report](../../../docs/602_width_templates/REPORT.md) and [new run/recovery guide](../602_four_prefetcher_formal/width_templates/README.md). The original seed7 [H16 checkpoint](models/h16/original_checkpoint.pt), [portable export](models/h16/model.bin) and [training metadata](models/h16/source_training_metadata.json) are retained independently of H8.


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
`/home/qianruw/cache/formal_study/archive/20261017/raw_v2/models/ampm/model.bin`,
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

This sibling preserves the actual `602_offline_lstm_ampm` v7 model. It does not use the Stride Hurdle checkpoint, PC input, per-PC h/c table, Stride decoder, replay table, or the later canonical h/c initialization apparatus.

The retained seed 7 h8 checkpoint is `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/colab_output/h8/model.pt` under the cache root. Its 14 FP32 tensors contain 2,821 parameters (11,284 bytes). `train_and_offline_infer.py` calls `formal_NN_training.common.direct_action_lstm.run_cli("ampm")`; the real class is `threshold_free_policy.VariableActionLSTM`, with a direct `nn.LSTM(64,8)`, no input projection, and one global chronological h/c pair. The decoder restarts per event from `tanh(initial_state(h))`.

Input means 64 LSB-first bits of the aligned **byte address**, `line << 6`, with six naturally zero low bits. It excludes PC, hit/miss, cycle, future events, teacher actions and teacher private tables. Request count is the Poisson mode `floor(float64 softplus(count_head(h)))`, permitting zero. The decoder computes all 12 outputs of a four-component mixture head; it selects the first argmax mixture logit and its corresponding mean. The signed log-delta coordinate becomes `sign * round(expm1(abs(coordinate)))`, with ties-even rounding and the original `2**57` magnitude guard, then wraps in the 58-bit aligned line domain. Every decoder GRU step, including the final one, consumes the model's unrounded coordinate. There is no learned fill head; the adapter maps the default output to L2.

The saved model was trained on instruction records **[0,20M]**, with 325,389 eligible rows. Its original independent guard is **[20M,25M]**, 80,725 rows; its old evaluation is **[25M,50M]**, 404,219 rows. Training carries global h/c chronologically across TBPTT chunks, detaches at chunk boundaries and resets at epoch start; teacher count/actions supply loss only. The original sources default to 8 epochs, chunk 1,024, 32 chunks per optimizer step and learning rate 0.002. The actual saved seed/h8 identity and metadata are checked during export. The new formal validation must use [20M,25M], outside those old training coordinates; the earlier [5M,10M] Stride validation would overlap this AMPM model's training.

Old `matched_comparison` results are keyed list replay. They are retained as historical replay results and do not establish live recurrent or finite-cost behavior. Their live native AMPM context row also includes warmup lifecycle counters and cannot be silently mixed with the newer unified measurement protocol. No old IPC values are imported as new formal rows.

The shared scalar implementation is in `formal_NN_training/common/live602_runtime/`. Its `LV602R1` binary stores the exact FP32 tensors with explicit count/decoder/feature/fill schemas. `python/export_model.py` reads only the known plain tensor checkpoint structures, verifies the real AMPM identity and exact binary tensor roundtrip, and records original training coordinates in `models/model_metadata.json`. It needs no local Torch. The full ordered-sequence `validation/validate_parity.py` uses the actual PyTorch class to compare every h/c vector, count and action against the C++ probe, including Reset; no simulation performance pilot is involved.

`Live602Runtime::InferState` computes without publishing global h/c, while `Infer` publishes the completed result. The current formal integration uses synchronous zero-cost live inference, with no serial queue, per-PC state capacity, or decoder32 limit. The runtime emits the complete learned action list and reports `truncated=false`. Common page/request-budget gates, if used by the formal adapter, are explicit external controls and are not checkpoint features. Finite-cost scheduling is a separate integration and must not be claimed from these zero-cost rows.

All compilation and temporary files belong under `cache/formal_study_20261017/runtime_v2/`, with `TMPDIR`, `PYTHONPYCACHEPREFIX` and `MPLCONFIGDIR` set there. Use the server's existing Torch environment for parity; no Stride source files need to be changed.
