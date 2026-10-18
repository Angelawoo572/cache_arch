# Frozen AMPM: checkpoint to live inference

- **Current H8 IPC vs native (ROI0 / ROI1):** uncapped **-0.3962% / -0.7916%**; cap 16,144 requests/M **+1.7481% / +1.3399%**.
- **Capped L2-intent quality (native → NN; ROI0 / ROI1):** accuracy 39.23% → 44.06% / 41.81% → 45.97%; useful/NoPF L2-miss ratio 77.92% → 87.52% / 48.27% → 53.12%; timeliness 99.7252% → 100.0000% / 99.0009% → 99.9984%. The useful/miss ratio is a coverage proxy, not causal miss reduction.
- **Capped traffic:** both arms nearly exhaust the ceiling; NN actual DRAM bytes change by +2.9850% / +4.3117% versus native. Requests and DRAM are separate measurements.
- **Scope:** 602, retained seed7, frozen H8 weights, global chronological h/c, synchronous MAC0. These comparisons do not establish finite-resource benefit or cross-seed/workload robustness.
- **Read the measurements:** [native-versus-NN results](../../../formal_study/results/602_width_templates/README.md) · [per-arm metrics](../../../formal_study/results/602_width_templates/reader_metrics.csv) · [paired comparisons](../../../formal_study/results/602_width_templates/reader_pairs.csv) · [own latest subset](results/width_templates_20261017/README.md) · [H8 archive](results/formal_20261017/README.md).

- **Run/recovery:** [four-route guide](../602_four_prefetcher_formal/README.md), [H16/template guide](../602_four_prefetcher_formal/width_templates/README.md).

## Retained models

| Item | H8 | H16 |
| --- | --- | --- |
| Original checkpoint | [model.pt copy](models/original_checkpoint.pt), 16,209 file bytes | [model.pt copy](models/h16/original_checkpoint.pt), 31,377 file bytes |
| Source metadata | [metadata](models/source_training_metadata.json) | [metadata](models/h16/source_training_metadata.json) |
| Portable export | [model.bin](models/model.bin), 11,798 file bytes | [model.bin](models/h16/model.bin) |
| FP32 parameters / weight bytes | 2,821 / 11,284 B | 6,653 / 26,612 B |
| Global FP32 h/c payload | 64 B | 128 B |

- Source, relative to cache: `formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/colab_output/{h8,h16}/model.pt`; adjacent `run_metadata.json` supplies original provenance.
- Identity: AMPM `source_input_variable_delta_free_running_v7`, 14 tensors, unprojected `nn.LSTM(64,H)`. H16 is the real retained v 7 checkpoint, not the older 5,713-parameter sweep or an H8 fallback.
- H8 frozen formal copies are under `formal_study/archive/20261017/raw_v2/models/ampm/`. H16 has separate exports and byte bindings in the supplement.

## Exact input, state and decoder contract

- Input is 64 LSB-first bits of the aligned **byte address** (`line << 6`). There is no PC, hit/miss, cycle, teacher action, private table or future-event input.
- One h/c pair carries all chronological inputs. Training carries state across TBPTT chunks, detaches at chunk boundaries and resets each epoch.
- Count is the Poisson mode `floor(float64 softplus(FP32 count affine))`, including zero. Each event restarts the decoder with `tanh(initial_state(h))`.
- The four-component mixture computes all 12 logit/mean/scale outputs. First-argmax logit selects its mean; signed `expm1` and nearest/ties-even rounding produce a delta from the original base line, with the `2^57` magnitude guard and modulo-`2^58` line wrapping.
- Every GRU step, including the final one, consumes the model's own **unrounded** coordinate. There is no fill head; outputs target `FILL_L2=2`.
- [Shared scalar backend](../../common/live602_runtime/): `InferState` computes without publishing h/c; `Infer` publishes completed h/c. It counts scale-head work even though scales do not select mode actions.
- Global formal inference is synchronous MAC0, without an imposed FIFO, per-PC state capacity, output queue or decoder 32 truncation. External page/request controls remain explicit.

## Training and evaluation scope

- Weight training: 0–20M, 325,389 LOAD rows, 8 epochs; chunk length 1,024 and 32 chunks per optimizer step. The source default learning rate is 0.002; do not treat that default alone as proof of a saved run's optimizer setting.
- Old offline inference: fresh h/c at guard start 20M → guard 20–25M (80,725 rows) → evaluation 25–50M (404,219 rows).
- New live inference: fresh h/c at the start of each 5M cache warmup, warmup output blocked, actual h/c retained into the ROI. Validation 20–25M; tests 25–50M (replication) and 75–100M (future ROI).
- Old 5–10M Stride validation overlaps AMPM training. Source training amount, live warmup and ROI callback/update axes are different quantities.
- Old `matched_comparison` is keyed replay; its native AMPM context counters include warmup. It does not establish live/finite-cost behavior or supply same-protocol IPC here. See [evidence alignment](../602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md).

## Export, parity and recovery

- [H8 exporter](python/export_model.py): `--checkpoint`, `--metadata`, `--output`; restricted plain-tensor archive reader, no local Torch required, exact FP32 roundtrip and identity checks. Its main entrypoint is H8-specific.
- **H16 uses [ampm_h16.py](../602_four_prefetcher_formal/width_templates/ampm_h16.py)**, which checks H16/6,653 parameters and the original revision. Do not use the H8 exporter main for H16.
- [H8 parity](validation/validate_parity.py) compares the actual PyTorch model with C++ across supplied ordered streams, every h/c/count/action and reset. H16 has its own parity gate in the supplement.
- Use the existing Sacramento Torch environment for parity. Keep all compiler/temporary/cache files under the configured study runtime inside cache.
- Restore missing model/metadata files from retained originals; compare bytes and preserve conflicts. Reuse a compatible accepted certificate or follow the [formal stage commands](../602_four_prefetcher_formal/README.md); exporting weights alone is not a completed formal experiment.
- Payload bytes exclude scratch, controller and output storage. Host object/heap evidence is neither complete hardware deployment storage nor RSS/PPA; [cost boundaries](../602_four_prefetcher_formal/report/COST.md).
