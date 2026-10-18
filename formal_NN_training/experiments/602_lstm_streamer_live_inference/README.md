# Frozen Streamer: checkpoint to live inference

- **Current H8 IPC vs native (ROI0 / ROI1):** uncapped **+1.2877% / -1.7337%**; cap 16,144 requests/M **+0.8144% / +0.5590%**.
- **H16 IPC vs native (ROI0 / ROI1):** uncapped **+1.3534% / +1.2852%**; cap 16,144 requests/M **+0.8039% / +0.4842%**. Full matched live results, original seed7 weights; [measurements](../../../formal_study/results/602_width_templates/README.md).
- **H8 capped L2-intent quality (native → NN; ROI0 / ROI1):** accuracy 45.24% → 47.35% / 48.20% → 50.07%; useful/NoPF L2-miss ratio 89.85% → 94.06% / 55.70% → 57.84%; timeliness 99.3662% → 99.1183% / 98.6170% → 98.2223%. The useful/miss ratio is a coverage proxy, not causal miss reduction.
- **H8 capped traffic:** both arms have the same measured allowed requests/M; NN actual DRAM bytes change by +0.0437% / +0.7305% versus native. Requests and DRAM are separate measurements.
- **Scope:** 602, retained seed7, frozen H8/H16 weights, global chronological h/c, synchronous MAC0. These comparisons do not establish finite-resource benefit or cross-seed/workload robustness.
- **Read the measurements:** [native-versus-NN results](../../../formal_study/results/602_width_templates/README.md) · [per-arm metrics](../../../formal_study/results/602_width_templates/reader_metrics.csv) · [paired comparisons](../../../formal_study/results/602_width_templates/reader_pairs.csv) · [own latest subset](results/width_templates_20261017/README.md) · [H8 archive](results/formal_20261017/README.md).

- **Run/recovery:** [four-route guide](../602_four_prefetcher_formal/README.md), [latest supplement](../602_four_prefetcher_formal/width_templates/README.md).

## Retained model

| Item | H8 | H16 |
| --- | --- | --- |
| Checkpoint | [original_checkpoint.pt](models/original_checkpoint.pt) | [original_checkpoint.pt](models/h16/original_checkpoint.pt) |
| Training provenance | [metadata](models/source_training_metadata.json) | [metadata](models/h16/source_training_metadata.json), [8-epoch history](models/h16/training_history.csv), [recipe/tensor verification](models/h16/checkpoint_verification.json) |
| Portable tensors | [model.bin](models/model.bin), 11,798 file bytes | [model.bin](models/h16/model.bin), 27,126 file bytes |
| FP32 parameters / bytes | 2,821 / 11,284 B | 6,653 / 26,612 B |
| Global h/c payload | 64 B | 128 B |
| Model | Unprojected 64-input LSTM, 14 tensors | Same architecture, H16 |

- Original checkpoint, relative to cache: `formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_output/{h8,h16}/model.pt`; metadata is adjacent `run_metadata.json`.
- Legacy checkpoint seed 7 is proven by external run metadata; export requires explicit `--seed 7 --allow-legacy-metadata`.
- Old 545-parameter H8 sweep/v 2 models are different. Other widths are not additional seeds.
- Both widths use the same seed7 0–20M training stream and eight epochs; width is not an additional training seed. [H16 correctness certificate](models/h16/parity_report.json) compares every h/c/count/ordered action on 8,236 callbacks, including reset and numeric boundaries.
- H8 formal copies are under `formal_study/archive/20261017/raw_v2/models/streamer/{model.bin,immutable_model.bin}`. Retained bytes can restore missing copies; preserve any conflicting file.

## Exact input, state and decoder contract

- Each eligible LOAD supplies 64 LSB-first bits of the line-aligned **byte address** (`line << 6`). No PC, teacher output, private table, future event or per-PC state is added.
- All chronological inputs share h/c. Per-PC h/c would change this model.
- Count is `floor(float64 softplus(FP32 count affine))`; zero is valid. Decoder initialization is `tanh(initial_state(h))`.
- Each slot selects the first argmax of four mixture logits and its corresponding signed-log-delta mean. Signed `expm1` plus nearest/ties-even rounding yields a delta relative to the **original** base line; addresses wrap modulo `2^58`, with the original `2^57` magnitude guard.
- The GRU consumes its own **unrounded** selected coordinate after every slot, including the final slot. No learned fill head exists; requests target `FILL_L2=2`.
- The backend evaluates mixture-scale outputs and counts that work even though scales do not select mode actions.
- [Shared scalar backend](../../common/live602_runtime/): `InferState` computes without publishing h/c; `Infer` publishes a completed result.
- Formal global inference is synchronous MAC0, with complete actions and no imposed FIFO, admission drops, per-PC capacity, output queue or decoder 32 truncation. Common page/request gates are separate controls.

## Training versus live history

| Evidence | Coordinates and state |
| --- | --- |
| Weight training | 0–20M; 325,389 LOAD rows; 8 epochs |
| Old offline evaluation | 25–50M; 404,219 rows; independently zero-initialized, without guard initialization |
| New validation | 20–25M, after 15–20M cache/state warmup |
| Live test | 25–50M replication or 75–100M future ROI, each after its own 5M cache/state warmup |

- The old 5–10M Stride validation overlaps Streamer weight training and is not held-out validation here.
- Live h/c starts at zero at cache warmup start, advances while prefetch output is blocked, and is retained at ROI start. ROI callback/update counters restart; training exposure is not live observation history.
- [Evidence alignment](../602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) separates the old offline state history from new live results.

## Export and correctness recovery

- [Exporter](python/export_live_model.py): `--checkpoint`, `--run-metadata`, `--seed`, `--allow-legacy-metadata`, `--out-dir`, optional `--runtime-dir`. It checks identity, seed, shape, finite FP32 tensors and exact LV602R1 roundtrip.
- [Reference](python/streamer_reference.py) and [parity script](validation/compare_python_cpp.py): actual PyTorch checkpoint versus the compiled C++ probe, every h/c/count/ordered action, sequential real inputs, reset and address/numeric boundaries.
- Use the existing Sacramento Torch environment `/home/qianruw/venvs/cache-few-nn-cpu/bin/python`; keep temporary/output files inside cache and use fresh output directories.
- Export/parity runtime and output paths must resolve inside the authorized cache root; canonical `formal_study/runs/...` and route `models/h16/` paths work. Outside paths are rejected. The default runtime is `.runtime/streamer_live`; the removed dated alias is not needed.
- Full H16 correctness evidence is under `formal_study/runs/602_width_templates_20261017/stride_streamer_h16/correctness/streamer/` on both machines. Its fresh probe uses the retained actual formal-build runtime source and `-O3 -ffp-contract=off`.
- The [formal recovery guide](../602_four_prefetcher_formal/README.md) retains model restoration and stage commands. Model/parity artifacts alone do not establish full-ROI IPC; the original trace, matched ChampSim/libbf, machine and build-bound certificate are also required.
