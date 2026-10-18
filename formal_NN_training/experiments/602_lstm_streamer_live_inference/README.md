# Frozen Streamer: checkpoint to live inference

- **Current H8 IPC vs native (ROI0 / ROI1):** uncapped **+1.2877% / -1.7337%**; cap 16,144 requests/M **+0.8144% / +0.5590%**.
- **H16 availability:** no H16 result exists in this common live protocol. The current Streamer native/NN comparison is H8; do not infer an H16 outcome.
- **H8 capped L2-intent quality (native → NN; ROI0 / ROI1):** accuracy 45.24% → 47.35% / 48.20% → 50.07%; useful/NoPF L2-miss ratio 89.85% → 94.06% / 55.70% → 57.84%; timeliness 99.3662% → 99.1183% / 98.6170% → 98.2223%. The useful/miss ratio is a coverage proxy, not causal miss reduction.
- **H8 capped traffic:** both arms have the same measured allowed requests/M; NN actual DRAM bytes change by +0.0437% / +0.7305% versus native. Requests and DRAM are separate measurements.
- **Scope:** 602, retained seed7, frozen H8 weights, global chronological h/c, synchronous MAC0. These comparisons do not establish finite-resource benefit or cross-seed/workload robustness.
- **Read the measurements:** [native-versus-NN results](../../../formal_study/results/602_width_templates/README.md) · [per-arm metrics](../../../formal_study/results/602_width_templates/reader_metrics.csv) · [paired comparisons](../../../formal_study/results/602_width_templates/reader_pairs.csv) · [own latest subset](results/width_templates_20261017/README.md) · [H8 archive](results/formal_20261017/README.md).

- **Run/recovery:** [four-route guide](../602_four_prefetcher_formal/README.md), [latest supplement](../602_four_prefetcher_formal/width_templates/README.md).

## Retained model

| Item | Exact identity |
| --- | --- |
| Checkpoint | [models/original_checkpoint.pt](models/original_checkpoint.pt), 16,209 file bytes |
| Provenance | [source_training_metadata.json](models/source_training_metadata.json), byte-identical to the source run |
| Portable tensors | [models/model.bin](models/model.bin), 11,798 file bytes; 2,821 FP32 parameters / 11,284 weight bytes |
| Encoder/state | Unprojected LSTM, 64 features, H8, 14 tensors; one global 64-byte FP32 h/c pair |
| Revision | `source_input_variable_delta_free_running_v7`, policy `streamer` |

- Original checkpoint, relative to cache: `formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_output/h8/model.pt`; metadata is adjacent `run_metadata.json`.
- Legacy checkpoint seed 7 is proven by external run metadata; export requires explicit `--seed 7 --allow-legacy-metadata`.
- Old 545-parameter H8 sweep/v 2 models are different. Other widths are not additional seeds.
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
- The standalone legacy Streamer scripts still restrict paths to the removed `formal_study_20261017` name. Do not recreate that alias or assume JSON path translation fixes this Python guard; preserve the accepted certificates and resolve this restriction before a fresh standalone export/parity run.
- The [formal recovery guide](../602_four_prefetcher_formal/README.md) retains model restoration and stage commands. Model/parity artifacts alone do not establish full-ROI IPC; the original trace, matched ChampSim/libbf, machine and build-bound certificate are also required.
