# Outcome-aware SPP-assisted LSTM Cache Action Predictor

This folder contains the current SPP-assisted LSTM cache-action experiment:

```text
ChampSim + SPP candidate logging
  -> lstm_events_<TRACE>.csv
  -> outcome-aware LSTM training in Colab/notebook
  -> full_lstm_cache_actions.csv
  -> list_replayer prefetch list
  -> ChampSim replay
  -> SPP / LSTM / no-prefetch comparison
```

For the diagram/story version, see:

```text
formal_NN_training/LSTM_cache_action_pipeline_story.md
```

For script usage, see:

```text
formal_NN_training/scripts/README.md
```

## Current framing

SPP is used as a **candidate generator + feature/context provider + supervision source**. The LSTM is not mainly a direct next-address predictor. It is a sequential cache-action learner that decides whether an SPP candidate is likely useful, duplicate, suppress-worthy, bypass-worthy, or timing-sensitive.

The key outcome-aware label is:

```text
good_prefetch = outcome_useful == 1 AND outcome_duplicate == 0
```

The older next-demand-line / next-delta formulation remains useful as a diagnostic, but it should not be reported as useful-prefetch precision.

## Replay correctness rule

Earlier replay results were invalid when the prefetch list used `event_id` / `cycle` instead of the L2 replay demand-access index, or when prefetch addresses were written in decimal. The valid list format is:

```text
replay_access_idx  0xprefetch_byte_addr
```

During conversion, this line must appear:

```text
[idx_col] replay_access_idx
```

If the converter prints `[idx_col] event_id`, that replay is invalid.

## Current result summary

All results below use 25M warmup / 25M simulation.

### System-level IPC

| Trace | Method | IPC | Speedup vs no-prefetch | Interpretation |
|---|---|---:|---:|---|
| 602.gcc_s-734B | no-prefetch | 0.5427 | 1.0000x | baseline |
| 602.gcc_s-734B | SPP | 1.4440 | 2.6608x | highest IPC on 602 |
| 602.gcc_s-734B | LSTM th0.20 | 0.7175 | 1.3221x | valid positive replay, but below SPP |
| 619.lbm_s-4268B | no-prefetch | 0.4345 | 1.0000x | baseline |
| 619.lbm_s-4268B | SPP | 0.5077 | 1.1685x | highest IPC on 619 |
| 619.lbm_s-4268B | LSTM replayidx th0.10-th0.35 | 0.4568 | 1.0513x | valid positive replay, but below SPP |

Current IPC conclusion:

```text
LSTM beats no-prefetch on both 602 and 619.
SPP still beats LSTM in final IPC on both traces.
```

## Accuracy / precision comparison: SPP vs LSTM

Do not collapse everything into one word, `accuracy`. Report these separately:

```text
issued-prefetch precision = USEFUL / ISSUED
coverage                  = total USEFUL
performance               = IPC
useful-vs-evicted ratio   = USEFUL / (USEFUL + USELESS)
```

`USEFUL / (USEFUL + USELESS)` is **not** the same as `USEFUL / ISSUED`, because ChampSim's `USELESS` counter is not all non-useful issued prefetches.

### 602.gcc_s-734B replay metrics

| Method | IPC | Requested | Issued | Useful | Useless | Useful / Issued | Useful / Requested | Useful / (Useful + Useless) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM th0.20 | 0.7175 | 112,446 | 112,446 | 64,473 | 48,530 | **57.3369%** | 57.3369% | 57.0542% |
| SPP | 1.4440 | 3,006,672 | 2,739,031 | 140,717 | 652 | **5.1375%** | 4.6802% | 99.5388% |

602 conclusion:

```text
LSTM is much more precise per issued prefetch: 57.34% vs SPP 5.14%.
SPP still wins coverage and IPC: 140,717 useful prefetches and 1.4440 IPC.
```

### 619.lbm_s-4268B replay metrics

For 619, LSTM thresholds 0.10, 0.15, 0.20, 0.25, 0.30, and 0.35 selected the same prefetch set and produced the same replay metrics.

| Method | IPC | Requested | Issued | Useful | Useless | Useful / Issued | Useful / Requested | Useful / (Useful + Useless) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM replayidx th0.10-th0.35 | 0.4568 | 166,217 | 166,217 | 157,564 | 5,946 | **94.7942%** | 94.7942% | 96.3635% |
| SPP | 0.5077 | 4,534,355 | 2,587,585 | 88,474 | 0 | **3.4192%** | 1.9512% | 100.0000% |

619 conclusion:

```text
LSTM has extremely high issued-prefetch precision: 94.79% vs SPP 3.42%.
SPP still wins IPC: 0.5077 vs LSTM 0.4568.
```

## Overall conclusion

```text
Across both 602.gcc and 619.lbm, the LSTM replay consistently improves issued-prefetch precision over SPP.

602: LSTM useful/issued = 57.34%, SPP useful/issued = 5.14%.
619: LSTM useful/issued = 94.79%, SPP useful/issued = 3.42%.

However, SPP still achieves higher IPC because it has higher coverage and captures more total useful prefetch opportunities in the current replay setup.
```

Short version for meetings:

```text
LSTM is a cleaner selector. SPP is still the stronger performance baseline.
The next research question is how to keep LSTM-level precision while increasing coverage and eventually distilling the policy into a low-latency hardware-feasible gate.
```

## Latency / practicality comparison

| Method | Critical-path practicality | Runtime / latency interpretation |
|---|---|---|
| SPP | Hardware table-based prefetcher; practical as a baseline | Low-latency online decision logic; already integrated in ChampSim |
| LSTM replay | Offline precomputed replay list | Current IPC result does **not** include neural inference latency |
| Real online LSTM | Would require model inference on or near the memory pipeline | Too expensive unless distilled into a tiny gate/table/fixed-point model |

Therefore:

```text
SPP is clearly better for latency today.
Current LSTM replay is an offline validation path, not yet a deployable online hardware design.
```

## Main files

```text
formal_NN_training/LSTM_cache_action_predictor.ipynb
formal_NN_training/LSTM_cache_action_pipeline_story.md
formal_NN_training/scripts/README.md
formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
formal_NN_training/scripts/01_run_spp_trace_dump.sh
formal_NN_training/scripts/02_actions_to_prefetch_list.py
formal_NN_training/scripts/03_run_lstm_replay.sh
formal_NN_training/scripts/04_eval_lstm_accuracy.py
formal_NN_training/scripts/07_prepare_actions_for_replay.py
formal_NN_training/scripts/08_scout_candidate_traces.sh
formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py
```

## Standard commands

### Generate / convert SPP candidate data

Full run:

```bash
TRACE=619.lbm_s-4268B \
WARMUP=25000000 \
SIM=25000000 \
BUILD=0 \
PATCH_SPP=0 \
bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Convert-only after fixing conversion logic or reusing an existing SPP event log:

```bash
TRACE=619.lbm_s-4268B \
CONVERT_ONLY=1 \
BUILD=0 \
PATCH_SPP=0 \
bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

After this step, verify `replay_access_idx` is nonblank in `lstm_events_<TRACE>.csv`.

### Prepare Colab output for replay

```bash
python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
  --trace 619.lbm_s-4268B \
  --copy-default
```

This restores packed Colab output if needed, validates the trace, merges missing `replay_access_idx`, and copies the prepared actions into:

```text
formal_NN_training/artifacts/full_lstm_cache_actions.csv
```

### Replay

```bash
TRACE=619.lbm_s-4268B \
WARMUP=25000000 \
SIM=25000000 \
POLICY=threshold \
PREFETCH_THRESHOLD=0.20 \
BYPASS_THRESHOLD=1.00 \
REPL_BIN=/scratch/qianruw/cache/external/ChampSim/bin/champsim.l2_replayer \
MODEL_TAG=LSTM_lbm_s-4268B_L2_replayidx_hex_th0.20_bp1.00 \
bash formal_NN_training/scripts/03_run_lstm_replay.sh
```

### Compare SPP vs LSTM replay accuracy

```bash
python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace 619.lbm_s-4268B \
  --include-lstm replayidx \
  --exclude-lstm aligned_hex \
  --out formal_NN_training/results/replay_compare/accuracy_compare_619.lbm_s-4268B.csv
```

For 602:

```bash
python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace 602.gcc_s-734B \
  --include-lstm L2_aligned_hex_th0.20 \
  --out formal_NN_training/results/replay_compare/accuracy_compare_602.gcc_s-734B.csv
```

## Next steps

1. Do not continue low-threshold sweep on 619; 0.10-0.35 produced the same selected set.
2. If sweeping 619 further, try higher thresholds such as 0.50, 0.70, 0.90, and 0.95.
3. For research direction, focus on increasing LSTM coverage while preserving precision, or distilling the learned selector into a low-latency hardware-feasible gate.
4. Repeat the same fixed replay-index flow on `605.mcf_s-994B` or `620.omnetpp_s-874B` for irregular / indirect behavior.
