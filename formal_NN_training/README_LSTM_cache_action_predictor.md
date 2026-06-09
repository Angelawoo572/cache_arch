# Outcome-aware SPP-assisted LSTM Cache Action Predictor

This folder implements the end-to-end flow for the current cache-prefetch learning experiment:

```text
trace + ChampSim
  -> instrumented SPP candidate/context/outcome events
  -> outcome-aware LSTM training table
  -> multi-task LSTM cache-action learner
  -> action CSV / list_replayer prefetch list
  -> ChampSim replay
  -> offline + system-level metrics
```

## Current framing

SPP is **not** treated as merely a yes/no filter and the LSTM is **not** a direct next-address predictor in the final interpretation.

At this stage SPP provides:

1. candidate address / delta,
2. context features such as SPP confidence, hit/miss state, MSHR pressure, L2 occupancy, and bandwidth pressure,
3. supervision signals such as `outcome_useful` and `outcome_duplicate`.

The LSTM learns cache actions:

```text
good-prefetch probability
bypass / low-priority probability
timing / urgency bucket
auxiliary candidate-delta/address head
```

The key label is outcome-aware:

```text
good_prefetch = outcome_useful == 1 AND outcome_duplicate == 0
```

The older next-demand-line formulation remains useful as a diagnostic, but it should not be reported as useful-prefetch precision.

## Important result checkpoint: 602.gcc_s-734B

The first valid positive system-level result is on `602.gcc_s-734B` with 25M warmup / 25M simulation.

Earlier replay results were invalid because the replay pipeline had two bugs:

1. the prefetch list used `event_id` / `cycle` instead of the L2 replay demand-access index;
2. the prefetch address was written in decimal, while `list_replayer` parses the second column as hexadecimal.

After fixing both:

```text
prefetch list format:
  replay_access_idx  0xprefetch_byte_addr
```

The fixed LSTM replay produces real useful L2 prefetches and improves IPC over no-prefetch.

### System-level IPC comparison

| Method | IPC | Speedup vs no-prefetch | Notes |
|---|---:|---:|---|
| no-prefetch | 0.5427 | 1.0000x | baseline |
| SPP baseline | 1.4440 | 2.6608x | strong hardware baseline |
| LSTM fixed replay, threshold 0.20 | 0.7175 | 1.3221x | best observed LSTM point so far |
| LSTM fixed replay, threshold 0.25 | 0.7172 | 1.3215x | essentially tied with 0.20 |
| LSTM fixed replay, threshold 0.35 | 0.6371 | 1.1739x | fewer useful prefetches |
| LSTM fixed replay, threshold 0.40 | 0.5521 | 1.0173x | high threshold, too conservative |

Current conclusion:

```text
On 602.gcc_s-734B, fixed LSTM replay reaches about +32% IPC over no-prefetch.
SPP is still much stronger in total IPC, but the LSTM result is now valid and positive.
```

### Accuracy comparison: SPP vs LSTM

Accuracy must be reported carefully because there are two different meanings.

#### 1. Offline candidate-selection accuracy

This asks:

```text
Among SPP candidate rows in the training/evaluation table,
how well does the model select good candidates?
```

For this offline candidate-classification view, the LSTM is much better than raw SPP candidate emission.

Observed on `602.gcc_s-734B`:

| Policy / view | Good precision | Good recall | Duplicate behavior | Interpretation |
|---|---:|---:|---|---|
| Raw SPP candidate stream | about 1.96% good-candidate precision | 100% candidate coverage | very high duplicate rate, about 97% | SPP exposes many candidates, most are not good under this label |
| LSTM action selection | about 54.5% good precision | about 61.0% recall | duplicate rate reduced to about 39.6% | LSTM is much better as an offline candidate selector |
| LSTM threshold 0.20 | about 51.9% good precision | about 88.9% recall | high-recall setting | good for system-level replay on gcc |
| LSTM threshold 0.40 | about 70.9% good precision | about 16.9% recall | high-precision / low-recall setting | too conservative for IPC |

So under **offline candidate-selection accuracy**, the LSTM is better.

#### 2. ChampSim prefetch accuracy

This asks:

```text
Among prefetches actually issued into ChampSim,
how many become useful before they become useless?
```

Using ChampSim's printed L2C counters and the common useful ratio

```text
prefetch_accuracy = USEFUL / (USEFUL + USELESS)
```

we get:

| Method | L2C requested | L2C useful | L2C useless | Approx. ChampSim prefetch accuracy | IPC |
|---|---:|---:|---:|---:|---:|
| SPP baseline | 3,006,672 | 140,717 | 652 | about 99.5% | 1.4440 |
| LSTM fixed replay, threshold 0.20 | 112,446 | 64,473 | 48,530 | about 57.0% | 0.7175 |
| LSTM fixed replay, threshold 0.25 | 112,388 | 64,420 | 48,533 | about 57.0% | 0.7172 |
| LSTM fixed replay, threshold 0.35 | 84,776 | 16,348 | 70,171 | about 18.9% | 0.6371 |
| LSTM fixed replay, threshold 0.40 | 7,545 | 3,933 | 4,142 | about 48.7% | 0.5521 |

So under **true ChampSim prefetch accuracy**, SPP is better than the current LSTM replay on `602.gcc_s-734B`.

The important nuance is:

```text
Offline candidate-selection accuracy:
  LSTM > raw SPP candidate stream

ChampSim prefetch accuracy + IPC:
  SPP > current LSTM replay

LSTM current value:
  proves the learned action model can improve over no-prefetch once replay is fixed,
  but it has not yet beaten SPP.
```

### Latency / practicality comparison

| Method | Critical-path practicality | Runtime / latency interpretation |
|---|---|---|
| SPP | Hardware table-based prefetcher; practical as a baseline | Low-latency online decision logic; already integrated in ChampSim |
| LSTM replay | Offline precomputed replay list | Current IPC result does **not** include neural inference latency |
| Real online LSTM | Would require model inference on or near the memory pipeline | Too expensive unless distilled into a tiny gate/table/fixed-point model |

Therefore:

```text
Accuracy-only view:
  depends on the definition.
  LSTM wins offline candidate-selection precision.
  SPP wins real ChampSim prefetch accuracy.

Latency view:
  SPP is clearly better today.
  Current LSTM replay is an offline upper-bound / validation path, not yet a deployable online hardware design.
```

## Main files

```text
formal_NN_training/LSTM_cache_action_predictor.ipynb
formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
formal_NN_training/scripts/01_run_spp_trace_dump.sh
formal_NN_training/scripts/train_lstm_cache_action.py
formal_NN_training/scripts/02_actions_to_prefetch_list.py
formal_NN_training/scripts/03_run_lstm_replay.sh
formal_NN_training/scripts/04_eval_lstm_accuracy.py
formal_NN_training/scripts/05_eval_current_label_lstm_vs_spp.py
```

## Default first trace

Use `602.gcc_s-734B` first because it is the most general initial trace in this project: mixed / phase behavior, not purely streaming and not purely pointer-chasing.

Then repeat on:

```text
619.lbm_s-4268B      streaming
605.mcf_s-994B      pointer-chase / irregular
620.omnetpp_s-874B  indirect / graph-like
```

## End-to-end runbook from repo root

### 0. Pull latest code

```bash
cd /scratch/qianruw/cache
git pull --ff-only
```

### 1. Generate SPP candidate/outcome data on cluster

```bash
TRACE=602.gcc_s-734B \
WARMUP=25000000 \
SIM=25000000 \
RESET_SPP=1 \
BUILD=1 \
PATCH_SPP=1 \
bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
```

Expected outputs:

```text
formal_NN_training/results/spp_trace_dump/events/spp_events_602.gcc_s-734B.csv
formal_NN_training/results/spp_trace_dump/candidate_table_602.gcc_s-734B.csv
formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv
```

Important replay-alignment fields:

```text
lstm_events_*.csv:
  trace,event_id,replay_access_idx,cycle,...,pf_addr,delta,...

full_lstm_cache_actions.csv:
  trace,event_id,replay_access_idx,cycle_num,...,prefetch_addr,...

prefetch_list_*.txt:
  replay_access_idx 0xprefetch_addr
```

### 2. Train outcome-aware LSTM

Fast smoke run:

```bash
python3 formal_NN_training/scripts/train_lstm_cache_action.py \
  --trace 602.gcc_s-734B \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --max-rows 500000 \
  --seq-len 32 \
  --epochs 2 \
  --batch-size 128 \
  --hidden-dim 96 \
  --emb-dim 32
```

Main run:

```bash
python3 formal_NN_training/scripts/train_lstm_cache_action.py \
  --trace 602.gcc_s-734B \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --max-rows 2000000 \
  --seq-len 64 \
  --epochs 8 \
  --batch-size 256 \
  --hidden-dim 128 \
  --emb-dim 32 \
  --good-threshold 0.50 \
  --bypass-threshold 0.60
```

Full-data run uses `--max-rows 0`.

Expected outputs:

```text
formal_NN_training/artifacts/outcome_lstm_cache_action_predictor.pt
formal_NN_training/artifacts/outcome_lstm_summary.json
formal_NN_training/artifacts/outcome_lstm_training_history.csv
formal_NN_training/artifacts/outcome_lstm_cache_actions.csv
formal_NN_training/artifacts/full_lstm_cache_actions.csv
```

`full_lstm_cache_actions.csv` is kept as the compatibility name expected by the replay script.

### 3. Offline outcome-aware evaluation

```bash
python3 formal_NN_training/scripts/04_eval_lstm_accuracy.py \
  --events formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv \
  --actions formal_NN_training/artifacts/full_lstm_cache_actions.csv \
  --out formal_NN_training/results/lstm_outcome_accuracy_602.gcc_s-734B.csv \
  --policy action \
  --prefetch-threshold 0.50 \
  --bypass-threshold 0.60
```

Primary metrics to read first:

```text
lstm_good_precision
lstm_good_recall
lstm_good_f1
lstm_duplicate_rate
spp_good_precision
spp_duplicate_rate
```

### 4. Convert actions to list_replayer prefetch list

The converter should use `replay_access_idx` first and write addresses as hexadecimal.

```bash
python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
  --actions formal_NN_training/artifacts/full_lstm_cache_actions.csv \
  --out formal_NN_training/results/replay_compare/prefetch_lists/prefetch_list_602_gcc_lstm.txt \
  --policy threshold \
  --prefetch-threshold 0.20 \
  --bypass-threshold 1.00
```

Expected diagnostics:

```text
[idx_col] replay_access_idx
bad_addr=0
```

Expected output format:

```text
40 0x2f6527780
56 0x2f6527840
...
```

### 5. Replay in ChampSim and compare

```bash
TRACE=602.gcc_s-734B \
WARMUP=25000000 \
SIM=25000000 \
POLICY=threshold \
PREFETCH_THRESHOLD=0.20 \
BYPASS_THRESHOLD=1.00 \
REPL_BIN=/scratch/qianruw/cache/external/ChampSim/bin/champsim.l2_replayer \
MODEL_TAG=LSTM_gcc_s-734B_L2_aligned_hex_th0.20_bp1.00 \
bash formal_NN_training/scripts/03_run_lstm_replay.sh
```

Expected summary:

```text
formal_NN_training/results/replay_compare/summary_602.gcc_s-734B.csv
```

Compare:

```text
no_prefetch
spp
LSTM_<trace>_L2_aligned_hex_th<TH>_bp1.00
```

## Colab data movement

To restore split input files in Colab:

```bash
TRACE=602.gcc_s-734B bash formal_NN_training/scripts/00_restore_colab_uploaded_data.sh
```

Expected upload layout:

```text
formal_NN_training/data/upload/602/lstm_events_602.gcc_s-734B.csv.gz.part_000 ...
```

To split large outputs manually:

```bash
mkdir -p formal_NN_training/artifacts/packed/602
cd formal_NN_training/artifacts

gzip -c full_lstm_cache_actions.csv > packed/602/full_lstm_cache_actions.csv.gz
split -b 90m packed/602/full_lstm_cache_actions.csv.gz packed/602/full_lstm_cache_actions.csv.gz.part_
```

## What to report

Do not only report `delta_top1`. Report in this order:

1. offline `good_prefetch` precision / recall / F1,
2. duplicate reduction / duplicate rate,
3. ChampSim useful / useless / prefetch accuracy,
4. cache hit rate / MPKI,
5. bandwidth / PQ / MSHR pressure,
6. IPC as the final system-level check,
7. latency / deployability separately from offline replay IPC.

## Important distinction

Old implemented objective:

```text
predict next demand-line delta / next demand-line address
```

New implemented objective:

```text
learn whether an SPP candidate/action is useful, duplicate, bypass-worthy, or timing-sensitive
using outcome_useful / outcome_duplicate labels
```

The auxiliary delta head remains useful, but it is not the main success criterion.
