# 623 Stride v25 — dual-context hurdle LSTM

This folder contains the active matched-input experiment for
`623.xalancbmk_s-700B`.

- Run: `623_offline_lstm_stride_dual_context_hurdle_unique_v25_seed7`
- Points: `h8, h16, h32, h64, h128`
- Status: design and code are ready; v25 training/replay results are pending

## What v23 actually established

Stride v23 did **not** train a new architecture. It loaded the byte-identical
v22 checkpoints and changed only decoding from weighted logits to
`logit - log(TRAIN class weight)`. It therefore tested a post-hoc prior
correction, not LSTM capacity.

| Evidence | Observation |
|---|---:|
| normal Stride | 166,147 requests; IPC 0.35340 |
| v23 five NN points | about 221,501--229,100 requests; 1.333--1.379x normal |
| v23 IPC | 0.35339--0.35340 for all five points |
| positive callback rate | TRAIN 6.29%; GUARD 1.32%; EVAL 6.91% |

The direct, supported diagnosis is a mismatch between weighted training,
post-hoc correction, and checkpoint selection. The near-identical behavior
across hidden sizes does not support adding h256/h1024. A second structural
problem is that the exact-PC-only recurrent state never observes the other-PC
callbacks between two uses of the same PC, so it cannot learn chronological
interference or staleness.

## Input boundary (unchanged)

The neural encoder receives exactly the same two source fields as before:

```text
current PC (64 lossless bits) + current cache-line number (58 lossless bits)
```

Captured Stride actions are labels and an offline-normal comparator only. The
NN does not receive the normal tracker's entries, confidence, last stride,
candidates, degree, request rate, queue state, future events, or action
outcomes. The v23 input archive is reused byte-for-byte.

## v25 model

Each configured `H` is the total recurrent width and is split evenly:

- a global chronological LSTM sees every callback;
- an exact-PC local LSTM preserves same-PC history;
- a learned fusion combines both contexts.

The output likelihood is

```text
P(ZERO/POSITIVE | h)
* P(K | h, POSITIVE)
* product over real ranks r<K of P(delta_r | h, r)
```

- The hurdle and positive count losses use the natural, unweighted labels.
- `K` is categorical over the positive counts observed in the applicable
  training partition. This dataset-derived support is not a copied normal
  degree or request budget.
- Every real teacher rank supervises all 58 modular delta bits. There is no
  token vocabulary or escape head, so every address-producing output is
  trained. The bit-head starts from add-one-smoothed per-bit marginals of FIT
  during selection and complete TRAIN during final retraining; this is only a
  natural-prior bias initialization, not loss reweighting.
- Ordered decoding masks an already emitted target and chooses the next
  feasible learned action; it never feeds an earlier action back into the NN,
  mutates a decoded address, or emits duplicate targets.

There are no class weights, prior correction, STOP padding, tuned threshold,
normal request budget, previous-action feedback, or normal-policy private
inputs.

## Selection and evidence

For epoch selection, the first 80% of original TRAIN is FIT and the last 20%
is blocked validation. FIT alone defines the selection count support and bit
initialization.
The selected epoch minimizes the complete per-callback NLL (hurdle, positive
count, and all 58 bits at every real rank), with the earlier epoch winning
an exact tie. The seed is then reset and a fresh model is trained on complete
TRAIN for exactly that many epochs using complete-TRAIN support. Original
GUARD is a phase-shift audit only; EVAL is decoded once.

The final report must keep capacity, target quality, trigger quality, request
pressure, and replay IPC separate. Oracle-count/NN-target and
NN-count/oracle-target paths are diagnosis only and are never replayed.

## Run

Use the shared short wrapper from the repository root:

```bash
bash formal_NN_training/experiments/run_623_v25.sh prepare
bash formal_NN_training/experiments/run_623_v25.sh install-outputs
bash formal_NN_training/experiments/run_623_v25.sh replay
bash formal_NN_training/experiments/run_623_v25.sh status
bash formal_NN_training/experiments/run_623_v25.sh package
```

A root `PASS` proves matched input, deterministic decoding, replay, and
accounting consistency. It does not by itself prove an IPC win.
