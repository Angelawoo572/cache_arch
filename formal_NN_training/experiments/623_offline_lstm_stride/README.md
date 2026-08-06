# 623 Stride v24 — natural-cardinality conditional-action LSTM

This is the active matched-input, open-loop Stride experiment for
`623.xalancbmk_s-700B`.

- Run: `623_offline_lstm_stride_natural_cardinality_v24_seed7`
- Model: `pc_keyed_raw_natural_cardinality_rank_delta_v24`
- Decoder: `categorical_count_then_conditional_rank_delta_map_v24`

v23 showed that correcting the weighted hurdle prior was not enough: all five
capacities still emitted about 1.33–1.38 times the offline-Stride requests and
had essentially identical IPC. v24 therefore changes the supervised object
rather than adding another threshold or prior patch.

## Input and fairness boundary

The runtime encoder remains exactly:

```text
current PC (64 lossless bits) + current cache-line number (58 lossless bits)
```

One shared LSTM is routed through state keyed by exact PC. Captured Stride
actions are labels and the offline-normal comparator only. The NN never sees
the Stride tracker, last stride, confidence, candidate list, normal request
rate, queue state, future rows, or action outcomes.

The v23 input archive is reused byte-for-byte. No recollection is required.

## Natural action-list likelihood

For callback `t` with teacher list
`A_t = (a_t,0, ..., a_t,K_t-1)`, v24 learns:

```text
P(A_t | h_t) = P(K_t | h_t) * product over r<K_t of P(a_t,r | h_t, r)
```

- `K_t` is one unweighted categorical label. `K=0` is the implicit
  no-request case.
- Count support is `0..max(TRAIN teacher count)`; it is dataset support, not
  a copied Stride degree or request budget.
- Delta loss exists only for real teacher ranks `r<K_t`.
- Exact delta symbols come from FIT-TRAIN, followed by one `OTHER` signed-log
  escape.
- Stride actions are always `FILL_L2`, so there is no meaningless fill head.
- Inference uses count argmax and emits exactly `K` independent rank-delta
  argmax actions.

There is no hurdle, log-count regression, STOP padding, class reweighting,
prior correction, probability threshold, page rule, degree cap, or
teacher/predicted-action feedback.

## Checkpoint selection

The last block of original TRAIN, with length equal to the original GUARD
callback count, is held out chronologically. Checkpoints minimize natural
action-list NLL on this blocked validation suffix; earlier epoch breaks exact
ties. Original GUARD is retained only as a phase-shift audit. EVAL is decoded
once after checkpoint selection.

Every capacity also records two diagnosis-only decompositions:

- oracle count + NN action;
- NN count + oracle-action upper bound.

Neither diagnostic is replayed or allowed to support a neural win.

## Sweep and evidence

The complete `h8, h16, h32, h64, h128` sweep remains mandatory. Report IPC,
miss rate, request pressure, count confusion/MAE, target and trigger metrics,
coverage, timeliness, entropy, usefulness, lateness, and raw action-list
hashes separately. Undefined rates use JSON `null` / blank CSV; a numeric
zero means the denominator existed.

A root `PASS` proves input, metadata, deterministic decoding, replay, and
accounting consistency. It does not prove that the NN beats Stride.
