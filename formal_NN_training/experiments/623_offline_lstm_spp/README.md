# 623 SPP v25 — fixed-global cardinality, fill, and exact delta bits

Active run: `623_offline_lstm_spp_global_cardinality_unique_v25_seed7`  
Models: `global_cardinality_unique_spp_lstm_h8/16/32/64/128`  
Parent input: `623_offline_lstm_spp_finite_joint_rank_v23_seed7`

## Why v23 failed

v23 padded every callback to ten ranks, making 93.40% of its token labels STOP
although 58.05% of EVAL callbacks had a teacher action.  h8 through h128 emitted
0, 1, 991, 2,325, and 5,096 requests versus 804,086 teacher requests; all five
neural replays stayed at no-prefetch IPC 0.35321.  h128 recovered only 0.63% of
the teacher request count.  This is a target/loss failure, not evidence that a
larger LSTM alone would solve SPP.

For comparison, normal SPP reached IPC 0.35390 with 804,086 requests, while the
non-neural modal `(+3, LLC)` control reached IPC 0.35424.  The control is not a
candidate for the NN; it is evidence that useful spatial actions exist in the
same replay and that v23 failed to express them.

The abandoned v24 draft removed STOP padding but introduced a global/event-core
selection and an exact-token/OTHER vocabulary.  Because every TRAIN pair fit in
the exact vocabulary, the OTHER payload heads received no TRAIN labels.  Its
decode also ranked the OTHER token without including payload probability.  v25
removes the token vocabulary entirely.

## Unchanged input and fixed core

The NN still receives exactly the normal SPP decision-effective external input:

```text
callback kind + demand line or fill-evicted line
```

It is losslessly encoded as 59 bits and processed in the recorded source order.
PC, candidates, SPP tables/GHR/filter, action outcomes, thresholds, and request
rate are not neural inputs.  All five sizes use one global chronological
one-layer LSTM; only hidden size changes.

## Direct supervised output

For each callback, v25 predicts categorical count `K`.  `K=0` means no request;
support is zero through the maximum TRAIN teacher count, not an output budget.
For every real teacher rank `r<K`, it supervises:

1. unweighted L2/LLC fill cross entropy; and
2. all 58 bits of `(target_line - base_line) mod 2^58`, using the head selected
   by the teacher fill.

Both fill-specific bit heads must have real TRAIN examples or training fails.
The fill bias starts from the add-one-smoothed natural TRAIN fill marginal; each
bit-head bias starts from its teacher-fill-specific add-one-smoothed TRAIN bit
marginal.  These are initializations only, not loss weights or decode correction.

TRAIN optimization and GUARD checkpoint selection use the identical per-callback
objective:

```text
count CE + sum(real-rank fill CE) + sum(all real-rank 58-bit Bernoulli NLL)
```

There is no STOP padding, class weighting, hurdle, prior correction, joint-action
token, action vocabulary, OTHER token, float delta, clipping, or rounding.

## Exact unique-target decode

Count argmax chooses `K`.  At each ordered rank, both fills produce a 58-bit
independent-Bernoulli distribution.  For each fill, exact k-best flip-subset
enumeration finds the maximum-probability payload whose target is not already
used.  The score compared across L2 and LLC is:

```text
fill log-probability + exact payload log-probability
```

The higher score wins; deterministic ties prefer L2 and then the lower target.
The mask is by target address across fills.  It never edits an address or lowers
`K`, and fails closed only if no feasible payload exists.  Earlier actions affect
only this feasibility mask; they are never neural inputs.  No same-page rule,
candidate bank, threshold, degree, or normal template is imposed.

For realized count classes `K`, parameter count is:

```text
9*H^2 + (191+K)*H + K+118
```

EVAL is loaded only after GUARD chooses the checkpoint and is policy-decoded
once.  Oracle decompositions remain diagnosis-only; the modal-LLC control remains
non-neural.  No IPC claim is valid until all five replay points pass validation.
