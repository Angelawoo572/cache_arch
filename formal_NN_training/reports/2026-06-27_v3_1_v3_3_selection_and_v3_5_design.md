# v3.1 / v3.3 selection and v3.5 design decision — 2026-06-27

This note turns the measured trace, event, and frozen-capacity evidence into the next notebook decision. It does not replace the raw evidence report.

## Direct answer

The data is sufficient to classify each trace's **bottleneck class** and to decide what v3.5 must measure. It is not sufficient to claim the exact causal reason for every missed standalone target, because the current frozen export records selected actions only. The next notebook must therefore add a decision ledger before interpreting a larger model or Transformer feature as the answer.

## v3.1 is not obsolete

Do not retain only v3.3. v3.3 is the primary candidate-family baseline for `602`, `620`, and especially `623`, but v3.1 remains the IPC winner for `619` and `605` in the default replay. Keeping both is required for a clean candidate-bank ablation.

| trace | v3.1 IPC | v3.3 IPC | current IPC winner | correct role in v3.5 |
|---|---:|---:|---|---|
| 602.gcc_s-734B | 0.42863 | 0.42887 | v3.3, by 0.00024 | primary baseline; retain v3.1 control |
| 619.lbm_s-4268B | 0.38492 | 0.38434 | v3.1, by 0.00058 | primary baseline and efficiency target |
| 605.mcf_s-994B | 0.18862 | 0.18729 | v3.1, by 0.00133 | control only; do not size-sweep first |
| 620.omnetpp_s-874B | 0.24503 | 0.24559 | v3.3, by 0.00056 | provisional baseline pending full normal-baseline normalization |
| 623.xalancbmk_s-700B | 0.36407 | 0.37893 | v3.3, by 0.01486 | main attention / context experiment |

The default replay winner is always determined by IPC, not selected accuracy or coverage alone.

## Per-trace diagnosis and action

| trace | measured bottleneck class | evidence | v3.5 response |
|---|---|---|---|
| 602 | coverage / final-selection gap | v3.3 is almost never late but covers about 90.1% of no-prefetch miss events; sandbox covers about 96.4%. Sandbox has 15,489 normal-only timely events; only 182 are logged as standalone-selected-but-late. | Add decision ledger. If targets were legal candidates but rejected, test candidate-aware attention / ranking policy. If targets were absent, improve the candidate generator instead. |
| 619 | saturated candidate reach; efficiency / compactness problem | v3.1 beats SMS across all five completed frozen-capacity points, with about half of SMS traffic and higher event coverage. | Hold candidate bank fixed. Run compactness sweep first: smaller LSTM versus current LSTM. Use as an attention saturation control, not as the primary attention target. |
| 605 | representation / action-space limited | only about 7% unique-event coverage, huge neither-timely bucket, and the most scattered page footprint. v3.3 regresses from v3.1. | Do not make model deeper or add attention first. Treat as abstain/control until a new, held-out candidate-generator experiment demonstrates increased reachable future targets. The current trace lacks memory data values, so it cannot directly dereference pointer contents. |
| 620 | low coverage plus baseline-comparability issue | v3.3 coverage is about 25.5% relative to SMS, but the full normal table also records streamer at IPC 0.39848, much higher than the restricted sweep's SMS baseline. | First validate and include streamer in every direct normal comparison. Then use ledger to choose candidate-generation versus ranker work. Do not claim parity with strongest normal yet. |
| 623 | strong context-conditioned reach and ranking opportunity | v3.3 beats SPP by 0.02502 IPC; it has 404,054 standalone-only timely events versus 120 SPP-only timely events. | Keep v3.3 context bank fixed. This is the first trace for a controlled attention / size ablation. |

## The mandatory v3.5 change: decision ledger

The v3.5 notebook must first reproduce the frozen v3.3 baseline while exporting, for every evaluated candidate, the following:

```text
trace, demand_idx, pc, line, page, offset, pc_line_occ
causal context: current delta, previous-PC context, reuse-gap buckets, history signature
candidate: target line, delta, source, slot, label-match flag
scores: utility score, lead probabilities, cycle probabilities, issue score
policy state: rank, threshold status, degree state, dedup/LRU/backfill state
terminal result: selected or one explicit reject reason
```

Required reject reasons:

```text
candidate_absent
below_threshold
lead_gate
cycle_gate
rank_cut
dedup
LRU_or_backfill
```

This is reporting-only at first. It must not use normal-prefetcher information as an NN input, label, candidate source, or runtime policy.

## Transformer feature: candidate-specific causal cross-attention

Do not replace the entire pipeline with a large full-sequence Transformer. Add a single controlled ranker feature after the ledger baseline:

```text
raw-only event embeddings
  -> causal LSTM encoder
  -> fixed ring buffer of preceding encoded states
  -> per-candidate query
  -> one causal multi-head cross-attention block over the history buffer
  -> existing utility / lead / cycle heads
```

Each legal candidate receives a distinct query. Attention therefore retrieves the particular recent accesses relevant to that candidate, instead of asking one LSTM hidden state to summarize all history equally.

The candidate bank, labels, calibration grid, and keyed replay must remain identical between the LSTM baseline and the attention ablation. Start with `623`; use `619` as a control that may show no gain because its candidate coverage is already saturated. Run `602` only when the ledger confirms that missed targets are candidate-present but policy-rejected.

## Model size: what the present data can decide

The current results do not identify an optimal parameter count. v3.1 versus v3.3 changes candidate construction as well as resulting selections, so it is not a clean size ablation.

Create three configurations around the verified current baseline and print exact parameter counts from the instantiated model at runtime:

```text
small: reduce recurrent width and/or layers
base: exact current v3.3 architecture
large: increase encoder width, or base plus the one attention block
```

Hold fixed: candidate bank, candidate slots, labels, split, loss weights, calibration grid, export policy, and replay protocol. Compare `623` first, then `619`. Select by replay IPC; use parameter count, issued traffic, useless rate, coverage, and timeliness as secondary trade-offs. Do not call any configuration hardware-feasible merely from parameter count.

## Run order

1. Ledger-only v3.3 reproduction, all five traces. Replay IPC must match existing v3.3.
2. Validate the `620` streamer baseline under the same binary and ROI, and add it to the normal comparison matrix.
3. Fixed-bank small/base/large ablation: `623`, then `619`.
4. One candidate-specific causal cross-attention ablation: `623`, then `619`.
5. Use ledger results to choose the next path for `602`.
6. Keep `605` as an explicit representation-limited control until a new candidate generator has demonstrated higher held-out reachability.

## Capacity result boundary

The completed L1D/L2C/LLC runs demonstrate frozen-list sensitivity only. They are not capacity-trained NN results. LLC 4 MiB is incomplete and must remain excluded from claims.