# Data-derived conclusions and v3.5 plan — 2026-06-27

This document interprets the measurements collected in `2026-06-27_trace_prefetch_capacity_evidence.md`. It separates facts established by the existing counters and replay results from hypotheses that require a new experiment.

## Scope and comparison rules

- The capacity results are **frozen-list sensitivity controls**: v3.1/v3.3 lists were exported from the baseline-capacity raw oracle. They are not capacity-trained NN results.
- The event attribution is L2C demand-event analysis. It explains measured overlaps, but does not establish a causal program-level explanation.
- `no earlier selected standalone export` means the final frozen list had no earlier selection. It does **not** establish that the candidate bank lacked that target; a candidate decision ledger is required.
- The capacity sweep's normal subset is `no_pref`, `sandbox`, `sms`, `ampm`, and `spp`. Wherever this document says `best normal in the sweep`, that restricted scope is explicit.

## A correction to the comparison labels

The default all-policy normal table contains a `620.omnetpp_s-874B` **streamer** run at IPC `0.39848`, whereas the standalone evidence table and the capacity sweep compare the NN against **SMS at `0.24695`**, the winner only within the five-policy sweep subset. Therefore:

1. `620` is **not** currently a near-parity result against the strongest normal prefetcher recorded in the default all-policy table.
2. `620` must be reported in two columns: `best normal in sweep subset = SMS` and `best observed normal in all-policy table = streamer`.
3. No next-model claim for `620` should use SMS alone as its global normal-baseline target until the configuration comparability of the streamer run is reconfirmed from build/run metadata.

The other default all-policy winners are unchanged by this correction: `602` sandbox, `619` SMS, `605` AMPM/SPP tie, and `623` SPP.

## Valid conclusions from the data

### 1. 619.lbm is the robust standalone success case

v3.1 beats SMS at all five completed frozen-list capacity points:

| point | v3.1 IPC minus SMS IPC |
|---|---:|
| L1D 16 KiB | +0.00369 |
| L1D 64 KiB | +0.00389 |
| L2C 128 KiB | +0.00398 |
| L2C 512 KiB | +0.00320 |
| LLC 1 MiB | +0.00408 |

At default capacity, v3.1 achieves IPC `0.38492` versus SMS `0.38105`, while issuing `293,605` prefetches versus SMS's `568,017`. It covers `0.9723` of the unique no-prefetch demand-miss events with event timeliness `0.9842`; SMS covers `0.8664` with event timeliness `0.9101`.

**Interpretation:** this is a genuine selectivity result, not merely a traffic result. The present candidate bank already reaches almost all relevant events. A larger model is not the first lever; the first model-size question is whether the current LSTM can be made smaller without losing IPC.

### 2. 623.xalancbmk is the strongest context-candidate result

At default capacity, v3.3 reaches IPC `0.37893`, beating SPP `0.35391` by `+0.02502`. It is timely on `404,054` demand events that SPP misses, while SPP is timely on only `120` events that v3.3 misses. v3.3 coverage is `0.8994` and event timeliness is `0.9974`.

The result remains positive at all five completed frozen-list capacity points. Its margin is sensitive to L2C capacity, however: v3.3 is `+0.01399` above the restricted-sweep best normal at L2C 128 KiB but only `+0.00172` at L2C 512 KiB. This is expected to be partly a cache-state / frozen-list interaction, not evidence that the candidate bank alone failed.

**Interpretation:** context-conditioned candidate generation is valuable here. This is the best trace on which to test a candidate-aware attention mechanism, because the action space is already reachable and a better scorer can plausibly improve the final selected list.

### 3. 602.gcc is coverage / selection limited, not simply late

v3.3 has selected accuracy `0.9809`, event timeliness `0.9990`, and unique event coverage `0.9014`, but remains `-0.00741` IPC below sandbox. Sandbox covers `0.9644` of no-prefetch demand-miss events and produces about `15,489` normal-only-timely events versus v3.3. Only `182` of those events are marked `selected but late`; `15,153` have no earlier entry in the final standalone export.

Sandbox wins while issuing `3,839,474` prefetches; v3.3 issues `186,723`. Thus the data establishes a coverage/selectivity gap, not a simple timing problem.

**Interpretation:** do not enlarge the LSTM first. First log whether the missing events were absent from the candidate bank or rejected by score, lead, cycle, rank, degree, or LRU/backfill policy. A larger ranker can only help the latter case.

### 4. 605.mcf is representation limited under the current finite candidate action space

v3.1 is essentially tied with AMPM at default capacity (`0.18862` versus `0.18874`), but its unique event coverage is only `0.0737` and its event timeliness is `0.8008`; v3.3 regresses. The ROI has `102,356` unique load pages, far more than any other trace, and the neither-timely bucket remains about `640k–654k` events.

**Interpretation:** neither a larger LSTM nor a Transformer scorer should be the next expensive experiment for this trace. The first missing capability is a new candidate-generation representation. A scorer cannot select a target that never enters its legal candidate list.

### 5. 620.omnetpp requires baseline normalization before neural-model tuning

The standalone policy is clean on the restricted-SMS comparison: v3.3 event timeliness is `0.9925` and coverage is `0.2552`. But it remains slightly below SMS in that subset, and the all-policy table reports streamer IPC `0.39848`, far above both SMS and the NN replay.

**Interpretation:** first validate the streamer configuration and add it to the direct comparison table. Then use the decision ledger to determine whether v3.3 misses targets through candidate absence or policy rejection. Do not use a Transformer-size sweep as the first response.

### 6. Capacity results establish robustness only within their stated control

Across the five completed points, `619` remains above the restricted-sweep best normal; `623` v3.3 remains above it; `602` remains below sandbox by roughly `0.0074–0.0081`; and `605` remains the weakest. These observations show that the current frozen lists are not a one-capacity fluke. They do **not** show how a model trained separately at each cache capacity would behave.

The LLC 4 MiB point remains incomplete and is excluded from all conclusions.

## What the present data does not establish

- It does not identify a globally optimal neural-network size. v3.1 versus v3.3 changes candidate-bank construction as well as the resulting policy; it is not a controlled size ablation.
- It does not prove why a final selection was absent. The frozen export records only selections, not all candidates and rejected candidates.
- It does not prove that a Transformer will improve any trace. The evidence only identifies where candidate-aware history retrieval is a plausible controlled ablation.
- It does not establish a hardware-feasible implementation cost. The current notebooks are offline research models with large hashed embedding tables.

## Current model-size baseline

The current v3.3 notebook defines the following controlled presets with the same raw-only features, 64 candidate slots, lead/cycle heads, and candidate-ranker structure:

| preset | parameters | sequence core |
|---|---:|---|
| XS | 1,952,801 | 1-layer LSTM, hidden 128 |
| S | 2,717,638 | 2-layer LSTM, hidden 160 |
| M | 3,394,987 | 2-layer LSTM, hidden 192 |

`S` is the current all-five v3.3 model. The parameter count is dominated by hashed embeddings, so `XS` is still an offline research model, not a direct hardware area claim.

## v3.5: required first notebook change — full decision ledger

Create a new notebook rather than mutating the frozen v3.3 record:

`LSTM_standalone_multihorizon_candidate_prefetcher_v3_5_ledger_attention.ipynb`

It must preserve the standalone contract:

```text
raw no-prefetch demand stream
  -> train-prefix-only candidate bank
  -> causal neural scorer
  -> calibrated finite policy
  -> frozen PC-line-occurrence keyed replay
```

The ledger is reporting-only and must contain, for every event/candidate evaluated during calibration and export:

| field group | minimum fields |
|---|---|
| event identity | trace, demand_idx, pc, line, page, offset, pc_line_occ |
| causal context | current delta, prior-PC hash, reuse-gap buckets, recent-history signature |
| candidate identity | candidate line/delta, candidate source, candidate slot, whether the future label matches |
| model outputs | utility logit/probability, lead-bin probabilities, cycle-bin probabilities, issue score |
| policy state | rank, threshold, selected flag, degree state, dedup/LRU state |
| rejection explanation | `candidate_absent`, `below_threshold`, `lead_gate`, `cycle_gate`, `rank_cut`, `dedup`, `LRU/backfill`, or another explicit terminal reason |

The report should aggregate these ledger records specifically for the normal-only-timely events on `602`, `619`, `620`, and `623`. This is the experiment that decides whether the next fix belongs in candidate generation or scoring/policy.

## Transformer feature: the right first ablation

Do **not** replace the whole pipeline with a large full-sequence Transformer. The current task is finite candidate ranking; the most targeted Transformer feature is candidate-specific causal cross-attention.

### Architecture

1. Keep the raw-only token embeddings and streaming LSTM encoder from v3.3.
2. Retain a fixed causal ring buffer of the previous `W = 128` encoded demand-event states.
3. For each of the existing 64 legal candidates, form a candidate query from its delta / target-offset / source embeddings and the current LSTM state.
4. Use one 4-head causal cross-attention block over that finite history buffer.
5. Feed `[current LSTM state, candidate embedding, retrieved attention context]` to the existing utility, lead, and cycle heads.

This adds a Transformer property that the LSTM does not have: each candidate can retrieve the particular recent accesses most relevant to that candidate, rather than relying only on one compressed recurrent state. The history remains strictly causal and the candidate bank remains train-prefix-only.

### Why this is the correct target

- **623:** candidate reach is already high; better candidate-specific contextual ranking is plausible.
- **602:** useful only if the ledger shows that many missed targets were present but rejected.
- **619:** serves as a saturation / efficiency control; attention should not be accepted if it increases size or traffic without IPC gain.
- **605:** not a primary attention target because current reachability is low.
- **620:** run only after baseline normalization and ledger diagnosis.

## Controlled experiment order

1. **Ledger baseline:** reproduce v3.3 exactly with ledger export on all five traces. Replay IPC must match the existing v3.3 values before interpreting any new model.
2. **Size sweep with no architectural change:** XS / S / M, fixed v3.3 bank and policy grids. Run `623` first and `619` second. Use `602` only after ledger classification. Do not spend a size sweep on `605`.
3. **Attention ablation:** compare LSTM-S versus LSTM-S + one candidate cross-attention block, with the same candidate bank, labels, loss weights, calibration grid, and replay script.
4. **Policy variants only for the winner:** balanced versus quality / coverage export policies, then replay. Select by IPC first; use issued traffic, useless rate, unique-event coverage, and timeliness to explain the choice.
5. **Separate candidate-generator work for 605:** only after a held-out candidate-reachability experiment demonstrates a meaningful new action-space gain.

## Decision rule

- Prefer **XS** if it matches S in replay IPC within a predeclared tolerance while lowering model size and traffic.
- Prefer attention only when it improves replay IPC on `623` or repairs a ledger-identified scoring bottleneck on `602`, without relying on any normal-prefetcher signal.
- Do not claim progress on `620` until the streamer comparison is normalized.
- Treat IPC as the winner metric; all other counters are explanatory.