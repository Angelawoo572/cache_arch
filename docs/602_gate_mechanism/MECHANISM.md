# Gate and payload controls

- [Stage results and limitations](REPORT.md) · [shared project overview](../../RESULTS_INDEX.md).
- Complete: **64 new full runs + 7 compatible references; 71/71 audited rows, no analysis failures**. This stage adds no NN training.
- Three independently trained balanced checkpoints remain frozen. Admitted PC/address events run the original forward computation; h/c publication follows its completion policy.

![Gate/payload controls](diagram.svg)

## Interventions

| Control | Gate | Completed payload | NN compute / h/c |
| --- | --- | --- | --- |
| Learned NN, original or filtered | Raw learned emit | Learned K and deltas | Full original forward |
| NN gate → fixed +2, filtered | Raw learned emit | One line+2 request | Full original forward |
| Unconditional +2 | Every eligible L2 LOAD | One line+2 request | None |
| Frozen PC gate → +2 | Frozen PC bit; absent PC silent | One line+2 request | None |

- Fixed NN payload uses **raw `result.emit`**, including emit=true with K=0/empty learned list. It preserves full decoder work, admission, completion, h/c and raw-K cost; it changes only the final payload.
- Cache feedback may change later callbacks and h/c trajectories. The intervention preserves the update policy, not an identical counterfactual future.
- Current-line filtering rejects a target equal to the triggering aligned line **before page/budget credits**. It adds no lookahead. Fixed +2 controls use the same filter; same-list duplicates are counted, not removed.
- Page policy, request ceiling and cache/PQ path remain explicit. Cheap controls and conventional baselines execute without a new finite lookup-service model.

## Frozen PC-table fit

- Score the **complete saved NoPF streams**: train 0–1M and validation 5–10M, **16,172 + 80,206 = 96,378 events**. Do not fit from the bounded 4,096-event example log.
- Each split independently starts with zero h/c, exact-PC capacity 64 LRU state and output limit 32. Validation receives no extra 4M state warmup; this offline fit differs from the 5M-history performance contract.
- Per seed, combine event frequencies and positive raw-emit votes; keep up to 64 PCs, frequency descending then unsigned PC ascending.
- Freeze `gate = (2 × positive_votes >= event_count)`; capacity 64 and threshold 0.5 are fixed. Unlisted PCs remain silent; no test fitting, threshold sweep or online update.
- The fitting union has **55 PCs**, so each table contains 55 entries. Count/positive fields validate loading; lookup retains PC, validity and a gate bit, without h/c.
- A table miss may be a PC seen during fitting but omitted from a capacity-limited table. Do not equate misses with never-observed PCs. Actual test misses are reported in [per-seed results](results/metrics_per_seed.csv).

| Seed | Train agreement | Validation agreement | Combined agreement | Gate-positive PCs |
| --- | ---: | ---: | ---: | ---: |
| 7 | 99.418748% | 99.467621% | 99.459420% | 24 /55 |
| 17 | 99.690824% | 99.503778% | 99.535164% | 29 /55 |
| 27 | 99.251793% | 99.819216% | 99.724003% | 20 /55 |

- Independent scalar rescoring matches all **six split outputs**; seeds have different weights. Actual C++ loading verifies all table PCs, unknown-PC silence and unchanged table files.
- These agreements describe fitting, not test accuracy or IPC. [Independent fit verification](pc_gate_fit/independent_verification.json), [counts/agreement](pc_gate_fit/agreement_summary.tsv), [real C++ loader](pc_gate_fit/engine_loader_verification.json).

## Formal scope and interpretation

- 602 only; ROIs **25–50M /75–100M**, each with **5M NoPF cache warmup and 5M NN-state history**. MAC0 covers seeds 7/17/27 and uncapped/capped controls; MAC4 covers seed 7 capped learned-payload original/filtered arms only.
- The existing validation-selected **2,363 requests/M** ceiling is not equal realized traffic. MAC0 NN/fixed-target capped rows submit **2,362.96/M**; conventional rows can submit slightly less.
- Uncapped unconditional +2 submits **15,671.84/M /20,631.32/M**, versus filtered NN **9,299–9,477/M /13,873–14,692/M**. Its higher uncapped IPC is confounded by quantity. [Exact comparisons](results/paired_comparisons.csv), [result table](results/RESULT_TABLE.md).
- Seven forward-Stride references retain their original build/spec/protocol; fresh NoPF equality enables the explicit reuse. Native SPP is rerun with the repaired cohort boundary.
- All original uncapped NN controls emit delta 0 first; second slots are **99.9256%–99.9980% +2**. Current-line addresses make up **50.0003%–50.0245%** of raw targets. [Full-ROI distributions](results/full_roi_distributions.csv), [five 5M phases](results/phase_distributions.csv).
- Full-profile checks cover all 64 fresh identities, all 6 original seed/ROI payload controls and all 28 fixed-payload controls. Fixed payload has K=0/1 and exact +2; bounded prefix logs are not full-ROI evidence. [Profile certificate](full_profile_verification.json).
- Shadow Stride advances at every eligible LOAD arrival, including NN-dropped events; labels remain attached to the event. Only the explicit teacher-silent-output intervention reads the label to suppress output.
- See [REPORT](REPORT.md) for numerical outcomes and the [71-row audit](results/audit.json). Close IPC is not statistical equivalence; PC-gate differences do not isolate recurrent state; request labels do not identify per-request causal IPC; unused is not pollution.
- No universal minimum observation/history, cross-workload, hardware-PPA or stateful-novelty claim follows. MAC0 is a functional measurement, not a mathematical IPC upper bound.
