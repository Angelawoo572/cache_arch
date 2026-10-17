# Concise paper story: widths and unconditional templates

- Status: **PASS**; new complete formal runs 24/24. This is a post-inspection supplemental mechanism study on602, using retained real seed7 H8/H16 teacher-action models and explicit compatible archived controls.

- The question is whether added recurrent width improves the same teacher-action student, and whether its learned decisions are needed after a frozen validation-derived unconditional payload table is available. The independent online NN and the table are actual simulation interventions; request labels alone do not establish a performance cause.

| Statement | Evidence status | Measured support / limit |
|---|---|---|
| H16 improves over H8 in the measured pairs | Supported in this matrix | 8 positive / 0 negative / 0 tied; IPC change +0.0295% to +7.9310%. Same seed/data/training budget within policy; no independent-seed claim. |
| Learned payload/state is universally necessary for the full-NN benefit | Not supported | H16 unconditional table vs NN: 2 positive / 2 negative / 4 tied; IPC change -0.1247% to +0.0858%. Failure of one table to match is not proof of stateful novelty. |
| Common ceiling gives equal realized traffic | Not supported | Cap16144/M is inherited from original validation NoPF, while actual allowed requests and completed DRAM bytes are separately measured. |
| Benefits survive limited inference resources | Unverified | IPC is synchronous MAC0; actual dense/nonlinear counts and FP32+h/c storage are reported. Counted dense work at four MAC/cycle / measured MAC0 ROI cycles is a necessary throughput condition, not finite-cost IPC. |
| Three seeds / broader workload generality | Unverified | One retained seed7,602, two complete25M test ROIs; no retraining,605, online updates or engine redesign. |

- Protocol: train0–20M, fit templates only on each matching full20–25M validation output; tests25–50M and75–100M, each5M common cache warmup. ROI0 was already observed; the supplemental comparisons are post-inspection. H8 fixed tables use the original H8 validation selection; H16 uses its own modalK and joint per-slot delta/fill. No test fitting, threshold selection or prefix extrapolation.

- Native SPP keeps prefetch-fill FILTER feedback; its NN observes all raw L2 fill callbacks. Target-L2 and target-LLC LOAD cohort accuracy/timeliness remain separate, with NA for incomplete tracking. Useful/NoPF-miss ratios are not causal miss reduction. Pending/resident-unused is retained and unused is not called pollution.

- [All paired outcomes](paired_comparisons.csv), [per-run IPC/requests/DRAM/cohorts](per_run.csv), [new24](per_run_new.csv), [audit](audit.json), [actual distributions](action_histograms_full_roi.csv), [phase distributions](action_histograms.csv), [storage and counted costs](storage_cost.csv), [full findings](FINDINGS.md).

![Actual frozen NN and unconditional-table routes](method.svg)
