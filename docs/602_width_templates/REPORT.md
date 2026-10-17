# 602 H16 and capped H8 templates

- **Status: 24/24 full runs verified; 9,748 checks, zero failures.** This includes 2 validation, 16 H16 test and 6 capped H8-template test runs. The 64 H8/native/NoPF references retain their original protocol identities.
- **Finding:** H16 retains an advantage over native at MAC0, but simple frozen templates explain almost all observed benefit. SPP matches IPC, cycles, requests and DRAM exactly in four ROI/cap cases; AMPM has small uncapped residuals and loses slightly to its capped template.
- Original seed7 checkpoints were recovered without retraining or changes to the common C++ engine. Read the [project overview](../../RESULTS_INDEX.md) first.
- Comparisons share the simulated machine, trace, L2 LOAD interface, external page/cap gate, warmup and ROI. Native SPP retains its internal same-page rule and prefetch-fill-only FILTER feedback; the student observes all raw fills. Equal effective inputs are not claimed. [Method limits](../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).

## H16 versus native and H8

- H16 wins **8/8 against native and 8/8 against its H8 NN**, within this MAC0/602/seed7 matrix. Native-relative IPC gains are AMPM **0.5320%–2.2054%** and SPP **1.4146%–4.1512%**.
- These are paired live results. Old replay IPC is not substituted into the comparison.

| Model | ROI | Cap | Native IPC | H16 NN IPC | Own template IPC | NN/native | NN/template |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMPM | 0 | 0 | 0.430021 | 0.432309 | 0.432157 | +0.5320% | +0.0351% |
| AMPM | 0 | 16144 | 0.418974 | 0.428215 | 0.428582 | +2.2054% | -0.0857% |
| AMPM | 1 | 0 | 0.392795 | 0.399042 | 0.398544 | +1.5903% | +0.1248% |
| AMPM | 1 | 16144 | 0.327222 | 0.331705 | 0.331756 | +1.3698% | -0.0154% |
| SPP | 0 | 0 | 0.427730 | 0.436556 | 0.436556 | +2.0634% | +0.0000% |
| SPP | 0 | 16144 | 0.423904 | 0.429901 | 0.429901 | +1.4146% | +0.0000% |
| SPP | 1 | 0 | 0.387512 | 0.403598 | 0.403598 | +4.1512% | +0.0000% |
| SPP | 1 | 16144 | 0.332809 | 0.337639 | 0.337639 | +1.4515% | +0.0000% |

- SPP's template matches all four measured performance/traffic cases. AMPM NN is only **0.0351% / 0.1248%** better uncapped and **0.0857% / 0.0154%** worse capped. Those residuals do not isolate a recurrent-state contribution.
- The capped AMPM H8 template also beats the H16 template; larger width is not uniformly better for every deployment.

## Capped H8 templates

- Tables contain address-relative K/delta/fill, without PC or NN state. All three H8 tables reuse the original frozen validation selection byte for byte.
- Each row covers a full 25M ROI. Positive template/NN values favor the template.

| Model | ROI | NN IPC | Template IPC | Template/NN | Actual requests/M: NN / template | DRAM MiB: NN / template |
|---|---:|---:|---:|---:|---:|---:|
| STREAMER | 0 | 0.431659 | 0.431661 | +0.0004% | 16143.96 /16143.96 | 11.028 /11.028 |
| STREAMER | 1 | 0.337604 | 0.337587 | -0.0051% | 16143.96 /16143.96 | 21.646 /21.649 |
| AMPM | 0 | 0.426299 | 0.430109 | +0.8939% | 16143.96 /16130.24 | 11.466 /11.570 |
| AMPM | 1 | 0.331607 | 0.334771 | +0.9542% | 16143.64 /16101.84 | 22.412 /22.447 |
| SPP | 0 | 0.417164 | 0.417164 | +0.0000% | 16143.96 /16143.96 | 12.661 /12.661 |
| SPP | 1 | 0.325346 | 0.325345 | -0.0006% | 16143.96 /16143.96 | 21.916 /21.915 |

- Streamer differs by at most **0.0052%** and SPP by **0.0006%**. AMPM templates improve IPC by **0.8939% / 0.9542%**, with fewer requests.
- This is an actual no-NN intervention. Failure of a PC table or template to match every case would not establish stateful novelty. The H8 losses against native remain in the evidence.

## Requests, DRAM and actions

| H16 NN | ROI | Cap | Actual requests/M: NN / template | Actual DRAM MiB: NN / template |
|---|---:|---:|---:|---:|
| AMPM | 0 | 0 | 32339.04 /32339.04 | 11.845 /11.834 |
| AMPM | 0 | 16144 | 16143.96 /16143.96 | 11.474 /11.476 |
| AMPM | 1 | 0 | 42628.64 /42628.64 | 23.806 /23.764 |
| AMPM | 1 | 16144 | 16143.96 /16143.96 | 22.427 /22.423 |
| SPP | 0 | 0 | 80847.59 /80847.59 | 13.537 /13.537 |
| SPP | 0 | 16144 | 16143.96 /16143.96 | 12.663 /12.663 |
| SPP | 1 | 0 | 106570.99 /106570.99 | 25.258 /25.258 |
| SPP | 1 | 16144 | 16143.96 /16143.96 | 21.326 /21.326 |

- The common cap is an upper limit, not equal actual submissions, acceptance or DRAM. H8 AMPM tables do not always exhaust it; equal request counts can still produce different DRAM.
- Uncapped SPP H16 submits fewer requests than native SPP but increases DRAM by **4.74% / 11.57%** in ROI0/ROI1.
- Capping lowers H16's absolute future-ROI IPC by **16.87% for AMPM** and **16.34% for SPP**. These negative results are retained.
- Full-validation H16 selections are AMPM **K2: (+9,L2), (+8,L2)** and SPP **K5: (+1,L2), (+2..+5,LLC)**.
- Across all **404,238 ROI0** and **532,855 ROI1** uncapped SPP H16 demand events, emit/K and each slot's delta/fill equal that template. This is an empirical fact for the complete ROIs, not a guarantee for all inputs.
- CSVs retain complete and phase-level K support, rare delta/fill, current-line/duplicate/slot classes, and allowed/redundant/useful counts.

## Metric definitions

- **IPC:** actual retired instructions / cycles. **Requests/M:** credit-allowed cache calls / actual retired instructions × 10⁶.
- **DRAM:** actual completed controller read/write bus bytes, with started/completed transactions and boundary pending work retained. Requests × 64 is not used.
- **Quality:** separate cohorts by first source-enqueue L2/LLC intent. Accuracy = LOAD-useful/enqueued; timeliness = timely/(timely + late); coverage = LOAD-useful / matched NoPF LOAD misses at that level.
- A later merge may upgrade fill level, so intent is not final fill destination or all-level usefulness. Incomplete tracking gives NA; L2/LLC ratios are not pooled.
- Pending/resident-unused requests remain censored. Evicted-unused is not pollution; coverage is not causal net miss elimination. Actual misses and MPKI are separate fields.
- Startup and phase curves use complete saved snapshots and actual retired instructions, callbacks and completed updates. Training, warmup history and measurement progress remain distinct; first cumulative crossing of NoPF is not a minimum observation requirement.

## Cost and storage

| Model | FP32 weights | Global h/c | Object + persistent heap, including weights/h/c | Observed fixture scratch peak |
|---|---:|---:|---:|---:|
| AMPM H16 | 26,612 B | 128 B | 28,596 B | 884 B |
| SPP H16 | 26,436 B | 128 B | 28,656 B | 1,148 B |

- These are actual ABI and ordinary C++ new/new[] requested bytes. They exclude the outer adapter, observers, logs, stack and allocator metadata; they are not complete hardware SRAM/PPA or RSS.
- Persistent heap already includes weights/h/c. Scratch peaks cover only the passed correctness fixtures; the decoder has no universal degree cap, so these are not worst-case bounds.
- Essential table fields are uint64 K plus int64 delta/int32 fill per slot: **32 B AMPM / 68 B SPP**. Actual Action ABI is **32 B/slot**; vector capacity, descriptors and control are additional. Essential fields are not the complete implementation footprint.
- IPC assumes synchronous **MAC0**: full forward completes before h/c and output publication, without modeled finite service/queue/bandwidth delay.
- Resident weights and **zero simulated NN-weight DRAM** are explicit assumptions, not a hardware cost result.
- Full-ROI **ceil(counted MACs/4) / measured MAC0 cycles** is **12.95 / 15.75 for AMPM** and **23.95 / 31.15 for SPP**. Four MACs/cycle cannot serve all counted work within those same MAC0 windows, even before other costs; this necessary throughput condition does not predict finite-resource IPC.
- Nonlinear, bias/control, state and storage scopes remain explicit in the cost records.

## Replay versus live evidence

- Recovered seed7 identities are AMPM v7 H16 **6,653 parameters** and SPP empirical-prior v2 H16 **6,609**. Both trained for **8 epochs**, but event counts, optimizer steps and compute differ; this is not matched-compute retraining.
- Old AMPM inference started with fresh h/c at guard **20M**; SPP started at train **0M**. Old replay used **25M cache warmup**. New runs start fresh h/c for **5M NoPF cache/state warmup**, retain h/c, then measure validation **20–25M**, ROI0 **25–50M**, or ROI1 **75–100M**.
- Old AMPM replay: **808,438 generated = loaded → 363,119 / 404,219 matched keys → 726,238 emitted**. SPP: **2,021,095 generated = loaded → 392,788 / 404,219 matched → 1,963,940 emitted**. Matching losses distinguish precomputed replay from independent online inference.
- [Concise alignment](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) and [seven-row original metric table](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/old_replay_alignment.csv) retain source/log evidence.
- [Slide corrections](SLIDE_ERRATA.md) distinguish **163,567 / 404,219 = 40.4649%** of all callbacks from **163,567 / 340,746 = 48.0026%** of teacher-silent callbacks. Requests are not DRAM; zero SPP H8 L2 coverage is not zero LLC benefit.
- Original deck, v1/v2, source and old results remain unchanged. Alignment covers only the retained local **v7_15** deck; no newer cloud slides are asserted.

## Completed routes and limits

| Route | New completed runs | Preserved history |
|---|---|---|
| Stride | 0; original per-PC frozen reference retained | Offline/live/formal/gate; 13 reference rows |
| Streamer H8 | 2 capped-template tests | H8 model, NoPF/native/live/template |
| AMPM H8/H16 | 11 full runs | Original v7 H8/H16 checkpoints and replay |
| SPP H8/H16 | 11 full runs | Empirical-prior H8/H16, fill-aware replay, v1/v2 |

- Scope is **602, one real training seed7 and two defined intervals**. ROI0 was observed; ROI1 is a distinct future interval but its H8 results were already known before this supplement. This is not a fully blind test.
- Global-model finite-engine IPC, extra seeds and cross-workload generalization remain untested. No new weights were trained.
- Small AMPM residuals need independent mechanism interventions; they do not establish stateful novelty.

## Evidence and recovery

- [Summary CSV](../../formal_study/results/602_width_templates/summary.csv) · [24 new rows](../../formal_study/results/602_width_templates/per_run_new.csv) · [88 labeled rows](../../formal_study/results/602_width_templates/per_run.csv) · [paired comparisons](../../formal_study/results/602_width_templates/paired_comparisons.csv).
- [Audit](../../formal_study/results/602_width_templates/audit.json) · [definitions](../../formal_study/results/602_width_templates/metric_definitions.json) · [storage/cost](../../formal_study/results/602_width_templates/storage_cost.csv) · [short paper story](PAPER_STORY.md).
- [ROI0 capped IPC/requests/DRAM](../../formal_study/results/602_width_templates/widths_roi0_cap16144.svg) · [ROI1 uncapped](../../formal_study/results/602_width_templates/widths_roi1_cap0.svg) · [capped H8](../../formal_study/results/602_width_templates/h8_template_cap.svg) · [cost figure](../../formal_study/results/602_width_templates/width_cost.svg).
- [Full action distributions](../../formal_study/results/602_width_templates/distributions_spp_roi1.svg) · [phases](../../formal_study/results/602_width_templates/phases_spp_roi1.svg) · [figure index](../../formal_study/results/602_width_templates/README.md) · [method](../../formal_study/results/602_width_templates/method.svg).
- [Configuration/run guide](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/README.md) · [canonical paths and recovery](LAYOUT_AND_RECOVERY.md).
- Full new raw roots: local `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/` and Sacramento `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`.
- Raw retains checkpoints/models, actual binary snapshot, full logs/callbacks/snapshots, source bindings, cost probes and analysis_final. The reused actual build remains in `formal_study/archive/20261017/raw_v2/builds/`.
- Historical rounds remain in `archive/20261016` and `archive/20261017`. The checkout is the cache root; `work/cache_gate_mechanism` is only a compatibility link. Removed date-root aliases are not recreated; historical JSON is translated only in memory.
- The recorded raw sync checked **345 server files / 176,314,065 bytes**, with all paths/sizes equal. Later local analysis, figures and arithmetic checks are separate from that inventory. Builds, traces and large logs stay local/server; no hash ledger is introduced.
