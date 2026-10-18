# 602 results: native prefetchers versus H8 and H16

- **Compare:** Stride, Streamer, AMPM and SPP with their own frozen NN students on IPC, accuracy, coverage, timeliness and traffic Pareto trade-offs.
- **Completed models:** H8 and H16 for all four routes. Compatible retained Stride/Streamer H16 checkpoints were recovered and verified; no retraining was needed. Older replay and live protocols remain separate.
- **Capped IPC result:** H8 wins 6/8 native pairs; H16 wins 8/8. Seed7, complete ROIs, frozen weights, live state and zero-cost inference (**MAC0**). [Exact run metrics](formal_study/results/602_width_templates/reader_metrics.csv).

## H8 and H16: quality and IPC

- **Common ceiling: 16,144 allowed requests/M.** Each quality cell is **native → NN**, in percent. IPC change is relative to the matching native.
- **C = coverage proxy:** useful LOAD requests ÷ matched NoPF LOAD misses; it is not unique misses eliminated. [Definitions and net miss reduction](formal_study/results/602_width_templates/README.md#definitions-and-scope).

| Model | ROI | NN IPC change | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- |
| Stride H8 | 0 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| Stride H16 | 0 | +10.62% | 29.0 → 48.6 | 17.2 → 58.4 | 99.5 → 99.9 |
| Stride H8 | 1 | +14.44% | 41.6 → 66.9 | 42.0 → 68.7 | 43.2 → 62.9 |
| Stride H16 | 1 | +16.68% | 41.6 → 67.5 | 42.0 → 73.1 | 43.2 → 61.6 |
| Streamer H8 | 0 | +0.81% | 45.2 → 47.4 | 89.9 → 94.1 | 99.4 → 99.1 |
| Streamer H16 | 0 | +0.80% | 45.2 → 47.2 | 89.9 → 93.9 | 99.4 → 99.2 |
| Streamer H8 | 1 | +0.56% | 48.2 → 50.1 | 55.7 → 57.8 | 98.6 → 98.2 |
| Streamer H16 | 1 | +0.48% | 48.2 → 49.8 | 55.7 → 57.6 | 98.6 → 98.5 |
| AMPM H8 | 0 | +1.75% | 39.2 → 44.1 | 77.9 → 87.5 | 99.7 → 100.0 |
| AMPM H16 | 0 | +2.21% | 39.2 → 45.2 | 77.9 → 89.9 | 99.7 → 100.0 |
| AMPM H8 | 1 | +1.34% | 41.8 → 46.0 | 48.3 → 53.1 | 99.0 → 100.0 |
| AMPM H16 | 1 | +1.37% | 41.8 → 46.7 | 48.3 → 54.0 | 99.0 → 100.0 |
| SPP H8 | 0 | -1.59% | 72.0 → NA | 62.3 → 0.0 | 99.1 → NA |
| SPP H16 | 0 | +1.41% | 72.0 → 66.8 | 62.3 → 91.8 | 99.1 → 98.9 |
| SPP H8 | 1 | -2.24% | 75.0 → 92.9 | 37.2 → <0.01 | 98.4 → 100.0 |
| SPP H16 | 1 | +1.45% | 75.0 → 68.5 | 37.2 → 56.0 | 98.4 → 97.9 |

- **H8:** IPC wins 6/8 capped pairs and 9/16 across both budgets. **H16:** wins 8/8 capped pairs and 16/16 across both budgets. [All pairs, including losses](formal_study/results/602_width_templates/reader_pairs.csv).
- **SPP differs by width:** H8 mostly targets LLC; its capped L2 cohort is empty in ROI0 and only 14 enqueues in ROI1 (13 useful). H16 improves IPC despite lower L2 accuracy than native in all four cases. Quality and IPC must be assessed together.
- **Both budgets:** [full H8/H16 tables](formal_study/results/602_width_templates/README.md) include uncapped results, actual IPC, counts and traffic. Values rounded to 100.0% are not always exactly 100%.

![Native, H8 and H16: IPC and L2 quality under the common ceiling](formal_study/results/602_width_templates/reader_quality_cap16144.svg)

## SPP: keep L2 and LLC quality separate

- Higher LLC accuracy in H8 coexists with lower IPC; H16 shifts usefulness toward L2 and improves IPC. These observations do not isolate a causal mechanism.
- LLC accuracy, coverage proxy and timeliness below use fully tracked first-enqueue-intent cohorts. A proxy above 100% is not more than 100% of unique misses eliminated. [LLC counts and both budgets](formal_study/results/602_width_templates/README.md#spp-llc-quality).

![SPP native, H8 and H16 LLC quality under the common ceiling](formal_study/results/602_width_templates/reader_spp_llc_cap16144.svg)

## Pareto: IPC versus actual requests and DRAM

- **Shown together:** four natives, four H8 students, four H16 students and NoPF. Each ROI/cap has its own frontier. Higher IPC and lower actual traffic are preferred; compute, area and energy are not measured axes.
- **H8 capped:** all eight NN cases use more DRAM than their own native. Native SPP dominates H8 NN on IPC/DRAM; the other routes trade more DRAM for higher IPC.
- **H16:** H16 dominates its own native in 13/16 IPC/request pairs and 1/16 IPC/DRAM pairs. A common request ceiling does not equalize actual traffic. [Exact pair classifications](formal_study/results/602_width_templates/reader_pairs.csv).

![Native, H8 and H16 IPC–traffic Pareto frontiers under the common ceiling](formal_study/results/602_width_templates/reader_pareto_h16_cap16144.svg)

## Conclusions and limits

- **Supported here:** route- and width-dependent quality/IPC/traffic outcomes. Read each own-native pair; better accuracy alone does not establish better performance.
- **Not established:** a universal NN advantage or a necessary learned payload/state. Full, fixed-payload, reset-state and NN-free controls constrain those claims. [Mechanism evidence](formal_study/results/602_width_templates/FINDINGS.md).
- **Still open:** more training seeds/workloads and finite-cost deployment. Route training/state budgets differ; native/student SPP fill feedback differs. ROI0 was previously inspected. [Method limits](formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md) · [cost assumptions](formal_study/results/602_width_templates/README.md#inference-work-and-width).

## Evidence and locations

- **Protocol:** validation 20–25M; ROI0 25–50M; ROI1 75–100M; 5M cache/state warmup each. Old replay and earlier finite-engine studies remain separate. [Replay alignment](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [earlier gate/finite results](docs/602_gate_mechanism/REPORT.md).
- **Completed evidence:** 64 original H8 rows + 24 AMPM/SPP width/template rows + 30 H16-completion runs (2 validation, 28 full tests). The main view selects 50 test rows; reused references are not extra replications. [Earlier labeled rows](formal_study/results/602_width_templates/per_run.csv) · [30 new rows](formal_study/results/602_width_templates/per_run_h16_completion.csv).
- **Size: parameters → bytes:** all eight models are tensor-verified. [Model size and inference state](formal_study/results/602_width_templates/README.md#model-size-and-inference-state) separates FP32 weights, shared/per-PC state and implementation overhead.
- **Shared figures/CSVs:** [formal_study/results/602_width_templates](formal_study/results/602_width_templates/README.md). Main figures use SVG; PNG exports remain available.
- **New complete raw, local:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; earlier width raw is its parent; original H8/native raw: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.
- **New complete raw, Sacramento:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; earlier width raw is its parent; original H8/native raw: `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Large logs, models/builds and traces stay on the machines.
- **Branches/recovery:** `main` is the shared reading branch; experiment/archive branches are preserved. [Layout and preservation](docs/602_width_templates/LAYOUT_AND_RECOVERY.md#repository-and-documentation-preservation).
