# 602 results: native prefetchers versus H8 and H16

- **Compare:** Stride, Streamer, AMPM and SPP with their own frozen NN students on IPC, accuracy, coverage, timeliness and traffic Pareto trade-offs.
- **Available live results:** H8 for all four routes; H16 for AMPM and SPP. Stride/Streamer H16 has no completed result in this two-ROI/cap protocol. [Earlier Stride H16](formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/summary.csv) is retained separately under a different protocol.
- **Capped IPC result:** H8 wins 6/8 native pairs; H16 AMPM/SPP wins 4/4. Seed7, complete ROIs, frozen weights, live state and zero-cost inference (**MAC0**). [Exact run metrics](formal_study/results/602_width_templates/reader_metrics.csv).

## H8 and H16: quality and IPC

- **Common ceiling: 16,144 allowed requests/M.** Each quality cell is **native → NN**, in percent. IPC change is relative to the matching native.
- **C = coverage proxy:** useful LOAD requests ÷ matched NoPF LOAD misses; it is not unique misses eliminated. [Definitions and net miss reduction](formal_study/results/602_width_templates/README.md#definitions-and-scope).

| Model | ROI | NN IPC change | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- |
| Stride H8 | 0 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| Stride H8 | 1 | +14.44% | 41.6 → 66.9 | 42.0 → 68.7 | 43.2 → 62.9 |
| Streamer H8 | 0 | +0.81% | 45.2 → 47.4 | 89.9 → 94.1 | 99.4 → 99.1 |
| Streamer H8 | 1 | +0.56% | 48.2 → 50.1 | 55.7 → 57.8 | 98.6 → 98.2 |
| AMPM H8 | 0 | +1.75% | 39.2 → 44.1 | 77.9 → 87.5 | 99.7 → 100.0 |
| AMPM H16 | 0 | +2.21% | 39.2 → 45.2 | 77.9 → 89.9 | 99.7 → 100.0 |
| AMPM H8 | 1 | +1.34% | 41.8 → 46.0 | 48.3 → 53.1 | 99.0 → 100.0 |
| AMPM H16 | 1 | +1.37% | 41.8 → 46.7 | 48.3 → 54.0 | 99.0 → 100.0 |
| SPP H8 | 0 | -1.59% | 72.0 → NA | 62.3 → 0.0 | 99.1 → NA |
| SPP H16 | 0 | +1.41% | 72.0 → 66.8 | 62.3 → 91.8 | 99.1 → 98.9 |
| SPP H8 | 1 | -2.24% | 75.0 → 92.9 | 37.2 → <0.01 | 98.4 → 100.0 |
| SPP H16 | 1 | +1.45% | 75.0 → 68.5 | 37.2 → 56.0 | 98.4 → 97.9 |

- **H8:** IPC wins 6/8 capped pairs and 9/16 across both budgets. **H16:** wins 4/4 capped pairs and 8/8 across both budgets, on AMPM/SPP only. [All pairs, including losses](formal_study/results/602_width_templates/reader_pairs.csv).
- **SPP differs by width:** H8 mostly targets LLC; its capped L2 cohort is empty in ROI0 and only 14 enqueues in ROI1 (13 useful). H16 improves IPC despite lower L2 accuracy than native in all four cases. Quality and IPC must be assessed together.
- **Both budgets:** [full H8/H16 tables](formal_study/results/602_width_templates/README.md) include uncapped results, actual IPC, counts and traffic. Values rounded to 100.0% are not always exactly 100%.

![Native, H8 and H16: IPC and L2 quality under the common ceiling](formal_study/results/602_width_templates/reader_quality_cap16144.svg)

## SPP: keep L2 and LLC quality separate

- Higher LLC accuracy in H8 coexists with lower IPC; H16 shifts usefulness toward L2 and improves IPC. These observations do not isolate a causal mechanism.
- LLC accuracy, coverage proxy and timeliness below use fully tracked first-enqueue-intent cohorts. A proxy above 100% is not more than 100% of unique misses eliminated. [LLC counts and both budgets](formal_study/results/602_width_templates/README.md#spp-llc-quality).

![SPP native, H8 and H16 LLC quality under the common ceiling](formal_study/results/602_width_templates/reader_spp_llc_cap16144.svg)

## Pareto: IPC versus actual requests and DRAM

- **Shown together:** four natives, four H8 students, two H16 students and NoPF. Each ROI/cap has its own frontier. Higher IPC and lower actual traffic are preferred; compute, area and energy are not measured axes.
- **H8 capped:** all eight NN cases use more DRAM than their own native. Native SPP dominates H8 NN on IPC/DRAM; the other routes trade more DRAM for higher IPC.
- **H16:** all eight pairs dominate their own native on IPC/actual requests. On IPC/DRAM, only capped ROI1 SPP dominates native; the other seven pairs use more DRAM. A common ceiling does not equalize actual traffic. [Exact pair classifications](formal_study/results/602_width_templates/reader_pairs.csv).

![Native, H8 and H16 IPC–traffic Pareto frontiers under the common ceiling](formal_study/results/602_width_templates/reader_pareto_h16_cap16144.svg)

## Conclusions and limits

- **Supported here:** mixed H8 outcomes; positive AMPM/SPP H16 IPC outcomes; different request/DRAM trade-offs. Better accuracy alone does not establish better performance.
- **Not supported:** a universal NN advantage or a necessary learned payload/state. Validation-frozen templates reproduce nearly all measured H16 IPC gain. [Mechanism evidence](formal_study/results/602_width_templates/FINDINGS.md).
- **Still open:** more training seeds/workloads and finite-cost deployment. Route training/state budgets differ; native/student SPP fill feedback differs. ROI0 was previously inspected. [Method limits](formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md) · [cost assumptions](formal_study/results/602_width_templates/storage_cost.csv).

## Evidence and locations

- **Protocol:** validation 20–25M; ROI0 25–50M; ROI1 75–100M; 5M cache/state warmup each. Old replay and earlier finite-engine studies remain separate. [Replay alignment](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [earlier gate/finite results](docs/602_gate_mechanism/REPORT.md).
- **Completed evidence:** original 64-row H8 study + 24 later H16/template runs. This reader view selects 34 reused H8/native/NoPF test rows and 8 H16 test rows; no new training or simulation. [Labeled source](formal_study/results/602_width_templates/per_run.csv) · [paper story](formal_study/results/602_width_templates/PAPER_STORY.md).
- **Shared figures/CSVs:** [formal_study/results/602_width_templates](formal_study/results/602_width_templates/README.md). Main figures use SVG; PNG exports remain available.
- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`; original H8/native raw: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.
- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; original H8/native raw: `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Large logs, models/builds and traces stay on the machines.
- **Branches/recovery:** `main` is the shared reading branch; experiment/archive branches are preserved. [Layout and preservation](docs/602_width_templates/LAYOUT_AND_RECOVERY.md#repository-and-documentation-preservation).
