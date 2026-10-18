# 602: H8 and H16 versus native prefetchers

- **Read:** [project overview](../../../RESULTS_INDEX.md) · [run metrics](reader_metrics.csv) · [native/NN pairs](reader_pairs.csv) · [Pareto flags](reader_pareto.csv) · [reader audit](reader_audit.json).
- **Coverage of this matrix:** H8 Stride/Streamer/AMPM/SPP; H16 AMPM/SPP. Missing Stride/Streamer H16 results are **not measured under this protocol**, not zero or failed results.
- **Scope:** original seed7 weights; complete two-ROI live inference; frozen weights, updating state; MAC0. Quality cells are **native → NN (%)**; C is the useful/NoPF-miss proxy.

## Capped results

- **Budget:** cap 16,144/M. H8 IPC wins 6/8; H16 AMPM/SPP wins 4/4. Each NN is paired with its own native.

### H8

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Stride | 0.3701 → 0.4090 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| 1 | Stride | 0.3143 → 0.3596 | +14.44% | 41.6 → 66.9 | 42.0 → 68.7 | 43.2 → 62.9 |
| 0 | Streamer | 0.4282 → 0.4317 | +0.81% | 45.2 → 47.4 | 89.9 → 94.1 | 99.4 → 99.1 |
| 1 | Streamer | 0.3357 → 0.3376 | +0.56% | 48.2 → 50.1 | 55.7 → 57.8 | 98.6 → 98.2 |
| 0 | AMPM | 0.4190 → 0.4263 | +1.75% | 39.2 → 44.1 | 77.9 → 87.5 | 99.7 → 100.0 |
| 1 | AMPM | 0.3272 → 0.3316 | +1.34% | 41.8 → 46.0 | 48.3 → 53.1 | 99.0 → 100.0 |
| 0 | SPP | 0.4239 → 0.4172 | -1.59% | 72.0 → NA | 62.3 → 0.0 | 99.1 → NA |
| 1 | SPP | 0.3328 → 0.3253 | -2.24% | 75.0 → 92.9 | 37.2 → <0.01 | 98.4 → 100.0 |

### H16

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | AMPM | 0.4190 → 0.4282 | +2.21% | 39.2 → 45.2 | 77.9 → 89.9 | 99.7 → 100.0 |
| 1 | AMPM | 0.3272 → 0.3317 | +1.37% | 41.8 → 46.7 | 48.3 → 54.0 | 99.0 → 100.0 |
| 0 | SPP | 0.4239 → 0.4299 | +1.41% | 72.0 → 66.8 | 62.3 → 91.8 | 99.1 → 98.9 |
| 1 | SPP | 0.3328 → 0.3376 | +1.45% | 75.0 → 68.5 | 37.2 → 56.0 | 98.4 → 97.9 |

- SPP H8 L2 enqueues: **0 / 14** (ROI0 / ROI1). ROI1 useful: 13. Its L2-only quality cannot describe its predominantly LLC-targeted policy.
- H16 SPP has lower L2 accuracy but higher IPC than native in both ROIs; a metric gain/loss is not a performance mechanism by itself.

![Native versus H8/H16 IPC and L2 quality, cap 16,144/M](reader_quality_cap16144.svg)

| Model | ROI | Actual requests/M native → NN | DRAM MiB native → NN | DRAM change | Net L2 miss reduction % native → NN |
| --- | --- | --- | --- | --- | --- |
| Stride H8 | 0 | 4,833.44 → 9,717.16 | 10.857 → 11.014 | +1.45% | 17.1 → 58.1 |
| Stride H8 | 1 | 14,105.32 → 14,337.28 | 20.580 → 21.522 | +4.58% | 18.1 → 43.1 |
| Streamer H8 | 0 | 16,143.96 → 16,143.96 | 11.023 → 11.028 | +0.04% | 89.3 → 93.2 |
| Streamer H8 | 1 | 16,143.96 → 16,143.96 | 21.489 → 21.646 | +0.73% | 54.8 → 56.7 |
| AMPM H8 | 0 | 16,143.96 → 16,143.96 | 11.134 → 11.466 | +2.98% | 77.7 → 87.5 |
| AMPM H8 | 1 | 16,143.96 → 16,143.64 | 21.485 → 22.412 | +4.31% | 47.7 → 52.9 |
| SPP H8 | 0 | 16,143.96 → 16,143.96 | 12.598 → 12.661 | +0.50% | 61.8 → <0.01 |
| SPP H8 | 1 | 16,143.96 → 16,143.96 | 21.400 → 21.916 | +2.41% | 36.6 → 0.0 |
| AMPM H16 | 0 | 16,143.96 → 16,143.96 | 11.134 → 11.474 | +3.06% | 77.7 → 89.8 |
| AMPM H16 | 1 | 16,143.96 → 16,143.96 | 21.485 → 22.427 | +4.38% | 47.7 → 53.8 |
| SPP H16 | 0 | 16,143.96 → 16,143.96 | 12.598 → 12.663 | +0.51% | 61.8 → 90.7 |
| SPP H16 | 1 | 16,143.96 → 16,143.96 | 21.400 → 21.326 | -0.34% | 36.6 → 54.7 |

## Uncapped results

- **Budget:** uncapped. H8 IPC wins 3/8; H16 AMPM/SPP wins 4/4. Each NN is paired with its own native.

### H8

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Stride | 0.3701 → 0.4090 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| 1 | Stride | 0.3190 → 0.3596 | +12.74% | 42.9 → 66.9 | 47.5 → 68.7 | 41.3 → 62.9 |
| 0 | Streamer | 0.4311 → 0.4366 | +1.29% | 10.4 → 24.6 | 92.4 → 97.6 | 100.0 → 99.9 |
| 1 | Streamer | 0.3978 → 0.3909 | -1.73% | 13.5 → 31.6 | 89.5 → 95.2 | 99.0 → 69.4 |
| 0 | AMPM | 0.4300 → 0.4283 | -0.40% | 13.1 → 36.9 | 91.9 → 89.4 | 99.7 → 100.0 |
| 1 | AMPM | 0.3928 → 0.3897 | -0.79% | 17.4 → 46.6 | 88.4 → 84.6 | 95.9 → 99.9 |
| 0 | SPP | 0.4277 → 0.4226 | -1.21% | 56.8 → NA | 65.3 → 0.0 | 99.7 → NA |
| 1 | SPP | 0.3875 → 0.3739 | -3.50% | 70.8 → 98.5 | 62.2 → 0.0 | 80.4 → 63.1 |

### H16

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | AMPM | 0.4300 → 0.4323 | +0.53% | 13.1 → 23.5 | 91.9 → 93.6 | 99.7 → 100.0 |
| 1 | AMPM | 0.3928 → 0.3990 | +1.59% | 17.4 → 29.3 | 88.4 → 89.1 | 95.9 → 99.9 |
| 0 | SPP | 0.4277 → 0.4366 | +2.06% | 56.8 → 49.0 | 65.3 → 97.2 | 99.7 → 99.6 |
| 1 | SPP | 0.3875 → 0.4036 | +4.15% | 70.8 → 61.9 | 62.2 → 94.0 | 80.4 → 78.6 |

- SPP H8 L2 enqueues: **0 / 66** (ROI0 / ROI1). ROI1 useful: 65. Its L2-only quality cannot describe its predominantly LLC-targeted policy.
- H16 SPP has lower L2 accuracy but higher IPC than native in both ROIs; a metric gain/loss is not a performance mechanism by itself.

![Native versus H8/H16 IPC and L2 quality, uncapped](reader_quality_cap0.svg)

| Model | ROI | Actual requests/M native → NN | DRAM MiB native → NN | DRAM change | Net L2 miss reduction % native → NN |
| --- | --- | --- | --- | --- | --- |
| Stride H8 | 0 | 4,833.44 → 9,717.24 | 10.857 → 11.014 | +1.45% | 17.1 → 58.1 |
| Stride H8 | 1 | 15,473.84 → 14,337.32 | 20.583 → 21.522 | +4.56% | 19.6 → 43.1 |
| Streamer H8 | 0 | 72,356.95 → 32,338.28 | 11.324 → 11.307 | -0.16% | 92.4 → 97.5 |
| Streamer H8 | 1 | 93,024.03 → 42,200.24 | 22.387 → 22.470 | +0.37% | 88.4 → 65.9 |
| AMPM H8 | 0 | 56,938.39 → 19,707.36 | 11.211 → 11.680 | +4.19% | 91.6 → 89.3 |
| AMPM H8 | 1 | 71,340.59 → 25,392.24 | 21.809 → 23.048 | +5.68% | 84.7 → 84.2 |
| SPP H8 | 0 | 90,352.51 → 80,847.59 | 12.924 → 13.533 | +4.71% | 65.1 → <0.01 |
| SPP H8 | 1 | 116,475.39 → 106,574.19 | 22.638 → 25.223 | +11.42% | 50.1 → 0.1 |
| AMPM H16 | 0 | 56,938.39 → 32,339.04 | 11.211 → 11.845 | +5.65% | 91.6 → 93.5 |
| AMPM H16 | 1 | 71,340.59 → 42,628.64 | 21.809 → 23.806 | +9.15% | 84.7 → 88.7 |
| SPP H16 | 0 | 90,352.51 → 80,847.59 | 12.924 → 13.537 | +4.74% | 65.1 → 97.0 |
| SPP H16 | 1 | 116,475.39 → 106,570.99 | 22.638 → 25.258 | +11.57% | 50.1 → 74.0 |

## SPP LLC quality

- Keep LLC and L2 first-enqueue intent separate. They are not final fill destinations; do not add their ratios.
- Empty useful/enqueued denominators give NA. Useful/NoPF-miss ratios can exceed 100%; they are not unique baseline misses eliminated.

| Budget | ROI | Arm | LLC enqueued | Accuracy % | C % | Timeliness % |
| --- | --- | --- | --- | --- | --- | --- |
| cap 16,144/M | 0 | Native | 227,567 | 26.1 | 33.4 | 99.5 |
| cap 16,144/M | 0 | H8 NN | 403,599 | 47.1 | 107.2 | 99.7 |
| cap 16,144/M | 0 | H16 NN | 124,458 | 3.4 | 2.4 | 100.0 |
| cap 16,144/M | 1 | Native | 229,888 | 28.6 | 19.7 | 98.5 |
| cap 16,144/M | 1 | H8 NN | 403,501 | 49.0 | 59.2 | 99.3 |
| cap 16,144/M | 1 | H16 NN | 118,113 | 8.4 | 3.0 | 99.9 |
| uncapped | 0 | Native | 2,023,000 | 2.9 | 33.4 | 99.9 |
| uncapped | 0 | H8 NN | 2,017,198 | 9.9 | 112.0 | 100.0 |
| uncapped | 0 | H16 NN | 1,613,016 | 0.1 | 0.6 | 99.7 |
| uncapped | 1 | Native | 2,597,738 | 3.8 | 29.8 | 98.9 |
| uncapped | 1 | H8 NN | 2,650,096 | 12.4 | 98.7 | 99.9 |
| uncapped | 1 | H16 NN | 2,118,303 | 0.2 | 1.5 | 97.0 |

![SPP native/H8/H16 LLC accuracy, coverage proxy and timeliness, capped](reader_spp_llc_cap16144.svg)
- [Uncapped LLC quality figure](reader_spp_llc_cap0.svg). AMPM has no LLC-intent enqueues: accuracy/timeliness are NA and the coverage proxy is 0.

## Pareto frontiers

- **Axes:** maximize IPC; minimize actual requests/M or completed DRAM bytes. Strict nondominance uses unrounded values. Ties remain tied; no significance or global-optimum claim is made.
- **Combined candidate set:** four natives + four H8 NNs + two H16 NNs + NoPF, separately per ROI and budget. H8-only membership is also retained in `reader_pareto.csv`.
- **H8 capped:** all eight NN cases increase DRAM. Native SPP dominates H8 NN on IPC/DRAM; Stride/Streamer/AMPM trade higher IPC for more DRAM.
- **H16:** all eight pairs dominate native on IPC/actual requests; only capped ROI1 SPP dominates native on IPC/DRAM. Native compute/area and finite NN service costs are outside these plots.
- NoPF is the same reused reference in both budget panels, not a new capped run. Requests below the final ceiling do not prove that cumulative credit never limited earlier output.

![Native, H8 and H16 Pareto comparison, capped](reader_pareto_h16_cap16144.svg)
- [Uncapped H8/H16 Pareto](reader_pareto_h16_cap0.svg) · [H8-only capped](reader_pareto_cap16144.svg) · [H8-only uncapped](reader_pareto_cap0.svg).

## Definitions and scope

- **IPC:** actual retired instructions ÷ cycles. **Net miss reduction:** (NoPF LOAD misses − candidate misses) ÷ NoPF misses; distinct from C.
- **Accuracy:** LOAD-useful ÷ enqueued requests in a fully tracked intent cohort, including ROI-end pending/resident-unused. Cache merges are not new enqueues.
- **Coverage proxy (C):** LOAD-useful cohort requests ÷ same-ROI NoPF LOAD misses at that level. It is not causal miss elimination.
- **Timeliness:** timely LOAD-useful ÷ (timely + late LOAD-useful). Zero denominator gives NA. RFO usefulness is excluded.
- **Requests/M:** credit-allowed cache calls ÷ actual retired instructions × 1M, including later cache merges/rejections. **DRAM:** actual completed read + write controller bus bytes; MiB = bytes ÷ 2²⁰. Requests × 64 is never used.
- **NoPF:** prefetch quality is not applicable; reader availability is false and ratios are NA, independently of source empty-cohort tracking flags.
- **Protocol:** validation 20–25M; ROI0 25–50M and ROI1 75–100M; 5M cache/state warmup each. ROI0 was previously inspected. Different route training/state budgets and native/student SPP fill inputs limit claims.
- **Cost:** synchronous MAC0 IPC; resident NN weights with no modeled weight DRAM fetches. No hardware compute/area/energy Pareto is established. [Storage fields](storage_cost.csv) · [full contract](metric_definitions.json) · [method limits](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).

## Evidence and recovery

- **42 selected test rows:** 34 H8/native/NoPF references + 8 H16 runs. The source bundle retains 64 reused rows + 24 later validation/test/template runs. No new simulation or training is needed for this view. [All labeled rows](per_run.csv).
- **Mechanisms/history:** [template findings](FINDINGS.md) · [paper story](PAPER_STORY.md) · [replay alignment](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [full-ROI actions](action_histograms_full_roi.csv) · [stages](phase_progress.csv).
- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`; `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.
- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. [Layout and preservation](../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).
- **Rebuild figures/tables only:** `python3 formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/build_reader_results.py` with matplotlib. [Independent checker](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/verify_reader_results.py). All runtime files stay inside cache.
