# 602: native prefetchers versus frozen NN students

- **Start here:** paired IPC, accuracy, coverage proxy and timeliness for Stride, Streamer, AMPM and SPP. [Project overview](../../../RESULTS_INDEX.md).
- **Measured scope:** complete 25M-instruction ROIs; original seed7 models; frozen weights with live state updates; synchronous zero-cost inference (MAC0). These are operating points, not a finite-hardware efficiency ranking.
- **Main comparison:** H8 NN versus its own native teacher. H16 AMPM/SPP is a separate width supplement. NoPF is the same matched reference in both budget panels.
- **Evidence:** [42 selected run rows](reader_metrics.csv) · [24 native/NN pairs](reader_pairs.csv) · [Pareto membership](reader_pareto.csv) · [reader audit](reader_audit.json) · [all 88 source rows](per_run.csv). Selection excludes validation, templates, reset-state and fixed-payload interventions.

## Main result: H8, common request ceiling

- **IPC:** NN wins 6/8 capped pairs: both Stride, Streamer and AMPM ROIs; SPP loses both. Across capped and uncapped cases, NN wins 9/16.
- **Quality:** higher accuracy does not imply higher IPC. SPP H8 mainly targets LLC; its L2-only figures do not describe the whole policy.
- **Traffic:** all eight capped NN cases transfer more DRAM bytes than their own native baseline. The common ceiling is not equal actual traffic.
- Values below are **native → NN**. IPC gain is relative to that native, not NoPF. Quality values are percentages; **C is a coverage proxy**, defined below.

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Stride | 0.3701 → 0.4090 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| 0 | Streamer | 0.4282 → 0.4317 | +0.81% | 45.2 → 47.4 | 89.9 → 94.1 | 99.4 → 99.1 |
| 0 | AMPM | 0.4190 → 0.4263 | +1.75% | 39.2 → 44.1 | 77.9 → 87.5 | 99.7 → 100.0 |
| 0 | SPP | 0.4239 → 0.4172 | -1.59% | 72.0 → NA | 62.3 → 0.0 | 99.1 → NA |
| 1 | Stride | 0.3143 → 0.3596 | +14.44% | 41.6 → 66.9 | 42.0 → 68.7 | 43.2 → 62.9 |
| 1 | Streamer | 0.3357 → 0.3376 | +0.56% | 48.2 → 50.1 | 55.7 → 57.8 | 98.6 → 98.2 |
| 1 | AMPM | 0.3272 → 0.3316 | +1.34% | 41.8 → 46.0 | 48.3 → 53.1 | 99.0 → 100.0 |
| 1 | SPP | 0.3328 → 0.3253 | -2.24% | 75.0 → 92.9 | 37.2 → 0.0 | 98.4 → 100.0 |

- SPP NN L2 enqueues: **ROI0 = 0; ROI1 = 14** under the cap (13 useful). ROI0 accuracy/timeliness are NA; ROI1’s high accuracy is a tiny cohort. Use the LLC table below.

![H8 native and NN quality and IPC, capped](reader_quality_cap16144.png)

## SPP: LLC-intent quality

- Completely tracked LLC-intent cohorts are reported separately from L2 intent. These are the first enqueue’s intent, not a guarantee of final fill level.
- Coverage proxy can exceed 100% because useful requests are not a count of unique baseline misses eliminated. Do not add L2 and LLC proxies.

| Budget | ROI | Arm | LLC enqueued | Accuracy % | C % | Timeliness % |
| --- | --- | --- | --- | --- | --- | --- |
| cap 16,144/M | 0 | Native | 227,567 | 26.1 | 33.4 | 99.5 |
| cap 16,144/M | 0 | H8 NN | 403,599 | 47.1 | 107.2 | 99.7 |
| cap 16,144/M | 1 | Native | 229,888 | 28.6 | 19.7 | 98.5 |
| cap 16,144/M | 1 | H8 NN | 403,501 | 49.0 | 59.2 | 99.3 |
| uncapped | 0 | Native | 2,023,000 | 2.9 | 33.4 | 99.9 |
| uncapped | 0 | H8 NN | 2,017,198 | 9.9 | 112.0 | 100.0 |
| uncapped | 1 | Native | 2,597,738 | 3.8 | 29.8 | 98.9 |
| uncapped | 1 | H8 NN | 2,650,096 | 12.4 | 98.7 | 99.9 |

## IPC versus actual requests and DRAM

- **Pareto axes:** higher IPC; fewer actual allowed cache calls/M or fewer completed DRAM bus bytes. A point is dominated only when another has no lower IPC and no higher cost, with at least one strict improvement.
- **Candidate set:** the four natives, their four H8 students and NoPF; each ROI and cap is evaluated separately. Finite compute, training cost and native hardware area are not included.
- **Capped own-native pairs:** Stride, Streamer and AMPM trade higher IPC for more DRAM; native SPP dominates its H8 NN on both IPC and DRAM in both ROIs. No capped H8 NN is a paired IPC/DRAM Pareto improvement.
- Frontiers are descriptive for these measured points. They are not statistical significance or a global optimum; coincident points remain tied. NoPF in the capped panel is a reused reference, not a separate capped run.

![H8/native/NoPF IPC versus actual requests and DRAM, capped](reader_pareto_cap16144.png)

| ROI | Policy | Actual requests/M native → NN | DRAM MiB native → NN | DRAM change | Net L2 miss reduction % native → NN |
| --- | --- | --- | --- | --- | --- |
| 0 | Stride | 4,833 → 9,717 | 10.857 → 11.014 | +1.45% | 17.1 → 58.1 |
| 0 | Streamer | 16,144 → 16,144 | 11.023 → 11.028 | +0.04% | 89.3 → 93.2 |
| 0 | AMPM | 16,144 → 16,144 | 11.134 → 11.466 | +2.98% | 77.7 → 87.5 |
| 0 | SPP | 16,144 → 16,144 | 12.598 → 12.661 | +0.50% | 61.8 → 0.0 |
| 1 | Stride | 14,105 → 14,337 | 20.580 → 21.522 | +4.58% | 18.1 → 43.1 |
| 1 | Streamer | 16,144 → 16,144 | 21.489 → 21.646 | +0.73% | 54.8 → 56.7 |
| 1 | AMPM | 16,144 → 16,144 | 21.485 → 22.412 | +4.31% | 47.7 → 52.9 |
| 1 | SPP | 16,144 → 16,144 | 21.400 → 21.916 | +2.41% | 36.6 → 0.0 |

- For example, capped ROI0 Stride emits about twice as many allowed requests under NN. The shared ceiling alone cannot establish an equal-traffic benefit.
- [Exact requests, bytes and paired changes](reader_pairs.csv) · [Frontier flags for both H8-only and H8+H16 candidate sets](reader_pareto.csv).

## Uncapped H8 comparison

- NN wins 3/8 uncapped pairs. Streamer loses ROI1; AMPM and SPP lose both. This is retained alongside the capped gains.
- The same definitions and complete-ROI scope apply; these are different operating points from the capped rows.

| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Stride | 0.3701 → 0.4090 | +10.51% | 29.0 → 48.7 | 17.2 → 58.2 | 99.5 → 99.9 |
| 0 | Streamer | 0.4311 → 0.4366 | +1.29% | 10.4 → 24.6 | 92.4 → 97.6 | 100.0 → 99.9 |
| 0 | AMPM | 0.4300 → 0.4283 | -0.40% | 13.1 → 36.9 | 91.9 → 89.4 | 99.7 → 100.0 |
| 0 | SPP | 0.4277 → 0.4226 | -1.21% | 56.8 → NA | 65.3 → 0.0 | 99.7 → NA |
| 1 | Stride | 0.3190 → 0.3596 | +12.74% | 42.9 → 66.9 | 47.5 → 68.7 | 41.3 → 62.9 |
| 1 | Streamer | 0.3978 → 0.3909 | -1.73% | 13.5 → 31.6 | 89.5 → 95.2 | 99.0 → 69.4 |
| 1 | AMPM | 0.3928 → 0.3897 | -0.79% | 17.4 → 46.6 | 88.4 → 84.6 | 95.9 → 99.9 |
| 1 | SPP | 0.3875 → 0.3739 | -3.50% | 70.8 → 98.5 | 62.2 → 0.0 | 80.4 → 63.1 |

- [Uncapped quality figure](reader_quality_cap0.png) · [Uncapped Pareto figure](reader_pareto_cap0.png).

## H16 width supplement

- H16 AMPM/SPP improves IPC over its own native in 8/8 measured pairs; this is not an additional training seed or a replacement for the four-route H8 comparison.
- Only capped ROI1 SPP H16 improves both IPC and DRAM versus its own native. The other seven H16 pairs exchange higher IPC for more DRAM.
- Simple templates reproduce nearly all measured H16 IPC gain. These runs do not establish a need for a learned payload or recurrent state. [Template pairs](paired_comparisons.csv) · [mechanism findings](FINDINGS.md).

| Policy | ROI | Uncapped IPC gain vs native | Capped IPC gain vs native |
| --- | --- | --- | --- |
| AMPM H16 | 0 | +0.53% | +2.21% |
| AMPM H16 | 1 | +1.59% | +1.37% |
| SPP H16 | 0 | +2.06% | +1.41% |
| SPP H16 | 1 | +4.15% | +1.45% |

- [H16 IPC and accuracy summary](reader_h16.png) · [H8+H16 Pareto supplement, capped](reader_pareto_h16_cap16144.png) · [uncapped](reader_pareto_h16_cap0.png).

## Metric definitions and limits

- **IPC / performance:** actual retired ROI instructions ÷ actual ROI cycles. Pair NN with its own native in the same ROI and budget. Speedup versus NoPF is a separate CSV field.
- **Reader CSV availability:** NoPF has no prefetch quality, so its reader-level availability flags are false and its ratios are NA. These are presentation flags, not replacements for source tracking flags.
- **Accuracy:** LOAD-useful ÷ enqueued requests for the fully tracked intent cohort. Includes ROI-end pending/resident-unused in the denominator; merges are not new enqueues.
- **Coverage proxy (C):** LOAD-useful cohort requests ÷ same-ROI NoPF LOAD misses at that level. It is not causal miss elimination. **Net miss reduction** separately uses (NoPF misses − candidate misses) ÷ NoPF misses.
- **Timeliness:** timely LOAD-useful ÷ (timely + late LOAD-useful). Zero useful denominator gives NA, not 0%. RFO usefulness is excluded from these quality metrics.
- **Actual requests/M:** allowed cache calls ÷ actual retired instructions × 1M; includes later cache acceptance, merge or rejection. This is not generated proposals or queue insertions.
- **DRAM:** actual completed read + write controller bus bytes, all traffic; MiB = bytes ÷ 2²⁰. Frozen resident NN weight fetches have no modeled DRAM traffic. Requests × 64 is not used.
- **Common protocol:** validation 20–25M; ROI0 25–50M and ROI1 75–100M; each test has 5M cache/state warmup. ROI0 was previously inspected. Page/fill rules remain those of the retained live protocol.
- **Fairness limits:** Stride uses per-PC state and a different training budget from global-state Streamer/AMPM/SPP. Native SPP uses prefetch-fill FILTER feedback; its student observes all raw L2 fill callbacks. These are own-teacher route comparisons, not a matched learning-algorithm ranking.
- **Cost limits:** MAC0 IPC gives no compute/area/energy Pareto result. [Actual fields and partial storage](storage_cost.csv) · [full metric contract](metric_definitions.json) · [method limitations](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).

## Provenance and complete evidence

- **Main H8/native/NoPF:** 34 selected completed test rows reused from `four602_frozen_live_20261017_v2`. **H16:** 8 selected completed test rows from `four602_width_templates_20261017_v1`. No new simulations or training were run to produce this reader view.
- The complete source bundle has 64 reused rows plus 24 later runs (including validation and templates). Reader selection is not an additional experiment or replication.
- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/` (later runs); `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/` (H8/native/NoPF).
- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Models/builds/logs retain their original versions; large raw data is not on GitHub.
- **Secondary reading:** [old replay versus live](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [full-ROI action distributions](action_histograms_full_roi.csv) · [stage curves](phase_progress.csv) · [paper story](PAPER_STORY.md) · [archive manifest](archive_manifest.json).
- **Rebuild only this view:** run `python3 formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/build_reader_results.py` from the repository root with matplotlib available. Source run CSVs remain unchanged.
