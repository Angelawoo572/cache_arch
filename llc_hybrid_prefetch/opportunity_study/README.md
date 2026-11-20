# Architectural opportunities for a learned–conventional LLC prefetcher

This study tests whether LLC-visible history helps a prefetcher select physical addresses early enough to improve fixed-work performance. [Full-scale follow-up](../full_workload/README.md) · [Earlier aligned-stream study](../transition_study/README.md)

In a late BFS-3 interval, a 16-entry page-history table used the previous line offset when a page returned. Two example offset sequences were `50 → 51 → 55` and `9 → 10 → 12`. Page history issued the two targets in time from the same trigger where Stride, next-line, Multi-delta and SPP missed. It saved 21,596 more cycles than the current-access table across the interval. With the same issue slots, next-line saved another 7,037 cycles over page history. In CactuBSSN, a same-page Stride predictor could not form certain cross-page targets. With a legal cross-page range, a fixed physical offset recovered a substantial share of the gain. [BFS replay and request examples](page_memory/README.md) · [CactuBSSN placement results](physical_replay/placement_results.csv)

| Report | Contents |
|---|---|
| [Architectural opportunities](reports/architectural_opportunities.md) | Matched target and cue examples, lead time, head waits, address reach and fixed-work effects |
| [Experiment methodology](reports/experiment_methodology.md) | Baseline audit, predictor-visible state, implementation, tests and commands |
| [History and feature study](history/history_feature_study.md) | Ordered and shuffled 4–64-event histories, transfer, support and physical address vocabularies |
| [Conventional baselines](conventional/conventional_baselines.md) | Matched degree and issue budgets, duplicate checks, queueing and criticality interventions |
| [Learned memory mechanisms](reports/learned_memory_mechanisms.md) | Prior work, candidate mechanisms, hardware costs and falsifiable tests |
| [Next experiments](reports/next_experiments.md) | Evidence and minimal tests for learned retention or retrieval |
| [Workload runs](workloads/README.md) | 31 paired cases, trace sources and commands |

The scripts, configurations and compact results are in this directory. Under both cache roots, `llc_hybrid_prefetch/opportunity_study/raw/traces` contains input views and `raw/runs`, `raw/page_memory` and `raw/physical_runs` contain simulator artifacts relative to the study. Follow the workload and individual experiment guides for provenance and exact paths. [Selective-cue results](deadlines/replay_results.csv) · [Original 48-run baseline](../MEASUREMENTS.md)
