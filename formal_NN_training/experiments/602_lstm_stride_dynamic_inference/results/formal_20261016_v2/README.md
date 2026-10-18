# Completed first formal study: frozen_formal_20261016_v2

- **Reader note:** Markdown has English translations and concise formatting; original reports and numeric artifacts remain preserved. [What changed](../../../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md#reader-documentation).

- This is an exact lightweight archive of the completed first formal study, placed with the Stride dynamic-inference experiment that produced it. It contains **101 completed rows: 89 test and 12 validation**, with three independent NN training seeds (7/17/27). The paired balanced/unweighted models use identical initialization, data, and training budget. No simulation or retraining was performed to create this archive. Negative results remain in every table.

- The original study trains a Stride-action student. Its conventional controls include NoPF, Stride variants, and native SPP. Tests on 605/623 use the 602 checkpoint and are cross-workload transfer, not per-workload retraining. This archive does not contain four independently trained teacher students; new four-prefetcher experiments belong to their own sibling experiment/protocol.

## Result entry points

- [All per-seed metrics](analysis/metrics_per_seed.csv),
- [full raw summary fields](analysis/raw_per_seed.csv), and [paired interventions](analysis/paired_comparisons.csv).
- [Actual DRAM queue/bus transactions and bytes](analysis/dram_traffic.csv),
- [request lifecycle](analysis/lifecycle.csv), and [realized request gaps](analysis/realized_request_gap.csv).
- [Storage](analysis/storage.csv), [complete source-field audit](audits/storage_audit.md),
- [per-PC history](analysis/history_per_pc.csv), [resident-state lifetimes](analysis/resident_state_lifetimes.csv), and [full service histograms](analysis/service_distributions.csv).
- [Original findings, including failures](findings.json),
- [metric definitions](analysis/metric_definitions.json), and [analysis notes as originally written](analysis/analysis_notes.md).
- [Main analysis certificate](verification/analysis_final_verification.json),
- [89-row completion certificate](verification/data_audit_final.json), [exact copied-byte inventory](archive_verification.json), and [actual forward source/binary consistency](forward_source_verification.json).
- All 25 original figures have PNG and SVG files in [analysis/](analysis/).
- For example: [ROI0 IPC](analysis/ipc_602_roi0_budget0.svg), [ROI1 IPC](analysis/ipc_602_roi1_budget0.svg), and [request/IPC tradeoff](analysis/request_ipc_602_roi0.svg).
- [Stored matrix](metadata/formal_matrix.json),
- [validation-only budget selection](metadata/budget_selection.json), [training pairing](metadata/training_pairs.json), [training split](metadata/training_split.json), and [resource envelope](metadata/resource_envelope.json).

## Scope and later corrections

- The original SPP LOAD lifecycle measurement mixes L2/LLC fill destinations. Its overall IPC and actual DRAM remain historical measurements, but its SPP accuracy/coverage/timeliness must not be presented as a fully tracked native SPP cohort or pooled with the later repaired gate study. The original field bytes and definitions are preserved here, rather than silently recomputed. The separate gate/payload study records its repaired cohort protocol.

- The first main-v2 history arms vary operational predictor history, including h/c and PC metadata (tags/LRU/occupancy); they do not isolate h/c. A later 14-row fixed-metadata supplement completed successfully. Its certificate is preserved separately at [verification/hc_final_verification.json](verification/hc_final_verification.json); its raw results are not pooled into the main 101 rows. Some original notes still say that supplement was pending: that is the original note's date, not the current completion status.

- Equal configured request ceilings do not imply equal realized traffic. Read `realized_request_gap.csv`; use the documented submitted/accepted denominators. DRAM bytes come from controller bus events. Resident NN weights and analytical internal operand reads are distinct. Evicted-unused is not measured pollution. Source-field storage is not physical SRAM area. Teacher tags describe requests; only the paired simulated intervention estimates the IPC effect of suppression.

## Full evidence and relocation

- The authoritative full local results remain at:

- `/Users/angelawoo/Documents/CMU/ece/cache/formal_study_20261016/research_output/raw_v2`

- The matching Sacramento root is:

- `/home/qianruw/cache/formal_study_20261016/research_output/raw_v2`

- The original metadata retains historical `/home/qianruw/cache_formal_20261016_runs_v2/...` identities; replace only that prefix with the authoritative relocated root when locating files. Do not edit completed metadata, install new hooks over an old simulator, or resume changed source under its old raw identity. The original larger nested `results.json`, raw event files, logs, checkpoints, dependency archive and actual build remain at the full evidence root. None was deleted to make this lightweight archive.

- See [forward/dependency reproduction](../../runtime/FORWARD_REPRODUCTION.md) for exact generation checks and recovery without a SHA/hash ledger.
