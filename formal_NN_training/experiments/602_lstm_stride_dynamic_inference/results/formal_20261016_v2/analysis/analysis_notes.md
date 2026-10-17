# Formal results analysis

Protocol: frozen_formal_20261016_v2. All tables use only this formal protocol and matching build/ROI controls.

Completed and validated results: 101. Status counts: {"complete": 101}.

Expected matrix plus discovered runs

Headline metrics use LOAD-only cohort outcomes. Legacy U/L metrics remain separately labeled. Pending and resident-unused requests remain censored. Unused is not pollution.

The common budget is a shared cap selected on validation. See realized_request_gap.csv before claiming matched realized traffic.

Teacher request categories are descriptive. Use paired_comparisons.csv silent-output interventions for simulation deltas; do not infer per-request causality from tags.

DRAM tables separate queue acceptance, bus start and completion, boundary pending work, and NN resident weights/internal SRAM reads.

All seed rows remain separate in raw/metric/pair/history/distribution tables. Mean/SD overlays never replace individual observations.

Storage: audited fixed-design source-field budgets are NN16920B, Stride2136B and nativeSPP5938B; common-cap control adds16B (16936/2152/5954B). Applied only when audit protocol/build/resources match. The original NN16808B counter is retained and relabeled core arrays/events/scratch, not full hardware storage.

Host object sizeof, whole-process peak RSS and conservative logical source-field reservations are separate. None is an RTL/SRAM-macro area result. NativeSPP finite compute cost is unmodeled. Candidate-array guard evidence is limited to its listed formal runs, not a universal source-code proof. Full inventory: storage_source_field_audit.json; per-run values: storage.csv.

NN dense weight-read bytes are analytical FP32 multiply-operand reads, exclude bias loads and are not measured SRAM/DRAM traffic. Weights are modeled resident; physical ports/energy are not established.

Main v2 history arms are operational predictor-history comparisons: h/c plus tag/LRU/occupancy can differ at the ROI boundary. They do not isolate h/c history. Cache warmup and future ROI match. A separate immutable-build fixed-metadata h/c-only full formal supplement is pending; do not claim its result from these rows. State updates/evictions are actual observations, not a minimum-history threshold.

Plots: complete. No cross-protocol historical IPC comparison or interpolated missing result is included.

NoPF denominator availability: 2 analyzed rows lack a matched NoPF: [{"run_id": "validation_605_r0_stride_forward1_s7_balanced_m0_b0_sil0_qfifo_hist4000000", "stage": "validation", "workload": "605.mcf_s-994B"}, {"run_id": "validation_623_r0_stride_forward1_s7_balanced_m0_b0_sil0_qfifo_hist4000000", "stage": "validation", "workload": "623.xalancbmk_s-700B"}]. Their NoPF-derived metrics are null; raw IPC/cycles and direct counters remain valid. This is not a raw-result validation failure.

SPP scoped array-write guard certificate: PASS; completed 8/8; guard trips 0; exact original-summary comparisons 1680. This covers only the listed complete formal cases, not a general unbounded static proof.
