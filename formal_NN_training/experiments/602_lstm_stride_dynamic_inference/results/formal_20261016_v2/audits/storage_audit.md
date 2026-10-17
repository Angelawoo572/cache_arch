# Storage, traffic and native SPP resource audit

- **Scope:** archived protocol `frozen_formal_20261016_v2`, build `build_20261016_185845`; actual Sacramento sources and formal-row parameters. This audit changed no active build or experiment.
- **Conclusion:** compare complete source-field budgets at the same abstraction level. C++ object size, whole-simulator RSS and hardware SRAM/PPA are different quantities.
- **Evidence:** [itemized JSON](storage_audit.json) and [SPP scoped bounds certificate](spp_bounds_certificate.json). Server sizeof output and reproduction sources remain in the local/Sacramento raw archive at `formal_study/archive/20261016/research_audit/engine/storage_source/`; they are not bundled in this lightweight audit directory.

## Reportable fixed-design budgets

| Method | Fixed source-field budget | With common 16 B cap gate | Boundary |
| --- | ---: | ---: | --- |
| NoPF | 0 B extra predictor state | No output requests | Shared CPU/cache/DRAM excluded |
| NN H8/state 64 | **16,920 B** | **16,936 B** | 16,808 B core arrays/events/scratch + 112 B explicit controller schema |
| Legacy / remove-current / forward 1 / forward 2 Stride | **2,136 B** | **2,152 B** | 2,064 B existing table/control + 56 B scalar scratch + 16 B output addresses |
| Native SPP | **5,938 B** | **5,954 B** | 5,554 B persistent fields + 256 B candidate arrays + 128 B scalar/control; listed 8 formal cases passed guards, not a general static proof |
| Native SPP C++ object `sizeof` | **33,432 B** | Not applicable | Host representation including diagnostic control; not the source-field hardware budget |

- Do not compare SPP's 33,432 B C++ object directly with NN's source-field reservation as if both described the same implementation layer.
- Each formal `host_time.txt` reports whole-simulator RSS, including code, trace reader, shared caches, allocator and diagnostics; it is not predictor SRAM.

## NN core and controller accounting

- Retain the raw **16,808 B** counter as core arrays, event payloads and forward scratch. Its historical name `deployment_bytes_configured` omits separate ring/control words; no raw result is rewritten.
- Core sum: **7,632 B weights + 6,144 B state table + 1,344 B scratch + 24 B active metadata + 512 B decoded delta/address + 384 B input FIFO + 768 B output FIFO =16,808 B**.
- Each PC entry reserves 32 B tag/controller/lifetime plus 64 B FP32 h/c. Old and staged h/c for active metadata are already included in scratch; do not charge twice.
- Unconfigured unbounded control has no finite maximum. Adding independent observed maxima gives a peak envelope only.
- Use conservative 64-bit words rather than assume minimum optimized RTL widths:

| NN controller field | 64-bit words |
| --- | ---: |
| Service due/remaining | 1 |
| Next output release | 1 |
| State allocation lifetime | 1 |
| Global LRU timestamp | 1 |
| State occupancy | 1 |
| Input FIFO head/tail/occupancy | 3 |
| Output FIFO head/tail/occupancy | 3 |
| Predicted K / decoded valid count | 2 |
| Active/failed/page/queue flag reservation | 1 |
| **Total** | **14 words =112 B** |

- Fixed resource settings give **16,808+ 112=16,920 B**. The common allowed-counter 64 +programmable-cap 64 adds 16 B, giving **16,936 B**.
- Optional dynamic configuration of MAC, state/decoder/input/output capacities, nonlinear cycles and bandwidth needs seven additional 64-bit words =**56 B**. This programmable-resource feature is excluded from the fixed-design total.
- Warmup/history controls, teacher ablation and diagnostic counters are not main-design deployment state.
- Host measurements: `sizeof(Dynamic602)=2,152 B`, Event 136 B, Entry 72 B, Output 32 B, RecurrentState 48 B, InferenceResult 136 B. They include vector/map/stream control and diagnostics, exclude dynamic allocations, and cannot replace the source-field budget.

## Stride accounting

- Actual `Tracker`: uint 64 PC, uint 64 line, int 64 stride; server `sizeof(Tracker)=24 B`. Produced stride is first truncated by int 32 arithmetic.
- Server `sizeof(StridePrefetcher)=192 B`, including 80 B deque control and 72 B diagnostic statistics; tracker objects, deque pointer blocks and allocator overhead are additional host storage.
- The 64-entry tracker payload is 1,536 B. Omitting LRU/valid/controller does not make this a complete hardware budget. All four action policies need the same number of algorithm trackers.
- Forward-helper duplicate tracking is an actual host representation; a minimal architecture can obtain stride from the same tracker. Host RSS includes the implemented duplication.
- Existing 16 B controller can hold matched-slot 64, degree 32 and emitted-count 32. Input PC/address, current line, stride, offset, candidate offset, page and target need 52 B scalar payload, rounded to a 56 B word reservation; immediate dual-address output needs 16 B.
- Complete fixed source-field budget: **2,064+ 56+ 16=2,136 B**; common cap gate gives **2,152 B**.

## SPP persistent fields and scratch

| Component | Capacity and retained widths | Packed bytes | C++ bytes |
| --- | --- | ---: | ---: |
| ST | 256 ×(valid 1 +tag 16 +last offset 6 +signature 12 +LRU8) | 1,376 | 4,352 |
| PT | 512×4 ×(signed delta 7 +count 4) + 512 ×signature count 4 | 3,072 | 18,432 |
| Filter | 1,024 ×(remainder 6 +valid 1 +useful 1) | 1,024 | 10,240 |
| GHR entries | 8 ×(valid 1 +signature 12 +confidence 32 +offset 6 +delta 7) | 58 | 136 payload |
| Global feedback | pf_useful 64 +pf_issued 64 +global_accuracy 64 | 24 | 24 |
| **Total** | All persistent decision fields | **5,554** | **33,184** |

- Packed widths require source masks or within-page delta bounds. Confidence has no explicit clamp, so retain uint 32. All three global feedback values affect prediction and cannot be excluded as diagnostics.
- Existing `estimate_prefetcher_storage.py` reports 5,508 B but omits these three counters, assumes 10-bit GHR confidence and excludes candidate arrays; it is not complete storage.
- Native `confidence_q[32]` and `delta_q[32]` total 256 B. A7-bit packed delta with confidence 32 would total 156 B, but that compression was not implemented.
- Reserve **128 B scalar/control**: caller 88 B plus maximum nested helper 40 B. Caller covers input/page/base, signature/delta/offset, candidate head/tail, depth, lookahead confidence/way, indices, target and flags.
- Helper envelope covers ST28 B +GHR12 B, or filter 32 B +hash key 8 B; PT is smaller. Diagnostic breadth/depth histograms are excluded.
- Fixed source-field total: **5,554+ 256+ 128=5,938 B**; cap gate gives **5,954 B**. Optional programmable threshold 32×2 adds**8 B**.
- SPP has no newly modeled finite-cost service engine. Hash/table/control ports, scalar registers, SRAM macro padding/ECC and timing cannot be inferred from C++ class size.

## Diagnostic exclusions and common credit gate

- Main NN storage excludes shadow teacher, teacher address snapshots, overlap labels, timestamps, full-event histograms, PC update/eviction statistics, CSV occurrence maps, cohort bookkeeping, files/allocator and offline training/optimizer.
- Native SPP delta/depth histograms are also diagnostics. The silent-only teacher gate is a causal research control, not a free main-deployment feature.
- The common request-credit gate affects output: at least one uint 64 allowed-request counter costs**8 B**; a programmable cap register adds**8 B**. Retired count reuses the CPU counter.
- Report this common 16 B separately for every capped method; the uncapped totals exclude it. Counters used only for observation remain excluded.
- These are auditable source-field schemas with explicit reuse/reservations, not RTL, SRAM area, pipeline-port or timing measurements. Native SPP/Stride remain functional baselines.

## Weight and actual DRAM traffic

- `inference_weight_read_bytes =4×inference_macs` is **analytical FP32 dense-weight operand reads**, not observed SRAM transactions; bias reads are excluded.
- Literal successful-forward bias loads add `4×[9H+2+e+K(6H+1)]` bytes under the source formula.
- The service model overlaps dense arithmetic and dense-weight reads using the slower term, then serializes nonlinear/control/scratch costs. Bias ports, energy and transfers are not charged separately; finite-engine conclusions inherit these assumptions.
- Weights have one resident predictor copy; modeled NN weight DRAM bytes are**0**. Do not add analytical weight reads to actual LLC↔DRAM traffic.
- Actual controller enqueue/merge, bus-start and timed bus-completion counters retain initial pending and ROI-end in-flight work. Requests×64 is not a DRAM traffic measurement.

## Archived native SPP inputs and bounds evidence

- This formal command loads `dynamic602` (method none, common observer) and `spp_dev2`. Actual compiled knobs/logs are fill threshold**90**, request threshold**25**, ST 256, PT 512×4, filter 1024, GHR8; no additional tuning.
- L2 demand operate receives LOAD hit/miss, not RFO. PREFETCH operate occurs only for requests from an upper cache; same-L2 generated SPP requests do not recursively invoke it. This build's L1 prefetcher is none.
- This archived native SPP also receives MSHR/writeback cache-fill callbacks, including demand/RFO/PF/writeback; evicted addresses update filter/global feedback. It has richer feedback than NN PC/address+h/c. Fairness means common machine, data, warmup, ROI, page policy and request budget, with native feedback disclosed—not identical algorithm inputs.
- Source `read_pattern` writes `pf_q_tail` into two 32-entry arrays without a source-level tail bound; lookahead do/while has no explicit depth cap. Same-page breadth statistics exclude cross-page GHR candidates and cannot prove array safety.
- **Scoped result:8/8 complete guarded formal repetitions passed**, covering 602's two ROIs,605 and 623, each original/capped, each 5M warmup + 25M ROI. Guard checks confidence append, delta append and terminal confidence write **before** invalid indexing; legal behavior is unchanged.
- All eight had**0 guard trips** and matched **all 210 summary fields per run**, **1,680 exact field comparisons** after the original matrix completed. The earlier 605 original `original_not_yet_complete` marker was replaced by its completed exact comparison.
- This certifies only the listed full tests, **not** every trace/state/configuration or a general static bound. Unbounded native source structure remains disclosed; no native finite-throughput or RTL proof is implied.
- Original matrix, engine, configs and data are unchanged; the guarded build was not copied into delivery evidence. Full [certificate](spp_bounds_certificate.json); raw proof/log path: `research_output/raw_v2/correctness_repetitions/spp_bounds/`.
- Legacy JSON `array_safety_verified=false` means no general source proof. Separate `array_safety_verified_for_listed_formal_runs=true` and 8/8 PASS record completed data checks; do not describe them as unverified.
