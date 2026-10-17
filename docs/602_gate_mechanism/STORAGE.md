# Storage and compute boundaries

- [Stage report](REPORT.md) · [shared project overview](../../RESULTS_INDEX.md).
- Keep **declared source-field reservations**, measured C++ objects/allocations and whole-process RSS separate. None is synthesized SRAM/CAM area, energy or PPA.
- [Full field/probe evidence](storage_fields.json), [original reference inventory](reference_storage_fields.json), [map probe](host_map_probe.cc), [engine-size probe](sizeof_engine_probe.cc). Probes do not change frozen source.

## Actual C++ PC gate

- Frozen `std::map<uint64_t,bool>`, capacity 64, actual 55 PCs per seed; absent PCs remain silent. File count/positive fields validate majority but are not retained in lookup state. [Fields](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L43), [loader](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L119), [lookup](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L184).
- Sacramento GCC 8.5/libstdc++ 20210514: pair 16 B; each node requests 48 B and has 56 B usable allocation.

| Host quantity | Actual 55 entries | Reserved 64 entries |
| --- | ---: | ---: |
| Map header | 48 B | 48 B |
| Node allocation requests | 2640 B | 3072 B |
| Header + requested nodes | 2688 B | 3120 B |
| Header + node malloc-usable bytes | 3128 B | 3632 B |

- Unified `Dynamic602` object 2344 B already includes the 48 B header. Object + 55 requested nodes is 4984 B, excluding other pointed-to allocations, deque blocks, streams, shadow teacher and allocator metadata outside usable blocks.
- Formal-run RSS measures the whole simulator, including diagnostics; it is not predictor deployment storage.

## Declared architectural reservations

| Fixed-design source-field schema | Uncapped | With common cap controls |
| --- | ---: | ---: |
| PC gate + fixed +2 | 560 B | 576 B |
| Unconditional +2 | 32 B | 48 B |
| Full NN, learned or fixed payload | 16920 B | 16936 B |
| Forward Stride reference | 2136 B | 2152 B |
| Native SPP reference | 5938 B | 5954 B |

- **PC gate:** hypothetical packed table payload 528 B = 64 × (64-bit PC + 1-bit gate + 1-bit valid). Add 8 B control flags and 24 B pc/line/address candidate reservation; common programmable cap/allowed-count controls add 16 B. Actual 55 entries do not shrink reserved capacity 64.
- **Unconditional +2:** the same 32 B control/candidate reserve, without a table. Cheap controls execute immediately; comparator/adder/lookup latency, area and energy are not synthesized or modeled.
- **NN:** 16808 B core arrays/events/scratch +112 B explicit controller =16920 B; cap controls add 16 B. The older raw counter's “deployment bytes” label omits that controller and is not a physical total.
- New method/payload/filter selectors fit the existing 64-bit flag reserve. A one-address candidate may reuse forward/result scratch after completion; this is an **architectural reuse assumption**, not C++ in-place reuse.
- Fixed NN payload retains full learned forward, decoder, h/c, serial engine and raw-K cost; raw emit may yield a fixed request even with an empty learned list. [Replacement](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L239), [completion](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L274).
- Actual software creates temporary one-element address/delta vectors with 8 B payload each; their descriptors and allocation overhead are separate.
- Stride/SPP reservations include the original explicit scratch/control inventory. SPP includes signature/pattern/filter/GHR/global-feedback fields and candidate arrays. Neither baseline receives a new finite compute model.

## Diagnostics and exclusions

- Unified object growth is 192 B: payload-selector string 32 +empty map header 48 +phase/ROI words 16 +two profile-map headers 96. The unused gate header remains present in NN software.
- Phase words, exact histogram/class maps, request labels and duplicate-counting sets are diagnostics. `Output` grows 32 → 48 B for annotations; modeled pc/line/address payload remains 24 B.
- Current-line annotation is checked against actual line equality before filtering; it adds no independent predictor feature. Shadow teacher, labels, counters, logs and allocator overhead are excluded from architectural budgets, while their host cost remains in RSS.
- Source references: [output/fields](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L35), [profile maps](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L63), [observer](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measurement/dynamic_observer.cc#L355).
- Filtering occurs after payload generation and before page/budget credits; same-list duplicates are counted without deduplication. [Gate order](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measurement/dynamic_observer.cc#L371).
- Full-ROI/five-phase aggregates do not derive from bounded example logs. No canonical h/c-history supplement is transplanted into this family.
- Dense read accounting is `MAC × 4`, excluding bias reads by the retained model assumption. Resident weights generate zero modeled DRAM bytes; actual DRAM transactions are measured separately.
