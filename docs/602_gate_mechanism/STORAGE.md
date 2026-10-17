# Gate mechanism storage boundaries

The implemented PC gate is a frozen, bounded `std::map<uint64_t,bool>`, not a 528B hardware table. The three fitted tables each contain 55 PCs. Capacity is 64; unseen PCs stay silent. `count` and `positives` validate the majority during loading and are not retained in the lookup state ([fields](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L43), [loader](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L119), [lookup](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L184)).

| Quantity | Actual 55 entries | Reserved 64 entries |
|---|---:|---:|
| C++ map header |48B|48B|
| Node allocation requests |2640B|3072B|
| Header + requested nodes |2688B|3120B|
| Header + node `malloc_usable_size` |3128B|3632B|

These values were measured with Sacramento GCC 8.5/libstdc++ 20210514: each pair is 16B, each tree-node allocation requests 48B and has 56B usable allocation. They exclude allocator metadata outside usable blocks and all other engine/observer memory. The unified `Dynamic602` object is 2344B and already contains the 48B map header. Thus object + 55 node requests is 4984B, still excluding deque blocks, streams, shadow teacher and other pointed-to allocations. Host RSS from each formal run remains a separate whole-process measurement.

The reported 528B field is a **hypothetical packed table payload**: 64×(64-bit PC+1 gate+1 valid)=4224bits. An explicit conservative architectural schema adds an 8B control-flags word and 24B immediate pc/line/address candidate reservation: 560B fixed, or 576B with the common 16B cap register/allowed-request counter. Actual valid entries 55 do not reduce the reserved 64-entry payload. The unconditional+2 arm uses the same 32B control/candidate reservation without the table, or 48B including the common budget controls. These are declared storage assumptions; no CAM/SRAM, PC-match comparator, adder, area, energy or lookup-latency synthesis was performed. Both cheap controls execute immediately in the traditional callback path, as legacy Stride does.

The NN fixed-payload arm retains the complete learned forward, decoder work, h/c update, serial engine and original raw K cost. It replaces the completed payload with one line+2 request if `result.emit` is true, even if a learned list is empty ([replacement](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L239), [state completion](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.cc#L274)). The established NN source-field reservation remains 16808B core arrays/events/scratch +112B explicit controller=16920B, or 16936B with common budget controls. The new payload/method/current-filter selector bits fit in the existing 64-bit flags reserve. The one-address fixed candidate can reuse the existing forward/result scratch after completion; this is an explicit architectural buffer-reuse assumption, not a claim that the C++ implementation performs in-place reuse. Software constructs a temporary vector with an 8B address and a second 8B diagnostic delta element.

The new software object grows 192B: 32B payload-selector string +48B empty gate-map header +16B phase/ROI words +96B histogram/class-map headers. The map header is present in NN's unified C++ class even when that arm never uses a PC table. The added phase words, exact maps, class labels and temporary duplicate-counting sets are diagnostic storage. `Output` grows from 32B to 48B because of phase/slot/current/duplicate annotations; its modeled deployment payload remains the existing 24B pc/line/address. The current annotation is checked against actual line/address equality before filtering, so it contains no independent decision feature. Both original and new diagnostic labels, shadow teacher, counters and logs are excluded from the architecture budgets and remain visible in host RSS ([fields](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L35), [profile maps](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/dynamic602_engine.h#L63), [gate](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measurement/dynamic_observer.cc#L355)).

Current-line filtering runs after payload generation and before page/budget credits ([gate order](../../formal_NN_training/experiments/602_lstm_stride_dynamic_inference/measurement/dynamic_observer.cc#L371)). Same-list duplicates are counted in all slots and are not deduplicated. Full-ROI five-phase aggregate distributions and request classes are independent of the bounded example log. No canonical h/c supplemental initialization was transplanted into this mechanism family.

`storage_fields.json` retains the full component/controller inventory and exact probe results. `host_map_probe.cc` accepts 55 or 64 as its argument. `sizeof_engine_probe.cc` compiles against the isolated simulator's `inc/`; neither probe modifies frozen source. Dense NN read accounting remains `MAC*4`, excluding bias reads by the existing cost-model assumption. Resident NN weights generate zero modeled DRAM bytes; physical DRAM transactions are measured independently.

The unchanged conventional reference inventories are retained in
[reference_storage_fields.json](reference_storage_fields.json), with the original
main-v2 audit identity. At that same source-field reservation level, forward
Stride is 2,136B and native SPP is 5,938B, plus 16B for the common programmable
cap. SPP includes all ST/PT/filter/GHR/global-feedback fields, native candidate
arrays and explicit scalar/control reservations. Neither baseline has a new
finite computation model. Those budgets, the measured C++ objects and whole
simulator RSS remain distinct quantities.
