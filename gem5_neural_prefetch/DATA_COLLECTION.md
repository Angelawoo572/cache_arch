# Exact data-collection specification

This file says what the gem5 capture records, when each field becomes visible to hardware, and how the capture is validated. It is a specification. Nothing in it is implemented yet. The [design memo](README.md) explains why each stream exists.

## Conventions

- **Time.** gem5 ticks. The manifest stores the core clock period in ticks. Cycles are ticks divided by that period.
- **Region.** Each run restores a checkpoint, warms up for a fixed number of committed instructions, then measures a fixed number. Windows are chosen before any result is seen.
- **Arms.** `nopf` disables every prefetcher. `fast` enables the conventional prefetchers only. `oracle` and `learned` come in later stages. All arms use the same binary, input, checkpoint and region, on one thread.
- **Keys.** `uop` is the zero-based index of a committed micro-op in the measured region. `ld` is the zero-based index among committed loads, which is the row index of S2. Both are the same in every arm. `seq` is gem5's dynamic sequence number. It also counts squashed micro-ops, so it is valid inside one arm only.
- **Addresses.** Byte addresses, virtual and physical. A line is `address >> 6`.
- **Files.** Each stream is `<name>.u64le.gz`: fixed-width rows of little-endian unsigned 64-bit integers in independent gzip members of 32,768 rows, with `<name>.index.csv` giving `first_row`, `rows`, `byte_offset` and `compressed_bytes` per member. The existing ChampSim captures use this layout, so their [indexed reader](../llc_hybrid_prefetch/learning_opportunity/capture/access.py) pattern applies.
- **Location.** `gem5_neural_prefetch/raw/capture/<workload>__<input>__<window>__<arm>/` under either cache root. `raw/` is git-ignored.

## Streams

Architectural streams S1 to S3 are recorded once per window, from the `nopf` arm. They are identical in every arm, and check V1 enforces that. Timing streams S4 to S10 are recorded in every arm.

| Stream | One row per | Columns, in file order |
|---|---|---|
| S1 `uop` | committed micro-op | `pc`, `meta`, `src`, `dst`, `dst_value` |
| S2 `load` | committed load | `uop`, `pc`, `vaddr`, `size_flags`, `value` |
| S3 `store` | committed store | `uop`, `pc`, `vaddr`, `size_flags`, `value` |
| S4 `load_time` | committed load, in S2 order | `seq`, `paddr`, `t_fetch`, `t_rename`, `t_issue`, `t_data`, `t_head`, `t_commit`, `service` |
| S5 `wrong_path_load` | executed load that was later squashed | `seq`, `pc`, `vaddr`, `t_issue`, `t_squash` |
| S6 `cache` | cache event at L1D, L2 or LLC | `tick`, `code`, `paddr_line`, `vaddr_line`, `who` |
| S7 `prefetch` | prefetch lifecycle event | `tick`, `code`, `request`, `trigger_seq`, `line`, `consumer_seq` |
| S8 `fill_content` | line filled at the fill level | `tick`, `paddr_line`, `vaddr_line`, `w0` to `w7` |
| S9 `dtlb` | data-TLB miss | `seq`, `vaddr`, `t_start`, `t_done` |
| S10 `pressure` | demand admission at the fill level | `tick`, `occupancy` |

Packed fields. Bit 0 is the least significant.

| Field | Layout |
|---|---|
| S1 `meta` | Bits 0–7 micro-op index within its instruction. 8–15 gem5 operation class. 16 load. 17 store. 18 control. 19 indirect control. 20 call. 21 return. 22 conditional. 32–39 number of source registers. 40–47 number of destination registers. |
| S1 `src` | Four 16-bit ids: the first four integer-class source registers. |
| S1 `dst` | Two 16-bit ids: the first two destination registers. |
| Register id | `(register class << 12) \| index`. `0xFFFF` means none. Micro-op temporaries keep their own indices. |
| S1 `dst_value` | The 64-bit result written to the first integer destination. 0 if there is none. |
| S2, S3 `size_flags` | Bits 0–7 access size in bytes. Bit 8 the access spans two lines. Bit 9 the value is truncated to its first 8 bytes. |
| S2, S3 `value` | The bytes read or written, little-endian, zero-extended. |
| S4 `service` | Bits 0–3 the level that supplied the data: 0 store forwarding, 1 L1D, 2 L2, 3 LLC, 4 memory. Bit 4 this load was the first demand use of a prefetched line. Bit 5 this load merged into a prefetch still in flight. Bits 8–11 the level of that line. Bits 12–19 the unit that requested it. |
| S6 `code` | Bits 0–3 level: 1 L1D, 2 L2, 3 LLC. Bits 4–7 event: 1 demand hit, 2 demand miss, 3 demand merged into an in-flight request, 4 fill, 5 eviction. Bit 8 the line or in-flight request came from a prefetch. Bits 12–19 the requesting unit. |
| S6 `who` | The `seq` of the demanding load, or the prefetch `request` id. |
| S7 `code` | Bits 0–7 unit: 1–15 conventional prefetchers in configuration order, 16 traversal unit. Bits 8–15 stage: 1 identified, 2 dropped, 3 queued, 4 issued, 5 filled, 6 first use, 7 evicted unused, 8 unresolved at the end of the region. Bits 16–23 reason. For a drop: 1 in cache, 2 in flight, 3 already queued, 4 queue full, 5 page crossing, 6 translation failed, 7 budget. For a first use: 1 timely, 2 late. |
| S7 `line` | Line number. Bit 63 is set when it is physical. |
| S10 `occupancy` | One byte each: L2 in-flight misses, LLC in-flight misses, L2 prefetch queue, LLC prefetch queue, LLC write buffer, memory-controller read queue. |

`t_issue` is the first tick at which the load left the issue queue. `t_data` is the tick at which its data returned, from the cache or from a forwarding store. `t_head` is the tick at which the load became the oldest micro-op in the reorder buffer, which is the commit tick of the micro-op before it. Stall weight is derived, not stored: `max(0, t_data − t_head)`, the time the load sat at the reorder-buffer head before its data returned. Only one micro-op is at the head at a time, so these intervals do not overlap.

Each run folder also holds `complete.json` with the gem5 version and commit, build variant, hash of the configuration script, hashes of the binary and input, checkpoint identifier, warm-up and measured instruction counts, clock period, cache and prefetcher parameters, unit ids, row counts per stream and the V1 hash. `capture_validation.json` is written only after checks V1 to V5 pass for that run.

## When each field becomes visible

| Field | Visible to hardware at | From |
|---|---|---|
| PC, operation class, register ids | rename | S1 |
| Which earlier load produced a base or index register | rename | derived from S1 |
| Load virtual address | issue, once the address is generated | S2, with S4 `t_issue` |
| Load value, and how long the load took | data return | S2, with S4 `t_data` and `service` |
| Commit confirmation | commit | S4 `t_commit` |
| Outcome of an earlier prefetch | its first use, or its eviction | S7 |
| Contents of a line | its fill at the fill level | S8 |
| Queue occupancy | the sample | S10 |

A feature may enter a decision at tick T only if its visible time plus the tap's transport delay is at most T. Transport delays are parameters of the analysis and are stored with every result. A wrong-path load in S5 is visible at issue like any other load, because hardware cannot tell it apart until the squash.

Never an input: a record with a later visible time. The `nopf` or `fast` outcome of a load that has not resolved. Any label. Any candidate of a conventional prefetcher. S7 rows for units 1–15 exist for labels and analysis only.

## gem5 hooks

Identifiers below were read in the gem5 source at stable release 26.0.0.0, commit [`9d36889`](https://github.com/gem5/gem5/tree/9d368896ce119c17e1289dcbb18f33624724b843), on 22 November 2026. Nothing was compiled or run, so run-time behaviour is confirmed in Stage 0. Use 25.1 or later: the 23.0, 24.1 and 25.0 releases assign the fetch and commit ticks only in tracing builds with a debug flag set.

| Stream | Hook | Upstream today, or patch |
|---|---|---|
| S1 `uop` | The `"Commit"` probe on the O3 CPU, a `ProbePointArg<DynInstPtr>` that fires per committed micro-op. From the instruction: `pcState()`, `opClass()`, the class flags, `srcRegIdx(i)`, `destRegIdx(i)`. The result is read with `o3::CPU::getReg(renamedDestIdx(0), tid)` | Listener only |
| S2 `load` | The same probe: `effAddr`, `effSize` and `memData`. For a load, `memData` holds the bytes once data has returned or been forwarded | Listener only |
| S3 `store` | `"Commit"` for the key and address. The value comes from `"DataAccessComplete"`, a `ProbePointArg<std::pair<DynInstPtr, PacketPtr>>`, at the store's writeback after commit, because a store's `memData` is still null at commit | Listener only |
| S4 `load_time` | `fetchTick`, `renameTick`, `commitTick` and `firstIssue` on the instruction, and `physEffAddr`. `t_data` is the tick of `"DataAccessComplete"`, or of `"ToCommit"` for a forwarded load, which never fires the former. `t_head` is kept by the listener. `service` is joined from S6 by `seq` | Listener only |
| S5 `wrong_path_load` | `"Execute"`, which fires before the squash check, with `"Squash"` and `"Commit"` to settle each load's fate | Listener only. `"Squash"` covers instructions that reached the reorder buffer |
| S6 `cache` | `"Hit"`, `"Miss"` and `"Fill"` on each classic cache, each carrying a `CacheAccessProbeArg` with the packet and a cache accessor. `"Data Update"` with empty `newData` marks an eviction. The demanding load is `pkt->req->getReqInstSeqNum()`. First use of a prefetched line is `hasBeenPrefetched(...)` on the accessor, read inside the `"Hit"` listener before the cache clears the mark | Probes are upstream, in timing mode only. A demand that merges into an in-flight request has no probe: small patch in the miss-queue path |
| S7 `prefetch` | Candidate, drop and issue events inside `prefetch::Queued` (`notify`, `insert`, `getPacket`) and the drops in `BaseCache::getNextQueueEntry`. First use comes from S6. Unused eviction is taken where the cache invalidates a block, because the `hwPrefetched` field of `"Data Update"` is not set reliably on invalidations | Small patch: new probe points. Upstream exposes these events as counters only |
| S8 `fill_content` | `"Data Update"`, a `CacheDataUpdateProbeArg`. A fill has empty `oldData` and the new line in `newData` as 64-bit words | Listener only |
| S9 `dtlb` | No probe was identified | Patch, or defer to Stage 4 |
| S10 `pressure` | In-flight miss and queue occupancy are not exposed through a probe argument | Small patch |

Source files for these hooks at that commit: [`src/cpu/o3/commit.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/commit.cc) · [`src/cpu/o3/lsq_unit.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/lsq_unit.cc) · [`src/cpu/o3/dyn_inst.hh`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/dyn_inst.hh) · [`src/mem/cache/base.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/base.cc) · [`src/mem/cache/cache_probe_arg.hh`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/cache_probe_arg.hh) · [`src/mem/cache/prefetch/queued.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/prefetch/queued.cc)

Traps found while reading the source:

- **Listener types.** A listener's argument type must equal the probe point's exactly, or gem5 stops with a "Wrong type of listener" panic.
- **`completeTick` is not data return.** On a load it marks when the access was initiated. `issueTick` is assigned only in tracing builds. This capture uses `firstIssue` and the probe ticks named above.
- **gem5's "late" is not ours.** The statistic `pfLate` counts prefetches dropped at issue because the line was already present or in flight. In this project a late prefetch is one that a demand merges into while it is still in flight. gem5 does not count such a demand in `pfUseful` and does not mark the block as prefetched. The S6 patch is what makes late prefetches measurable.
- **Hits are hidden from prefetchers by default.** A prefetcher sees ordinary hits only with `prefetch_on_access=True` and `prefetch_on_pf_hit=False`.
- **No data at a read miss.** `PrefetchInfo` carries data on read hits and on writes, and never on read misses. A traversal unit therefore reads lines on the fill path.
- **Classic caches only.** The hit, miss and fill probes fire only in the classic cache model in timing mode. The stock GAPBS, NPB and SPEC scripts use Ruby caches, so this capture needs its own configuration with classic caches and the O3 CPU.
- **Which level served a load.** `Request::getAccessDepth()` exists, but no caller that increments it was found. `service` is therefore derived from S6.
- **ElasticTrace is a reference, not a substitute.** It tracks producer-to-consumer register dependences at rename through the same probes, but records no values and no register identities, and it skips split requests. Its rename bookkeeping is the model for the Stage 3 dependence tag.
- **The upstream indirect-memory prefetcher needs a check.** Read by inspection, its generation loop appears to push the same address on every iteration. Confirm its behaviour before using it as a baseline.

## Labels and the streams they need

Definitions are in [section 4 of the memo](README.md#4-prediction-task-and-labels).

| Label | Built from |
|---|---|
| L0 join key | S2 row index |
| L1 future footprint | S2, plus S4 and S6 of the `nopf` and `fast` arms |
| L2 dependence edges | S1, S2, S3 |
| L3 edge timing | S4, S6 and S8 of the reference arm |
| L4 plan label, L5 gate label | L1, L2, L3 |

The dependence slice walks S1 backward from a consumer load's address registers. It follows register copies, immediate adds and shifts, and store-to-load pairs matched through S3 and S2, and stops at the nearest producer loads or at a window bound. Shift and base are then fitted numerically from the recorded values: for a candidate producer value *v* and consumer address *a*, the edge is affine if `a − (v << s)` is the same constant, or the same other recorded value, across instances. Because the fit uses values, the capture does not need each instruction's immediate operands.

## Validation

| Check | Passes when |
|---|---|
| V1 Architectural determinism | A hash over `pc`, `vaddr`, `size_flags` and `value` of S2 is equal in every arm and in a repeated run. |
| V2 Non-intrusion | With the capture off and on, every gem5 statistic and the final tick are equal. |
| V3 Known structure | On the validation microbenchmarks the extracted L2 edges equal the constructed ones, and the stride array yields none. |
| V4 Reconciliation | Stream counts equal gem5's own statistics: committed loads and stores, demand hits and misses per level, prefetches issued, useful, unused and late. |
| V5 Storage | Every gzip member inflates with a valid CRC, and index rows sum to the manifest counts. |
| V6 Causal prefix | Changing any record after time T leaves every feature for triggers at or before T unchanged. Tested on a synthetic trace. |

### Validation microbenchmarks

| Program | Structure | Constructed ground truth |
|---|---|---|
| `stride` | One array walked at a fixed step | No dependence edge |
| `list` | Nodes linked through a `next` field at a set offset, in shuffled memory order | One self edge at that field |
| `tree` | Binary search tree with `left` and `right` fields, random lookups | Two self edges |
| `csr` | `offsets`, `neighbours` and `values` arrays, loop over neighbour lists | Index edges with the element shift and array base, and a range over each list |
| `hash` | Chained hash table with random keys | Chain edges. The bucket root is not affine in any loaded value |

Each has a working set larger than the LLC and a fixed seed. Their structure validates the extractor in V3. It is never used as a training label.

## Size and sampling

- Row sizes in bytes: S1 40, S2 40, S3 40, S4 72, S5 40, S6 40, S7 48, S8 88, S9 32, S10 16.
- S1 dominates at 2.0 GB per 50M committed micro-ops before compression. It is needed for label extraction in Stage 1 and can be dropped afterwards.
- Start with windows of 50M committed instructions. For scale, one 200M-instruction BFS-3 window in the ChampSim study had 47,703,581 load operands and 5,072,480 LLC admissions. [Source](../llc_hybrid_prefetch/learning_opportunity/DATA.md)
- Compression ratio, capture slowdown and wall time per window are measured in Stage 0 before the matrix is sized.
