# Exact data-collection specification

This file says what the gem5 capture records, when each field becomes visible to hardware, and how the capture is validated. The [design memo](README.md) explains why each stream exists.

The capture is [`sim/np_ext/np_capture.cc`](sim/np_ext/np_capture.cc). It records S1 to S4, S6 and S8 with probe listeners only and changes no gem5 source file. S5, S7, S9 and S10 are specified here and not recorded yet. The captures taken with it, their validation and what they show are in the [measurement findings](measurement/FINDINGS.md).

## Conventions

- **Time.** gem5 ticks. The manifest stores the core clock period in ticks. Cycles are ticks divided by that period.
- **Region.** Each run restores a checkpoint, warms up for a fixed number of committed instructions, then measures a fixed number. Windows are chosen before any result is seen. The capture object counts committed instructions itself and opens and closes the window on exact instruction boundaries, so every arm records the same instructions whatever its timing. gem5's own statistics are reset and dumped at the nearest cycle boundaries, which can differ by a few instructions.
- **Arms.** `nopf` disables every prefetcher. `fast` enables an L1D stride prefetcher and an L2 signature-path prefetcher. `stride`, `bop` and `ampm` are other conventional configurations, listed in the [simulator README](sim/README.md). `oracle` and `learned` come in later stages. All arms use the same binary, input, checkpoint and region, on one thread.
- **Keys.** `uop` is the zero-based index of a committed micro-op in the measured region. `ld` is the zero-based index among committed loads, which is the row index of S2. Both are the same in every arm. `seq` is gem5's dynamic sequence number. It also counts squashed micro-ops, so it is valid inside one arm only.
- **Addresses.** Byte addresses, virtual and physical. A line is `address >> 6`.
- **Files.** Each stream is `<name>.u64le.gz`: fixed-width rows of little-endian unsigned 64-bit integers in independent gzip members of 32,768 rows, with `<name>.index.csv` giving `first_row`, `rows`, `byte_offset` and `compressed_bytes` per member. The existing ChampSim captures use this layout, so their [indexed reader](../llc_hybrid_prefetch/learning_opportunity/capture/access.py) pattern applies.
- **Location.** `gem5_neural_prefetch/raw/runs/<workload>__o<offset>__<arm>/capture/` on Sacramento, where `offset` is the number of instructions between the workload's marker and the checkpoint. `raw/` is git-ignored.

## Streams

Architectural streams S1 to S3 are recorded once per window, from the `nopf` arm. They are identical in every arm, and check V1 enforces that. Timing streams S4 to S10 are recorded in every arm.

| Stream | One row per | Columns, in file order |
|---|---|---|
| S1 `uop` | committed micro-op | `pc`, `meta`, `src`, `dst`, `dst_value` |
| S2 `load` | committed load | `uop`, `pc`, `vaddr`, `size_flags`, `value` |
| S3 `store` | committed store | `uop`, `pc`, `vaddr`, `size_flags`, `value` |
| S4 `load_time` | committed load, in S2 order | `seq`, `paddr`, `t_fetch`, `t_rename`, `t_issue`, `t_data`, `t_head`, `t_commit`, `service` |
| S5 `wrong_path_load` | executed load that was later squashed | `seq`, `pc`, `vaddr`, `t_issue`, `t_squash` |
| S6 `cache` | cache event at L1D, L2 or LLC | `tick`, `code`, `paddr_line`, `vaddr_line`, `who`, `pc` |
| S7 `prefetch` | prefetch lifecycle event | `tick`, `code`, `request`, `trigger_seq`, `line`, `consumer_seq` |
| S8 `fill_content` | line filled at the fill level | `tick`, `paddr_line`, `vaddr_line`, `w0` to `w7` |
| S9 `dtlb` | data-TLB miss | `seq`, `vaddr`, `t_start`, `t_done` |
| S10 `pressure` | demand admission at the fill level | `tick`, `occupancy` |

Packed fields. Bit 0 is the least significant.

| Field | Layout |
|---|---|
| S1 `meta` | Bits 0–7 micro-op index within its instruction. 8–15 gem5 operation class. 16 load. 17 store. 18 control. 19 indirect control. 20 call. 21 return. 22 conditional. 23 `dst_value` is valid. 32–39 number of source registers. 40–47 number of destination registers. |
| S1 `src` | Four 16-bit ids: the first four integer-class source registers. |
| S1 `dst` | Two 16-bit ids: the first two integer-class destination registers. |
| Register id | `(register class << 12) \| index`. `0xFFFF` means none. Micro-op temporaries keep their own indices. |
| S1 `dst_value` | The 64-bit result written to the first integer destination. 0 if there is none. |
| S2, S3 `size_flags` | Bits 0–7 access size in bytes. Bit 8 the access spans two lines. Bit 9 the value is truncated to its first 8 bytes. Bit 10 no data was read, as for a software prefetch. |
| S2, S3 `value` | The bytes read or written, little-endian, zero-extended. |
| S4 `service` | Bit 0 the data came back from the cache. Clear means a store forwarded it. The level that served a load, first use of a prefetched line and merging into a prefetch in flight are derived from S6 through `seq`. |
| S6 `code` | Bits 0–3 level: 1 L1D, 2 L2, 3 LLC. Bits 4–7 event: 1 hit, 2 miss, 4 fill. Bit 8 a hit that is the first use of a prefetched line. Bit 9 the packet is a hardware prefetch, or for a fill, answers one. Bit 10 write. Bit 11 instruction fetch. Bit 12 software prefetch. Bit 13 an eviction or writeback announced by the level above, which fires that level's hit or miss probe like any request. Bit 14 an upgrade request from a store, which gem5 counts apart from demand accesses. Bits 16–31 gem5's requestor id. An ordinary L1D hit is not recorded: a load with no L1D row hit there. |
| S6 `who`, `pc` | The `seq` and PC of the instruction behind the request. Zero when the request carries none, as most prefetches do. |
| S7 `code` | Bits 0–7 unit: 1–15 conventional prefetchers in configuration order, 16 traversal unit. Bits 8–15 stage: 1 identified, 2 dropped, 3 queued, 4 issued, 5 filled, 6 first use, 7 evicted unused, 8 unresolved at the end of the region. Bits 16–23 reason. For a drop: 1 in cache, 2 in flight, 3 already queued, 4 queue full, 5 page crossing, 6 translation failed, 7 budget. For a first use: 1 timely, 2 late. |
| S7 `line` | Line number. Bit 63 is set when it is physical. |
| S10 `occupancy` | One byte each: L2 in-flight misses, LLC in-flight misses, L2 prefetch queue, LLC prefetch queue, LLC write buffer, memory-controller read queue. |

`t_issue` is the first tick at which the load left the issue queue. `t_data` is the tick at which its data returned, from the cache or from a forwarding store. `t_head` is the tick at which the load became the oldest micro-op in the reorder buffer, which is the commit tick of the micro-op before it. Stall weight is derived, not stored: `max(0, t_data − t_head)`, the time the load sat at the reorder-buffer head before its data returned. Only one micro-op is at the head at a time, so these intervals do not overlap.

Each run folder also holds two small files. `capture/capture_counts.json` has the row count of every stream, the committed instructions, micro-ops, loads and stores, the ticks at which the window opened and closed, and hashes of the load stream. `m5out/run_info.json` has the binary and its arguments, the checkpoint, the arm and its prefetchers, the signature-path class used, the warm-up and measured instruction counts, the clock period, and the cache and core sizes. `gem5.log` records the gem5 version and build time. Validation results are not written per run: they are the tables `measurement/tables/validation*.csv`.

Captures written before the rebuild of 23 November 2026 differ in two details. They hash load addresses and values together in one field, `v1_load_hash`, where later ones have `v1_address_hash` and `load_value_hash`. And they do not set bit 14 of the S6 code. The analysis reads both.

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

Identifiers below were read in the gem5 source at stable release 26.0.0.0, commit [`9d36889`](https://github.com/gem5/gem5/tree/9d368896ce119c17e1289dcbb18f33624724b843), on 22 November 2026. The hooks for S1 to S4, S6 and S8 have since been built and run, and the traps at the end of this section include what running them showed. Use 25.1 or later: the 23.0, 24.1 and 25.0 releases assign the fetch and commit ticks only in tracing builds with a debug flag set.

| Stream | Hook | Upstream today, or patch |
|---|---|---|
| S1 `uop` | The `"Commit"` probe on the O3 CPU, a `ProbePointArg<DynInstPtr>` that fires per committed micro-op. From the instruction: `pcState()`, `opClass()`, the class flags, `srcRegIdx(i)`, `destRegIdx(i)`. The result is read with `o3::CPU::getReg(renamedDestIdx(0), tid)` | Listener only |
| S2 `load` | The same probe: `effAddr`, `effSize` and `memData`. For a load, `memData` holds the bytes once data has returned or been forwarded | Listener only |
| S3 `store` | `"Commit"` for the key and address. The value comes from `"DataAccessComplete"`, a `ProbePointArg<std::pair<DynInstPtr, PacketPtr>>`, at the store's writeback after commit, because a store's `memData` is still null at commit | Listener only |
| S4 `load_time` | `fetchTick`, `renameTick`, `commitTick` and `firstIssue` on the instruction, and `physEffAddr`. `t_data` is the tick of `"DataAccessComplete"`, or of `"ToCommit"` for a forwarded load, which never fires the former. `t_head` is kept by the listener. `service` is joined from S6 by `seq` | Listener only |
| S5 `wrong_path_load` | `"Execute"`, which fires before the squash check, with `"Squash"` and `"Commit"` to settle each load's fate | Listener only. `"Squash"` covers instructions that reached the reorder buffer |
| S6 `cache` | `"Hit"`, `"Miss"` and `"Fill"` on each classic cache, each carrying a `CacheAccessProbeArg` with the packet and a cache accessor. `"Data Update"` with empty `newData` marks an eviction. The demanding load is `pkt->req->getReqInstSeqNum()`. First use of a prefetched line is `hasBeenPrefetched(...)` on the accessor, read inside the `"Hit"` listener before the cache clears the mark | Listener only. The probes exist in timing mode only. A demand that merges into an in-flight request has no probe, so a late prefetch is derived from the next fill of the line |
| S7 `prefetch` | Candidate, drop and issue events inside `prefetch::Queued` (`notify`, `insert`, `getPacket`) and the drops in `BaseCache::getNextQueueEntry`. First use comes from S6. Unused eviction is taken where the cache invalidates a block, because the `hwPrefetched` field of `"Data Update"` is not set reliably on invalidations | Small patch: new probe points. Upstream exposes these events as counters only |
| S8 `fill_content` | The `"Fill"` probe: the response packet carries the whole line. `"Data Update"` cannot be used, see the traps below | Listener only |
| S9 `dtlb` | No probe was identified | Patch, or defer to Stage 4 |
| S10 `pressure` | In-flight miss and queue occupancy are not exposed through a probe argument | Small patch |

Source files for these hooks at that commit: [`src/cpu/o3/commit.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/commit.cc) · [`src/cpu/o3/lsq_unit.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/lsq_unit.cc) · [`src/cpu/o3/dyn_inst.hh`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/cpu/o3/dyn_inst.hh) · [`src/mem/cache/base.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/base.cc) · [`src/mem/cache/cache_probe_arg.hh`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/cache_probe_arg.hh) · [`src/mem/cache/prefetch/queued.cc`](https://github.com/gem5/gem5/blob/9d368896ce119c17e1289dcbb18f33624724b843/src/mem/cache/prefetch/queued.cc)

Traps found while reading the source and while running it:

- **Listener types.** A listener's argument type must equal the probe point's exactly, or gem5 stops with a "Wrong type of listener" panic.
- **`"Data Update"` does not carry the line.** It builds its `oldData` and `newData` vectors from the first eight bytes of the block, one byte per element, so seven eighths of the line are missing. Line contents come from the `"Fill"` packet.
- **A demand that merges at a cache has no deeper event of its own.** The request that opened the miss owns the events at the next level. A late prefetch is found from the fill instead: a demand miss is late when the next fill of its line at that level answers a prefetch.
- **`completeTick` is not data return.** On a load it marks when the access was initiated. `issueTick` is assigned only in tracing builds. This capture uses `firstIssue` and the probe ticks named above.
- **gem5's "late" is not ours.** The statistic `pfLate` counts prefetches dropped at issue because the line was already present or in flight. In this project a late prefetch is one that a demand merges into while it is still in flight. gem5 does not count such a demand in `pfUseful` and does not mark the block as prefetched. This capture finds late prefetches from the fill that follows the demand miss.
- **A prefetch looks like a read everywhere but where it was issued.** The issuing cache holds the `HardPFReq` packet. What it sends down is a plain read on the same request, and the fill is a plain response. The capture recognises prefetch traffic by the task id `Prefetcher` that gem5 puts on the request.
- **A prefetcher is starved when demand fills the miss registers.** A cache issues a prefetch only when it has a free miss register beyond its demand reserve. A loop that keeps every L1D miss register busy leaves its L1D prefetcher issuing nothing.
- **Hits are hidden from prefetchers by default.** A prefetcher sees ordinary hits only with `prefetch_on_access=True` and `prefetch_on_pf_hit=False`.
- **No data at a read miss.** `PrefetchInfo` carries data on read hits and on writes, and never on read misses. A traversal unit therefore reads lines on the fill path.
- **Classic caches only.** The hit, miss and fill probes fire only in the classic cache model in timing mode. The stock GAPBS, NPB and SPEC scripts use Ruby caches, so this capture needs its own configuration with classic caches and the O3 CPU.
- **Which level served a load.** `Request::getAccessDepth()` exists, but no caller that increments it was found. `service` is therefore derived from S6.
- **ElasticTrace is a reference, not a substitute.** It tracks producer-to-consumer register dependences at rename through the same probes, but records no values and no register identities, and it skips split requests. Its rename bookkeeping is the model for the Stage 3 dependence tag.
- **The upstream indirect-memory prefetcher does not run.** With `prefetch_on_access=True` gem5 26.0 stops with `PrefetchInfo::get called with a request with no data` at the first index it reads. A read hit is still a request when the prefetcher is notified, the request command carries no data flag, and since this release the notification then holds no data. Read by inspection, the prefetcher's generation loop also pushes the address of the current index on every iteration instead of looking ahead. It is not a usable baseline as shipped.
- **The upstream signature-path prefetcher can walk without end.** Its lookahead continues while a confidence stays above a threshold. Version 2 multiplies the confidence at each step by useful over issued prefetches and by a counter ratio. When both are 1 the confidence never falls, the walk never ends, the simulator stops advancing and its memory grows by tens of megabytes per second. Seen once in 16 GAP windows, during warm-up. `np_ext` adds a subclass that stops a walk after 1,024 steps.
- **Evictions look like requests one level down.** A writeback or clean eviction from L1D fires L2's hit or miss probe. The capture marks them with bit 13.
- **A store's permission request is not a demand access.** gem5 counts an upgrade request apart from demand hits and misses. The capture marks it with bit 14.
- **Some packet queries assert on responses.** `Packet::needsWritable()` may only be called on a request. A fill listener sees responses.
- **The host C library misreads gem5's processor.** gem5's x86 CPUID reports no cache sizes. The C library then computes a non-temporal threshold of zero, and its `memcpy` takes the large-copy path for small copies and reads far past the source, into unmapped memory. The workloads run with `GLIBC_TUNABLES` that state the cache sizes and the threshold.
- **A program that reads a clock is deterministic in addresses, not in values.** GAP's kernels time their phases. Arms with different timing read different clock values, with the same loads at the same addresses. V1 therefore compares addresses, and values are reported apart.

## Labels and the streams they need

Definitions are in [section 4 of the memo](README.md#4-prediction-task-and-labels).

| Label | Built from |
|---|---|
| L0 join key | S2 row index |
| L1 future footprint | S2, plus S4 and S6 of the `nopf` and `fast` arms |
| L2 dependence edges | S1, S2, S3 |
| L3 edge timing | S4, S6 and S8 of the reference arm |
| L4 plan label, L5 gate label | L1, L2, L3 |

The dependence slice walks S1 backward from a consumer load's address registers. It follows register copies, immediate adds and shifts, and store-to-load pairs matched through S3 and S2, and stops at the nearest producer loads or at a window bound. Shift and base are then fitted numerically from the recorded values: for a candidate producer value *v* and consumer address *a*, the edge is exact if `a − (v << s)` is the same constant across instances of the same pair of PCs, for a shift *s* of 0 to 3. A load that no such form explains is tested against the word of a bit vector that holds bit *v*: `a − ((v >> k) << (k − 3))` constant, for words of 64, 32 and 8 bits. A relation needs at least 16 instances with one base, a tenth of the pair's instances, and four distinct addresses. Because the fit uses values, the capture does not need each instruction's immediate operands.

## Validation

| Check | Passes when |
|---|---|
| V1 Architectural determinism | Every arm and every repeated run commits the same loads at the same addresses, row by row. Load values are hashed separately and reported: a program that reads a clock inside the window gets different values in different arms, with identical addresses and control flow. |
| V2 Non-intrusion | With the capture off and on, every gem5 statistic is equal. Run on the validation microbenchmarks. |
| V3 Known structure | On the validation microbenchmarks the extracted L2 edges equal the constructed ones, and the stride array yields none. |
| V4 Reconciliation | Stream counts equal gem5's own statistics: committed instructions, and requests that miss at each level. Hits at L2 and the LLC are compared where the capture marks upgrade requests. The statistics window is cut at cycle boundaries and runs 2,000 instructions longer than the capture window, so counts may differ by that much. Prefetch counts wait for S7. |
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

- Row sizes in bytes: S1 40, S2 40, S3 40, S4 72, S5 40, S6 48, S7 48, S8 88, S9 32, S10 16.
- Windows in the measurement study: 5M committed instructions of warm-up, then 20M measured, two windows per program and input. A 20M-instruction GAP window holds 3.2M to 8.3M committed loads.
- Measured on the 16 GAP windows: the arm that holds every stream takes 0.20 to 0.75 GB compressed per window, 5.4 GB in all for 39.6 GB of rows, a compression of 5.2 to 10.5 times. An arm without the architectural streams takes 0.13 to 0.62 GB. The dependence edges derived from S1 to S3 add 80 bytes per load, uncompressed.
- gem5 simulated 16,000 to 42,000 instructions per second per run on the shared server (median 28,000), so a window takes 10 to 25 minutes. On the validation microbenchmarks recording cost 7% to 25% of simulation speed.
- S1 is needed only for the dependence slice and can be dropped once the edges are extracted.
