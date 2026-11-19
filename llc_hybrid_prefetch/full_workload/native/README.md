# Full-scale conventional LLC measurements

The principal endpoint is the retirement time of fixed work. For each independently registered source view, one core of the existing two-core processor warms for 50 million instructions and retires 200 million more. All five policies run the same demand-only warmup with their usual observation/training enabled where applicable. Actual prefetch issue starts in the measured interval. The inherited fast warmup enables normal cache and pipeline latencies at the synchronized measurement barrier. The early and far panels of BFS-3 and MCF-472B have separate warmups. A panel is a measured interval of its named complete source recording. Multiple intervals from one recording share program and input provenance.

## Machine and policy controls

The existing native simulator models private 32 KiB L1D and 256 KiB L2 caches, a shared 4 MiB, 16-way LLC with 128 MSHRs and a 64-entry prefetch queue, and a single DDR4-2400 channel. Cache lines are 64 bytes and pages are 4 KiB. Solo full-window runs activate core 0 in physical placement slot 0. The pair experiment uses two active cores and stable per-program slots. The source traces supply virtual addresses. Trace-defined first-touch order determines simulated physical pages in 64-page chunks. The original operating-system page placement is unavailable. For far windows the prefix VPN list reconstructs placement before warmup. No cache, predictor or processor timing state crosses the gap.

| Arm | Information available at the LLC trigger | Candidate and issue rules |
|---|---|---|
| NoPF | Demand cache hierarchy and processor execute normally | No prefetcher issues |
| Stride | Per-core native 64-entry PC stride tracker on admitted LLC demand reads | Two matching consecutive nonzero displacements generate the next stride on the same physical page |
| Multi-delta | Per-core previous delta for the PC and 256 tagged direct-mapped successor entries with two-bit confidence | Use a confident successor delta when available and Stride otherwise. One same-page target |
| SPP | Existing LLC port of Pythia v1.3 SPP, signature and pattern tables with confidence and usefulness feedback | Trace up to four path stages, retain the highest-confidence distinct same-page candidate, degree one |
| Temporal | One tagged 256-entry per-core table mapping a previously observed physical line to its next LLC-visible line with confidence threshold two | Retrieve one remembered physical successor, including another page if previously observed |

The trigger is an admitted demand-data read at the LLC, before its tag lookup. L1/L2 hits, many merged requests, instruction fetches and writes do not create prediction decisions. All issuing arms use the shared tag, MSHR, read/write/prefetch queue, outstanding and capacity checks. Degree one limits proposals per trigger. Accepted bytes and occupancy can differ. The simulator charges normal queues, DRAM and cache effects. Lookup latency, hardware critical paths, predictor power and area require separate evaluation. Original Pythia SPP at an unrestricted degree and a 256-entry simple temporal table answer different questions from this degree-matched comparison.

At the project’s illustrative 48-bit byte-address/PC width, the 256 tagged multi-delta entries need 4.28 KiB/core (48-bit PC, two 43-bit deltas, two confidence bits and valid). A 256-entry exact-line temporal table needs 2.72 KiB/core (two 42-bit line numbers, confidence and valid). The multi-delta arm additionally needs the PC tracker and Stride fallback. The SPP port includes 256 signature entries, a 512-set four-way pattern table and 1,024 feedback slots. Native runs equalize the trigger and issue degree. These bit counts leave out storage banking, access ports, comparators and physical timing. [Existing conventional controls and storage assumptions](../../opportunity_study/conventional/conventional_baselines.md)

## Reproduction

From the cache root, run the existing two-core simulator builds where their binaries are missing. The project's respective build scripts obtain Pythia v1.3 and libbf v1.0.0. Prepare the input views and full-source validation records following [trace intake](../intake/).

```sh
[ ! -f .runtime/project_env.sh ] || . .runtime/project_env.sh
python3 llc_hybrid_prefetch/sim/build.py --cores 2
python3 llc_hybrid_prefetch/opportunity_study/conventional/build.py
python3 llc_hybrid_prefetch/full_workload/native/build.py
python3 llc_hybrid_prefetch/full_workload/native/batch.py --jobs 6
python3 llc_hybrid_prefetch/full_workload/native/batch.py --jobs 2 --cases bfs_discovery_far mcf17_discovery_far
python3 llc_hybrid_prefetch/full_workload/native/batch.py --jobs 2 --policies temporal --cases bfs_discovery bfs_transfer mcf17_discovery mcf17_transfer mcf06_discovery soplex06_heldout pr_heldout bfs_discovery_far mcf17_discovery_far
python3 llc_hybrid_prefetch/full_workload/native/summarize.py
python3 llc_hybrid_prefetch/full_workload/native/timing.py
python3 llc_hybrid_prefetch/full_workload/native/exposure.py
python3 llc_hybrid_prefetch/full_workload/native/temporal_audit.py
python3 llc_hybrid_prefetch/full_workload/native/describe.py
```

Run the separately registered PageRank complete-recording check and summarize it in its own directory.

```sh
python3 llc_hybrid_prefetch/full_workload/native/entire_recording.py
python3 llc_hybrid_prefetch/full_workload/native/summarize.py --cases pr_heldout_entire --out llc_hybrid_prefetch/full_workload/native/entire
python3 llc_hybrid_prefetch/full_workload/native/timing.py --cases pr_heldout_entire --out llc_hybrid_prefetch/full_workload/native/entire
python3 llc_hybrid_prefetch/full_workload/native/exposure.py --cases pr_heldout_entire --out llc_hybrid_prefetch/full_workload/native/entire
python3 llc_hybrid_prefetch/full_workload/native/temporal_audit.py --cases pr_heldout_entire --out llc_hybrid_prefetch/full_workload/native/entire
python3 llc_hybrid_prefetch/full_workload/native/describe.py --dir llc_hybrid_prefetch/full_workload/native/entire
python3 llc_hybrid_prefetch/full_workload/native/validate_overlap.py
```

The raw run directory contains config.json, the exact native command and environment in complete.json, the simulator log, cores.csv, phases.csv and stats.csv. A run is marked complete only after fixed-work, phase timing and first-use lifecycle checks pass. Repeating an identical command reuses a completed arm. Trace materializations, seed lists and per-instruction streams stay in the raw folder. The full-source origin and the number of skipped records inside a materialized view are distinct settings.

## Alignment and metric definitions

Each arm emits a two-byte little-endian LLC outcome word for every retired instruction in the ROI. In each of four operand nibbles, bit 0 means LLC tag hit, bit 1 tag miss, bit 2 first demand arriving late to a prefetch, and bit 3 a timely first-use claim. The allowed nonempty states are ordinary hit, miss, miss with late claim and hit with timely claim. A zero can mean upper-level service, early merging, a forwarded request or an absent operand. Exact ordinal, core and operand ID determine correspondence between arms. Page-mapping output must match across policies. Full joins reconcile hit, miss, timely and late totals per independent quarter against native counters. The issue decision never consults the alignment stream.

| Quantity | Numerator and denominator |
|---|---|
| IPC and time | Measured instructions divided by elapsed retirement cycles. Time reduction uses the same view's NoPF cycles |
| LLC/L2 tag hit and miss rates | Successful demand tag outcomes divided by completed demand tag lookups at that level |
| MPKI | Demand tag misses per 1,000 retired instructions |
| Useful accuracy | Timely first-use hits plus first late demand merges divided by accepted requests |
| Timeliness | Timely first uses divided by timely plus late first uses |
| Timely coverage | Original NoPF misses receiving a directly attributed first prefetched hit on the identical dynamic operand divided by all original NoPF LLC tag misses |
| Inclusive coverage | Timely coverage numerator plus original misses with a first late merge, divided by original misses |
| Net miss coverage | Difference in aggregate NoPF and policy tag-miss counts divided by original NoPF tag misses. Ordinary residency changes and different tag-lookup visibility contribute |
| Pressure | LLC MSHR and DRAM read/write queue occupancies averaged across all measured cycles. DRAM bytes count completed transfers by traffic class |
| Exposed wait | Union of cycles with an unfinished, fetched-in-flight load at retirement head. One cycle counts once per core. This diagnostic proxy cannot isolate the marginal performance cost of a given miss |

LLC and L2 MPKI count data-load tag misses. DRAM demand bytes include non-prefetch reads at the memory channel, which can also carry instruction fetches and ownership reads. Blank ratios denote a zero denominator. Proposals, acceptances, timely and late first uses have different denominators. A non-prefetched ordinary tag hit on a former NoPF miss is recorded separately from direct coverage. A late merge may still save service time. Each core/quarter's cycle endpoint is the retirement of its final instruction. Occupancy and DRAM totals cover the full measured execution.

## History windows and tests

Four fine-grained windows were fixed before baseline measurement. They cover the first 250,000 dynamic instructions in each consecutive 50M measured quarter. One million of the 200M measured instructions receive detailed access, queue and head logs. The full 200M uses native counters and dynamic identity for every operand. Request lifecycle rows are retained when their trigger is inside a sampled window or a first-use event identifies an in-window consumer. A trigger outside a window can yield an in-window demand with only a terminal record. Requests proposed inside a window retain subsequent service and terminal events even after the window ends. Other incomplete request timelines are censored. Histories reset for analyses of isolated samples, while all caches and predictors keep state across them. The [continuous arrival-order audit](../continuous/FINDINGS.md) replays BFS-3 and Soplex with every Stride demand admission exported. It measures how much the island rates and program-order predecessor relationships differ from the full arrival sequence.

```sh
# Prepare the earlier 994B compatibility fixture if running in a new checkout.
python3 -B llc_hybrid_prefetch/experiments/prepare_traces.py
python3 llc_hybrid_prefetch/full_workload/native/validate.py
python3 llc_hybrid_prefetch/full_workload/native/validate_placement.py
python3 llc_hybrid_prefetch/full_workload/native/validate_slots.py
python3 -B llc_hybrid_prefetch/opportunity_study/conventional/temporal_fixture.py
python3 llc_hybrid_prefetch/full_workload/native/validate.py --temporal-fixture
```

The first test compares all four main policies against the pre-existing native implementation on a 350K-instruction MCF validation span. Core cycles, lifecycle counters, complete demand/candidate/head logs and physical maps agree byte for byte. Every tag outcome and first-use bit agrees with the independent reference CSV join. The second test checks original-source skip against a first-touch-seeded far materialization and verifies identical source ordinals, physical addresses and sidecars. A two-core directed fixture places 2,000 independently checked tag outcomes in each of the eight core/operand slots. The optional temporal control reuses the existing nineteen-line conflict fixture produced by `opportunity_study/conventional/temporal_fixture.py`. It reproduces 592 accepted cross-page requests, 591 timely claims and the previously recorded retirement time using the full-workload binary. See [instrumentation parity](parity.csv), [far-view equivalence](placement_validation.txt), [multi-operand parity](operand_validation.txt), and [positive temporal control](temporal_validation.csv).

## Output tables

Read the [registered window comparisons](BASELINE_FINDINGS.md) and [end-to-EOF PageRank comparison](entire/BASELINE_FINDINGS.md). The [per-view validation](validation_summary.csv) checks exact source bounds, retirement, quarter counts, matched outcome partitions and candidate legality. The [PageRank overlap check](entire/pagerank_overlap.csv) compares every dynamic outcome in the shared prefix, allowing for changed in-flight work immediately before the shorter run stops fetching.

- [Performance](performance.csv) contains full-ROI counts, fixed-work cycles, causal coverage, traffic and pressure.
- [Quarters](quarters.csv) identifies exact global ordinals and time for each measured quarter.
- [Transitions](transitions.csv) is the operand-level 5-by-3 outcome contingency table per quarter. Empty/empty pairs are omitted because some slots do not denote a memory access.
- [Joint conventional coverage](joint_coverage.csv) counts original misses made timely by any one, all four, or a unique issuing policy. Each target is joined by instruction and operand. The union combines outcomes under different executions. It gives a retrospective target count and cannot establish a deployable coordination policy or completion time.
- [Rejection reasons](rejections.csv) reports each observed admission denial on the full interval. Numerators are proposals, including repeated proposals for a resident or outstanding line.
- [Timeliness distributions](sampled_timeliness.csv) follows accepted requests from the four fixed trigger windows. It includes dispatch, fill, demand ingress and consumer lookup/merge timestamps. A positive fill margin means the line filled before the stated demand milestone. The table gives eligible and observed counts because a consumer can fall outside the logged interval. Quantiles use the inverse empirical CDF. A strong difference in lead between timely and late cohorts describes the observed executions and does not identify the causal gain from changing issue time.
- [Sampled head waits](sampled_exposure.csv) pairs identical original-miss operands and their physical targets across arms, then compares retirement-head wait distributions within the timely, late, persistent, ordinary-hit and no-policy-tag cohorts. A recorded zero is retained. These grouped waits characterize exposure under each execution. Fixed-work cycles measure the system-level benefit.
- [Temporal association support](temporal_support.csv) counts recurring exact physical predecessor lines and recurring PC/page-offset contexts in each NoPF logging island. Two causal shadows use the native confidence update on the same observations, once with the original 256-slot direct mapping and once without capacity limits. Both start fresh at each island edge. Repeated PC/offset keys quantify support. Predictive value requires consistent future targets on held-out data. Immediate-successor matches assess address selection before cache legality, duplicates, queue service and performance.
