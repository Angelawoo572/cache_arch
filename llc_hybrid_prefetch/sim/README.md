# A–C simulator

The two workloads execute on identical out-of-order cores in the Pythia v1.3 fork of ChampSim. The executable also supports a single active program on the same two-core machine. Both modes retain the 4 MiB, 16-way shared LLC with 128 MSHRs, 64-entry read/write/prefetch queues and one DRAM channel. Per-core private prefetchers are disabled. NoPF emits no proposals. Stride uses the release's 64-entry PC tracker on each requesting core, with a single future same-page line after two equal nonzero line deltas. The issue hook runs at admission of an L2 demand load miss into the LLC read queue, before its LLC tag lookup. [Pythia v1.3](https://github.com/CMU-SAFARI/Pythia/tree/v1.3) · [Original Stride implementation](https://github.com/CMU-SAFARI/Pythia/blob/v1.3/prefetcher/stride.cc)

## Build and run

Run from the new project's root. The build creates its own Pythia and libbf source trees in the requested build directory. An existing local Git clone of the releases can supply the build offline. The build reads the released base files even if that source checkout contains edits.

```sh
python3 sim/build.py --cores 2 --output raw/build-2
python3 sim/check.py --binary raw/build-2/pythia/bin/champsim
```

The matrix runner supplies the following arguments and environment. For a direct invocation, create an empty output directory first.

```sh
mkdir -p raw/example
HYBRID_OUT=raw/example LLC_PF_POLICY=stride HYBRID_COMPRESS=1 \
  HYBRID_ACTIVE_CORES=2 HYBRID_PAGE_SLOTS=0,1 \
  raw/build-2/pythia/bin/champsim \
  --warmup_instructions=10000000 --simulation_instructions=5000000 \
  -traces raw/traces/605.mcf_s-665B.champsimtrace raw/traces/607.cactuBSSN_s-2421B.champsimtrace
```

- `HYBRID_ACTIVE_CORES=1` leaves the second core idle in the same compiled cache/DRAM configuration. Use `HYBRID_PAGE_SLOTS=1` for a solo program assigned slot 1 in its reference mix. Specify one distinct physical allocation slot per active address space in the same order as the traces.
- `HYBRID_SKIPS` gives nonnegative leading record counts per active trace. The program reads those records to reconstruct identical physical placement. Caches, Stride, the TLB and all other timed hardware start at the requested warmup offset. Trace inputs can be raw 64-byte little-endian SPEC/ChampSim records, gzip or xz.
- `HYBRID_PLAN` points to a hindsight intervention CSV with the header `cpu,instr_id,operand,delta`. In a Stride run, a selected admitted anchor replaces its normal choice with one signed line delta. It receives the same candidate limit, eligibility checks and LLC queue timing. Other anchors follow native Stride. This can increase total traffic if the selected anchor was silent in the normal run. The plan is an offline experiment, and plan-matched counts report how many selected anchors survived the change in execution.
- `HYBRID_DETAIL=0` disables detailed event writes while retaining identical transitions, per-core counters and performance statistics. `python3 sim/logging_control.py` compares both settings for MCF alone and MCF with CactuBSSN under both policies.
- `HYBRID_DETAIL=victims` captures a compact LLC lookup file and every cache installation with its displaced physical line, valid bit, source and prefetch ID. `python3 sim/pollution.py` combines those Stride probes with the corresponding full NoPF runs. It verifies equal Stride core cycles, DRAM traffic and pressure before attributing a prior eviction.
- `HYBRID_COMPRESS=1` directly writes gzip event files. `HYBRID_COMPACT=1` omits L1 event rows and can be used for smaller exploratory logs. The full matrix uses the settings recorded by its runner.

## Paired execution

Each core stops admitting instructions at the exact warmup target. The global barrier starts ROI timing after all active cores retire their assigned warmup work. The ROB admissions then stop at the last measured instruction. A faster program finishes without fetching post-ROI work while the remaining program continues. Completion time is the last core's retirement of its final instruction. Traffic includes transfers that occur in this common wall-time interval. Outstanding requests at the endpoint may remain unfinished. Fully warmed caches and Stride history survive the boundary, and conventional issue begins at ROI start. Native Pythia uses accelerated warmup latencies. ROI DRAM counters classify bus transfers as prefetch reads, other reads (including instruction fetches and read-for-ownership), or writes. They cover transfers inside the time interval, including carry-over already in a queue at its start.

The placement map is a simulator-only OS surrogate. It assigns pages using each program's trace order, including instruction, store and load pages, and interleaves 64-page chunks across the allocated slots. It respects physical capacity and preserves a page's line offsets. Physical placement is independent of the two simulated cores' first-touch interleaving. The predictor only receives physical line, requesting PC and core at LLC ingress. The tracer uses virtual lines and dynamic identities to verify runs. The physical arrangement is one controlled layout and does not establish placement robustness.

## Event files

All integer fields are decimal. Load operands use their input trace slot. Instruction identities are one-based per-core trace positions, including skipped instructions. Wall-clock cycles are simulator cycles from run start. Every run records actual retired counts and elapsed cycles separately.

| File | Event and interpretation |
| --- | --- |
| `loads.csv` | `CORE/trace` enumerates all measured load operands at ROB admission. Its physical line is a zero placeholder. `LLC/ingress` is the admitted predictor trigger. Cache `lookup` rows carry a final handled hit (1) or miss (0) at the named level. `merged`, `forwarded` and `full` are read-queue dispositions. `mshr_merge` records coalescence with an existing miss. `full` may repeat on retry. LLC `dispatch` records a primary demand handed to DRAM, and `fill` records installation of a demand-owned response. The cache columns are sampled from the named level. DRAM queue columns correspond to channel zero. |
| `candidates.csv` | A unique proposal ID sticks with its originating core, instruction and operand. Each matched-stride candidate records the physical base and target, including the arithmetic target clipped by the native page check. A silent trigger has target zero. `proposed` ends in `rejected` (with a reason) or `accepted` (a new queue entry). `dispatch` and `fill` are timestamps. Each acceptance receives one terminal classification from `timely`, `late`, `evicted`, `redundant` or `unresolved`. A late merge can fill later. Consumer identities appear only when a data load first uses the request. Use `unresolved/resident` and `unresolved/inflight` separately at ROI end. |
| `head.csv` | Per-operand cycles spent by an unfinished oldest instruction with its load outstanding. Multiple pending operands can overlap. Use the independent unique head-cycle count in `stats.csv` to avoid summing overlapping cycles. A head wait is a retirement-stall proxy. |
| `cores.csv`, `stats.csv` | Per-core warmup work, measured instructions and cycles. Stats hold DRAM read/write bytes transferred during the ROI, cycle-weighted LLC/DRAM pressure, plan-anchor hits and unique head-load wait cycles. A byte count covers an actual bus transfer. An admission, outstanding request and observed transfer count different events. |
| `pages.csv`, `victims.csv` | When present, these record the exact program-to-physical page mapping and each valid line displaced at a prefetch fill. A later matched hit-to-miss can be checked against its victim's identity and order. One eviction is an observation, and establishing lost execution time requires a paired intervention. |

Cycle sums for queue pressure use one sample per simulated cycle from the common start through the last completion. Start-of-ROI occupancies in the updated probe identify residual queue state that can affect traffic near the boundary. Prefetch accept/reject checks use the simulator's tags and queue contents at issue. The code models no extra cycles or ports for those searches. The branch/tag arbitration, physical layout and timing values here belong to the simulator and require separate evaluation for a hardware implementation.

For eviction analysis, a candidate displacement is associated with a loss only when the same dynamic load was a NoPF LLC hit and a Stride LLC miss and the last observed change for that physical line was its displacement by a prefetch, without an intervening installation. The compact probe records installs from load, instruction, read-for-ownership, writeback and prefetch fills. A prefetch overtaken by a demand can fill as demand-owned and is conservatively classified by its actual fill type. The count covers direct matched lookups. Changes from indirect replacement-state or MSHR effects may fall outside it.
