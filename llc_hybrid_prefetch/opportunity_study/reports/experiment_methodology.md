# Experiments, instrumentation and validation

## Baseline audit

The parent project is the reference. We verified its 45 published project files, ran its unit/short native tests, and revalidated the 48 recorded runs and 62 per-core page maps before adding this study. All new builds and runs use separate directories. [Original method](../../README.md) · [Original measurements](../../MEASUREMENTS.md) · [Simulator implementation](../../sim/hybrid.cc)

| Design question | What the reference actually implements | Consequence for inference |
|---|---|---|
| Which events reach the predictor? | The shared LLC calls the policy at admitted demand-load ingress from L2, before resolving the current LLC tag. Private hits, many merged requests, instruction fetches, writebacks and prefetches supply no such trigger. The LLC can see both eventual hits and misses. | Define history in admitted ingress events; keep instruction ordinals separately. A source-level cue hidden by a private hit is unavailable here. |
| What addresses and history exist? | Each core has its own native 64-entry PC Stride tracker. Two identical nonzero physical line deltas produce one next-stride proposal. Enabled Stride trains during demand-only warmup; issue starts at ROI. A line is 64 B and a page 4 KiB. The baseline rejects a different-page target. A stable per-program placement slot maps virtual pages by program-order discovery into 64-page physical chunks. | Only PC, the current physical line and already updated hardware state can supply current inputs. Cross-run identity uses the virtual line as a check. Virtual adjacency cannot grant physical reach. A second frame permutation tests placement sensitivity. |
| How are duplicates and timing represented? | One shared issuer checks tag residency, MSHR, read/write/prefetch queues, outstanding addresses and PQ capacity. It counts an acceptance only after actual enqueue. Queueing, arbitration, fill, demand merge and eviction execute in native simulator order. | Two proposed candidates can denote one physical transaction. Separate candidate provenance and lifecycle. The baseline charges no extra timing for predictor lookups or duplicate-check ports. |
| What measures service and criticality? | Per-core fixed retirement, a shared 4 MiB 16-way LLC, 128 LLC MSHRs and one shared DRAM channel. Warm caches/stride state, request timing and both cores' retirement affect the result. Detailed observers log load stages, cache victim events, prefetch metadata IDs and retirement-head waiting. | Per-core IPC = retired instructions / core cycles. Mix wall time = larger completion time; report each core and bytes independently. Unique head-wait cycles diagnose exposure. Summed per-load waits are not counterfactual runtime. |
| Can runs be joined? | Match (trace recording, core, original instruction ordinal, load operand), check PC and virtual line. Within a run join events using that load identity or prefetch ID and physical line. A lookup can disappear or be serviced upstream. Paired comparisons use identical virtual workloads and placement within a layout. | Different cycles and LLC event indices can refer to the same input load; conversely the same cycle need not correspond to the same work. Missing lookup paths and cache-residency changes receive their own counts. |

## What changed

| Addition | Measurement or control | Test that makes it interpretable |
|---|---|---|
| [New workloads](../workloads/README.md) | Registered synthetic interference, phase and row schedules, a compiled C trace with recorded PCs/operands, two graph recordings alone and a mix, adjacent MCF. Later fixed graph windows, three-level rows, three streams and a separated-cue XOR control were registered after observing the early graph prefixes. | Compare identical dynamic input streams across policies, matching virtual-to-physical maps, known early/masked cues, interval boundaries and fixed retirement. Report early and late windows even if uninteresting. [Initial registration](../workloads/PREREGISTRATION.md) · [Follow-up](../workloads/FOLLOWUP.md) |
| [Deadline and history probes](../history/) | At an earlier real Stride ingress, enumerate same-page, signed physical, recently observed page and temporal-association candidates. Separately ask whether a learned-from-counts rule identifies the target. Sample original misses stratified by paired outcome and exposure. | A current lookup outcome becomes a feature only after lookup; a later target, its head wait, service and peer-run NoPF outcome are labels. Discard anchors whose horizon crosses a split. Compare ordered, shuffled, PC-local, cross-PC and simple fallback tables with the same issue opportunities. |
| [Selective cue replay](../deadlines/protocol.md) | Freeze three equally sized conventional tables. Use either the current context, previous event or latest local stride transition. Replay in the native shared issuer with the original Stride on other triggers. | Chronological 50/20/30 by original work, frozen thresholds, distinct test reports, no-issue native equality, dynamic prefix key validation and per-request first-use attribution. All models see a common capacity ceiling; actual occupancy and numbers of offers may differ. |
| [Native page-indexed history](../page_memory/protocol.md) | Replay three frozen context tables on the later BFS-3 suffix with one grant per eight ingresses. Compare the previous offset on this page, previous page PC plus offset, next-line and repeat-delta, including a finite 16-record store. | A shadow run checks native choices against all 1,460 reference grants. Count timely and late first uses, traffic and changes in completion time. |
| [Physical candidate replay](../physical_replay/protocol.md) | Replay an R1-only context/ranking table on a different CactuBSSN recording. Start from actual ingress in each run. Compare a fixed +128-line control, two page layouts and a 16-entry gate. | Compare R2 no-issue to the reference, both layouts to their own Stride, and Python vs C++ choices across the unmodified stream. The tested R2 workload and the illustrative target were selected for follow-up after the offline analysis. |
| [Conventional and causal controls](../conventional/) | Vary simple lookahead, multi-delta, SPP, tagged associations, admission, delayed scheduling, agreement priority and bounded issue. Intervene on matched high-wait vs zero-wait loads. | Degree-one arbitration throughout the matched sets; open, paced and equal-acceptance comparisons answer different questions. A directed two-core physical alias separately tests simultaneous proposals, outstanding requests and residency. |
| [Hit neighborhoods](hit_neighborhoods.csv) | For four paired cohorts measure the PC, page, outcome and timing of ingress neighbors 1 and 4 events away on both sides. | Report the fractions of neighbor tag results already known at target ingress and at the original prefetch proposal. Follow-on hits describe behavior for the analyst. Later outcomes would leak into an earlier decision. |

The history sweep and selective-cue replay use different splits and objectives. The history sweep asks whether sampled, eligible original misses can be reached with a service-time margin under a fixed grant schedule. The cue replay trains for paired persistent misses, including an earlier-than-Stride condition where the ordinary request merged late. Neither gave its target or head-wait label to the live predictor. Use each experiment's own denominators when comparing results.

## Metric definitions

- **Proposals and acceptance.** Count raw candidates by source and target, rejected reasons, grant collisions and accepted physical transactions. Define issued degree as accepts per trigger; separately count raw suggestions per trigger. A resident tag hit, a pending request and a queue-full denial are different events.
- **Accuracy, timeliness and coverage.** For the original lifecycle, accepted accuracy = (timely first uses + late first merges) / accepted. Timeliness among useful requests = timely / (timely + late). Timely coverage = distinct matched NoPF misses receiving a first prefetched hit / original NoPF demand misses. Inclusive coverage adds matched late merges. Unresolved requests at ROI end remain visible. Report raw counts before ratios and leave zero denominators undefined.
- **Cache and memory.** LLC and L2 demand hit and miss rates each use completed tag lookups at that level; report missing and merged service paths, and misses per thousand retired instructions. Traffic is measured DRAM read + prefetch + write bytes. Measure mean occupancy across cycles and per-decision pressure separately. A prefetch eviction followed by a matched lost hit supplies evidence of pollution; its causal cost needs an intervention.
- **Performance.** Report makespan, time saved (baseline minus test, divided by baseline), each core's fixed-work IPC and time. The reachability ceiling and a service-time proxy do not include an alternative request's actual queue wait, eviction or benefit. Native replay provides that evidence for a specified policy. No suite-wide average is implied by these selected recordings.

The [per-core metrics table](../conventional/paired_metrics.csv) contains IPC, level-specific denominators, demand MPKI, accuracy, timeliness, inclusive/timely coverage and issue degree for every new workload pair.

## Feature availability and hardware path

| Signal | Earliest availability | Implementation cost or ambiguity | Decision tested or justified |
|---|---|---|---|
| Current PC, physical line/offset, previous PC/delta and run length | At the admitted request, after translation | PC metadata transport, per-core state and tags; PC aliases or relocation affect transfer | Candidate direction/region and controlled comparisons between memory representations |
| Past lookup hit, prior Stride outcome/state | At completed tag lookup, or immediately when Stride computes its current candidate | Per-core tagged feedback, outstanding lookups can reorder | Past hit and Stride-state controls against address-only history |
| Prior usefulness, lateness, unused eviction | On first use, merge or eviction, sometimes long after proposal | Source bits and outstanding request association; unresolved at ROI end | Delayed confidence/admission, never same-request future feedback |
| MSHR/queue pressure and MLP | LLC queues can be sampled locally at decision; private-core MLP requires a separate path | Occupied MSHRs mix request types and waiting time; fabric feedback may be stale | Shared pressure gating/priority; compare actual traffic, makespan and companion |
| Retirement waiting and prior load latency | The exact target's head wait/completion is known after service; a per-PC summary of previously resolved loads could be sent later | Tag/hysteresis, update bandwidth, alignment and transport delay. A high wait may overlap or move when timing changes | Matched replay tests whether the signal would justify prioritizing one request over another. A queue occupancy counter or degree cap is a cheaper control. |

## Reproduction

Run from the project directory for the first block and from its new study directory for the second. Use Python with NumPy; the figures also use Matplotlib. A C++14-capable compiler and the baseline build prerequisites are required. The compiled C trace workflow lists its own tracing dependencies and supported platform. The graph acquisition downloads bounded prefixes of public traces. Old runs and all new raw logs belong under their respective ignored raw directories. The tracked CSVs are sufficient to inspect the reported comparisons.

```sh
python3 -B sim/build.py --cores 2 --output raw/build-2
python3 -B sim/check.py --binary raw/build-2/pythia/bin/champsim
python3 -B -m unittest discover -s analysis -p 'test*.py'
python3 -B experiments/validate_runs.py --runs raw/runs --out raw/baseline_page_audit.csv
cd opportunity_study
```

```sh
python3 -B workloads/real_traces.py
python3 -B workloads/generate_controls.py
python3 -B workloads/record_compiled.py
python3 -B workloads/run.py --jobs 4
python3 -B workloads/summarize.py --jobs 4
python3 -B workloads/check_context.py
python3 -B history/extract.py --jobs 4
python3 -B history/extract.py --runs raw/runs --jobs 4
python3 -B history/run.py --jobs 4
python3 -B deadlines/build.py
python3 -B deadlines/run.py --jobs 4
python3 -B deadlines/summarize.py
python3 -B history/export_rule.py
python3 -B history/check_frozen.py
python3 -B physical_replay/build.py
python3 -B physical_replay/run.py
python3 -B physical_replay/summarize.py
python3 -B page_memory/prepare.py
python3 -B page_memory/build.py
python3 -B page_memory/run.py
python3 -B page_memory/summarize.py
python3 -B reports/bfs_timing.py
python3 -B reports/neighborhoods.py
python3 -B reports/figures.py
python3 -B verify_results.py
```

Validation completed on the local machine. The original two alignment tests, three history/causality tests, simulator smoke check and complete 48-run placement audit passed. A no-action page-memory shadow agreed with every reference choice on all 1,460 scored grants. On Sacramento a separately compiled C++14 scoring program reproduced all 4,807 CactuBSSN reference choices for each of the two controls. A native Linux build then reran all ten R2 placement/policy arms: committed core cycles and the full native aggregate counters matched the local run in every case. To check the archived comparison on Sacramento, run `python3 -B verify_results.py --repeat raw/physical_linux` from this study. [Cross-check implementation](../verify_results.py)

The [workload commands](../workloads/README.md) describe how to obtain the compiled-trace dependencies. The [conventional study](../conventional/) gives its native control commands. A reused completed run must have the same inputs and settings; for variants with changed inputs use another named output directory.
