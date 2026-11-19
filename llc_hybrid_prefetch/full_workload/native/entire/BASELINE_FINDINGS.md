# Conventional performance on the registered recordings

Each row reports one uninterrupted execution per policy after 50M instructions of demand-only warmup. All cache and predictor state carries across the four measured quarters. The attribution table selects the lowest-cycle arm among policies with at least one accepted request on that recording. Independently choosing a policy on future workloads would require additional evidence.

## Fixed-work time

| Recording | Measured instructions | NoPF cycles | Stride time saved | Multi-delta time saved | SPP time saved | Temporal time saved |
|---|---:|---:|---:|---:|---:|---:|
| GAP PageRank-3 through EOF | 450,000,001 | 10,085,429,754 | -1.03% | -0.96% | -1.30% | 0.00% |

A positive percentage means that the identical measured instruction stream retired sooner than with NoPF. The CSV retains exact cycles, IPC, L2 and LLC MPKI, tag hit and miss rates, first-use counts, pressure and DRAM bytes for every arm. [Full performance](performance.csv) · [Quarter boundaries and cycles](quarters.csv)

## First-use attribution and traffic of the fastest issuing arm

| Recording | Fastest | NoPF LLC misses | Direct timely coverage | Matched late merges | Useful accuracy | ROI cycles waiting at ROB head | Total DRAM bytes / NoPF |
|---|---|---:|---:|---:|---:|---:|---:|
| GAP PageRank-3 through EOF | Multi-delta | 35,499,013 | 12.47% | 1,221,496 | 85.85% | 87.58% | 1.065× |

Direct timely coverage uses a matched NoPF miss and the first demand use of a completed prefetch. A late merge still waits for data. Head-wait fraction is the fraction of ROI cycles with a qualifying in-flight demand at the ROB head. It is a stall proxy. A cycle reduction from one arm includes its effects on every request and cache occupant. [Matched lookup and first-use outcomes](transitions.csv) · [Sampled lead and service](sampled_timeliness.csv) · [Head-wait distributions](sampled_exposure.csv)

## Execution-time changes across consecutive quarters

Each row keeps the same issuing arm selected on the whole interval above. Each cell compares its retirement time to NoPF for the same consecutive dynamic instructions. All four quarters run in sequence with live processor and cache state.

| Recording | Arm | Quarter 1 time saved | Quarter 2 | Quarter 3 | Quarter 4 |
|---|---|---:|---:|---:|---:|
| GAP PageRank-3 through EOF | Multi-delta | -1.07% | -0.85% | -0.82% | -1.09% |

[All quarter counts and program ordinals](quarters.csv)

## What the four predictors cover together in hindsight

| Recording | Timely under any arm | Timely under none | Late under at least one among those never timely | Plain tag miss under all four |
|---|---:|---:|---:|---:|
| GAP PageRank-3 through EOF | 13.44% | 30,727,577 | 1,249,264 | 27,587,612 |

A late first-use cohort is a candidate for earlier legally issuable address generation. Persistent misses require testing whether an informative cue arrives before the deadline. The four arms produce different memory-system states, so their retrospective union supplies a target census only. The table also distinguishes targets unique to each conventional arm. [Joint target coverage](joint_coverage.csv)

## Scope of the exact-line temporal control

The 256-entry temporal control proposed zero targets throughout the measured intervals in this table. Its policy requires a physical predecessor-line tag to survive and repeat the same successor before it can propose. On the separate recurring nineteen-line conflict fixture, the same native binary issued 592 cross-page requests and provided 591 timely first uses. These results apply to this finite table and trigger interface. Larger temporal metadata stores, PC-conditioned associations and other candidate generators require their own comparison.

On PageRank's four sampled NoPF ingress sequences, 74,963 of 178,028 observations repeated an earlier physical line in the same sample. An unbounded exact-line association table reached confidence for 728 evaluable next observations and matched 65. The 256-entry shadow reached confidence 0 times. These per-sample shadows reset at the start of each island. [Exact-line retention and immediate-successor matches](temporal_support.csv)
