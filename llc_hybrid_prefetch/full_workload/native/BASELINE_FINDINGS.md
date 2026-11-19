# Conventional performance on the registered recordings

Each row reports one uninterrupted execution per policy after 50M instructions of demand-only warmup. All cache and predictor state carries across the four measured quarters. The attribution table selects the lowest-cycle arm among policies with at least one accepted request on that recording. Independently choosing a policy on future workloads would require additional evidence.

## Fixed-work time

| Recording | Measured instructions | NoPF cycles | Stride time saved | Multi-delta time saved | SPP time saved | Temporal time saved |
|---|---:|---:|---:|---:|---:|---:|
| SPEC17 MCF-472B | 200,000,000 | 605,530,483 | 3.66% | 6.59% | 13.09% | 0.00% |
| SPEC17 MCF-1644B | 200,000,000 | 1,484,969,011 | 8.64% | 11.71% | 11.48% | 0.00% |
| GAP BFS-3 | 200,000,000 | 767,261,611 | 8.02% | 11.93% | 16.95% | 0.00% |
| GAP BFS-14 | 200,000,000 | 856,409,664 | 9.98% | 13.97% | 19.83% | 0.00% |
| GAP PageRank-3 | 200,000,000 | 4,452,897,603 | -1.02% | -0.95% | -1.30% | 0.00% |
| SPEC06 MCF-22B | 200,000,000 | 567,687,452 | 12.51% | 16.64% | 14.46% | 0.00% |
| SPEC06 Soplex-92B | 200,000,000 | 543,788,550 | -1.71% | -0.63% | 0.92% | 0.00% |
| GAP BFS-3 far | 200,000,000 | 851,437,656 | 9.37% | 12.74% | 19.66% | 0.00% |
| SPEC17 MCF-472B far | 200,000,000 | 608,412,752 | 4.91% | 7.20% | 14.90% | 0.00% |

A positive percentage means that the identical measured instruction stream retired sooner than with NoPF. The CSV retains exact cycles, IPC, L2 and LLC MPKI, tag hit and miss rates, first-use counts, pressure and DRAM bytes for every arm. [Full performance](performance.csv) · [Quarter boundaries and cycles](quarters.csv)

## First-use attribution and traffic of the fastest issuing arm

| Recording | Fastest | NoPF LLC misses | Direct timely coverage | Matched late merges | Useful accuracy | ROI cycles waiting at ROB head | Total DRAM bytes / NoPF |
|---|---|---:|---:|---:|---:|---:|---:|
| SPEC17 MCF-472B | SPP | 3,067,839 | 10.77% | 1,448,604 | 78.94% | 59.66% | 1.300× |
| SPEC17 MCF-1644B | Multi-delta | 5,009,921 | 40.79% | 2,223,165 | 98.80% | 30.28% | 1.009× |
| GAP BFS-3 | SPP | 2,244,917 | 67.30% | 160,638 | 87.42% | 45.27% | 1.620× |
| GAP BFS-14 | SPP | 2,452,070 | 69.28% | 165,838 | 87.86% | 41.02% | 1.564× |
| GAP PageRank-3 | Multi-delta | 15,678,522 | 12.51% | 550,255 | 85.96% | 87.56% | 1.066× |
| SPEC06 MCF-22B | Multi-delta | 4,165,259 | 7.43% | 107,220 | 89.75% | 66.70% | 0.962× |
| SPEC06 Soplex-92B | SPP | 2,003,166 | 31.54% | 527,124 | 79.83% | 50.46% | 1.523× |
| GAP BFS-3 far | SPP | 2,409,071 | 67.13% | 156,580 | 84.26% | 43.93% | 1.553× |
| SPEC17 MCF-472B far | SPP | 2,256,312 | 15.44% | 349,164 | 71.80% | 59.27% | 1.303× |

Direct timely coverage uses a matched NoPF miss and the first demand use of a completed prefetch. A late merge still waits for data. Head-wait fraction is the fraction of ROI cycles with a qualifying in-flight demand at the ROB head. It is a stall proxy. A cycle reduction from one arm includes its effects on every request and cache occupant. [Matched lookup and first-use outcomes](transitions.csv) · [Sampled lead and service](sampled_timeliness.csv) · [Head-wait distributions](sampled_exposure.csv)

## Execution-time changes across consecutive quarters

Each row keeps the same issuing arm selected on the whole interval above. Each cell compares its retirement time to NoPF for the same consecutive dynamic instructions. All four quarters run in sequence with live processor and cache state.

| Recording | Arm | Quarter 1 time saved | Quarter 2 | Quarter 3 | Quarter 4 |
|---|---|---:|---:|---:|---:|
| SPEC17 MCF-472B | SPP | 13.31% | 13.84% | 13.82% | 11.25% |
| SPEC17 MCF-1644B | Multi-delta | 11.65% | 11.24% | 11.85% | 12.11% |
| GAP BFS-3 | SPP | 18.30% | 19.09% | 12.11% | 18.28% |
| GAP BFS-14 | SPP | 19.98% | 19.42% | 20.56% | 18.98% |
| GAP PageRank-3 | Multi-delta | -1.07% | -1.06% | -0.45% | -1.12% |
| SPEC06 MCF-22B | Multi-delta | 20.06% | 14.45% | 16.62% | 15.49% |
| SPEC06 Soplex-92B | SPP | 4.14% | -2.71% | 4.51% | -3.01% |
| GAP BFS-3 far | SPP | 16.79% | 21.10% | 21.05% | 20.05% |
| SPEC17 MCF-472B far | SPP | 15.48% | 15.17% | 14.35% | 14.64% |

[All quarter counts and program ordinals](quarters.csv)

## What the four predictors cover together in hindsight

| Recording | Timely under any arm | Timely under none | Late under at least one among those never timely | Plain tag miss under all four |
|---|---:|---:|---:|---:|
| SPEC17 MCF-472B | 12.30% | 2,690,400 | 1,479,490 | 389,407 |
| SPEC17 MCF-1644B | 47.42% | 2,634,451 | 1,953,761 | 676,976 |
| GAP BFS-3 | 70.56% | 660,987 | 119,673 | 516,314 |
| GAP BFS-14 | 72.00% | 686,532 | 125,710 | 555,399 |
| GAP PageRank-3 | 13.49% | 13,563,125 | 561,640 | 12,168,169 |
| SPEC06 MCF-22B | 20.45% | 3,313,280 | 1,570,024 | 1,266,717 |
| SPEC06 Soplex-92B | 36.89% | 1,264,172 | 488,430 | 740,986 |
| GAP BFS-3 far | 69.83% | 726,824 | 125,413 | 589,959 |
| SPEC17 MCF-472B far | 17.61% | 1,859,070 | 385,502 | 433,272 |

A late first-use cohort is a candidate for earlier legally issuable address generation. Persistent misses require testing whether an informative cue arrives before the deadline. The four arms produce different memory-system states, so their retrospective union supplies a target census only. The table also distinguishes targets unique to each conventional arm. [Joint target coverage](joint_coverage.csv)

## Scope of the exact-line temporal control

The 256-entry temporal control proposed zero targets throughout the measured intervals in this table. Its policy requires a physical predecessor-line tag to survive and repeat the same successor before it can propose. On the separate recurring nineteen-line conflict fixture, the same native binary issued 592 cross-page requests and provided 591 timely first uses. These results apply to this finite table and trigger interface. Larger temporal metadata stores, PC-conditioned associations and other candidate generators require their own comparison.

On PageRank's four sampled NoPF ingress sequences, 75,289 of 177,848 observations repeated an earlier physical line in the same sample. An unbounded exact-line association table reached confidence for 682 evaluable next observations and matched 67. The 256-entry shadow reached confidence 0 times. These per-sample shadows reset at the start of each island. [Exact-line retention and immediate-successor matches](temporal_support.csv)

## Why timing and MPKI can move in different directions

On MCF-472B, LLC load MPKI was 15.339 with NoPF and 15.615 with SPP. SPP saved 13.09% of fixed-work cycles. It directly converted 330,501 original misses to timely first uses, and 1,448,604 original misses merged with an outstanding prefetch. These counts motivate measuring earlier legal cues and the cost of displaced requests. They do not assign the cycle savings to individual late loads. [Complete MCF and other controls](performance.csv)
