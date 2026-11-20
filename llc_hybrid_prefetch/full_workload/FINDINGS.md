# What earlier history buys at the LLC

A recent address increment from one load PC sometimes predicts the next physical line of another. In the far MCF-472B window, an online cross-PC rule independently issued 23 requests that became timely first uses among 418 sampled targets chosen for recheck. The same rule produced small gains over a local increment in two MCF windows and slowed the third. Among the seven primary windows where the frozen structural rule ran, it never beat the best conventional policy. [MCF replay](streams/mcf_transfer/README.md) · [Structural rule replay](discovery/FINDINGS.md)

## Baseline performance

Nine matched intervals each use 50M warmup and 200M measured instructions. Time saved is relative to NoPF. The best policy is selected from NoPF, Stride, Multi-delta, degree-one SPP, and a 256-entry temporal table.

| Recording and interval | Original NoPF LLC misses | Best policy | Time saved |
|---|---:|---|---:|
| SPEC06 MCF-22B | 4,165,259 | Multi-delta | 16.64% |
| SPEC06 Soplex-92B, held out | 2,003,166 | SPP | 0.92% |
| SPEC17 MCF-472B, start | 3,067,839 | SPP | 13.09% |
| SPEC17 MCF-472B, end | 2,256,312 | SPP | 14.90% |
| SPEC17 MCF-1644B | 5,009,921 | Multi-delta | 11.71% |
| GAP BFS-3, start | 2,244,917 | SPP | 16.95% |
| GAP BFS-3, end | 2,409,071 | SPP | 19.66% |
| GAP BFS-14 | 2,452,070 | SPP | 19.83% |
| GAP PageRank-3, held out | 15,678,522 | NoPF and Temporal | 0.00% |

SPP shortened the first MCF-472B window by 13.09% even though LLC MPKI rose from 15.339 to 15.615. Under SPP, 1,448,604 original misses merged into a pending prefetch. In the sampled events, the median trigger-to-demand lead was 1,328 cycles for timely SPP uses and 58 cycles for late ones. Median dispatch-to-fill service was 190 cycles for both. IPC gives the same ranking as fixed-work cycles for these solo windows. In the two-active-core diagnostic, Stride made BFS finish 20,378,618 cycles sooner and MCF finish 4,381,789 cycles later than paired NoPF. [All metrics](native/performance.csv) · [Sampled timing](native/sampled_timeliness.csv) · [Two-core results](interference/README.md)

![Original-miss timely coverage and fixed-work time saved](figures/coverage_and_cycles.svg)

## Complete PageRank recording

The first 50M of 500,000,001 instructions warm the machine. The remaining 450,000,001 contain 35,499,013 original LLC misses. The prefix shared with the primary 200M window matches every dynamic operand outcome for each of the five baseline policies.

| Policy | Fixed-work cycles | More time than NoPF |
|---|---:|---:|
| NoPF | 10,085,429,754 | 0.0000% |
| Temporal | 10,085,429,754 | 0.0000% |
| Stride | 10,188,897,972 | 1.0259% |
| Multi-delta | 10,182,387,967 | 0.9614% |
| SPP | 10,216,578,416 | 1.3004% |
| Frozen structural rule + Stride | 10,188,171,676 | 1.0187% |

The frozen rule preserved 4,211,131 timely first uses from Stride, added 41,395 (33,908 previously late), and lost 124,178. It saved 726,296 cycles against Stride and took 102,741,922 more than NoPF. The separate two-stride lookahead test on the 200M PageRank window added a net 449,489 timely uses and increased cycles by 0.235% relative to Stride. [Full-recording baselines](native/entire/BASELINE_FINDINGS.md) · [Overlap check](native/entire/pagerank_overlap.csv) · [Operand outcomes](discovery/full_complete_load_alignment.csv) · [Lookahead](long_lookahead/README.md)

## Two address predictions that can be checked load by load

### MCF-472B: another PC supplies a signed increment

The four logged samples in the far window contain 418 original misses that met a training-derived 130-cycle lead threshold, with the candidate constrained to the current physical page and the next 16 LLC arrivals. The same-PC rule, fixed increments ±1, ±2 and ±4, and the four conventional controls did not provide equivalent coverage. The equivalent counts were 51 at the start of 472B and one in 1644B. For the 418 targets, the earliest matching increments were −3 for 113, −7 for 111, and −10 for 96. A table using those three increments could cover at least 320 of the targets if it chose the right value and trigger. [Sampled targets](streams/mcf_arithmetic_targets.csv) · [Denominators and controls](streams/arithmetic.csv)

In the online replay, PC `0x402dcb` moved from physical line 57,454 to 57,451. A later PC at source instruction 1,800,071,809 applied that −3 to line 53,187 and issued line 53,184.

| Request event | Cycle |
|---|---:|
| Candidate generated | 28,669,229 |
| Dispatched | 28,669,249 |
| Filled | 28,669,452 |
| Demand reached the LLC | 28,669,479 |

The fill beat demand by 27 cycles. In the same 418-target cohort, 23 targets got an independently issued timely request and one got an ordinary Stride request. The local-increment replay got zero timely uses. Over the whole 200M window, cross-PC took 573,910,934 cycles, saving 0.0755% versus local increment and 0.7972% versus Stride. SPP took 517,754,410 cycles. Cross-PC saved 0.1506% versus local at the start of 472B and cost 0.6231% in 1644B. The independent rule made three of the 51 sampled targets timely at the start and converted the one sampled target in 1644B. [Full MCF evidence](streams/mcf_transfer/README.md) · [Individual target records](streams/mcf_transfer/target_details.csv)

### BFS-3: an earlier offer arrives on time

At source instruction 100,100,888, line 3,500,603 reached the LLC. A frozen rule using producer-PC equality, page offset and local deltas generated line 3,500,606 in cycle 240,298,184. It dispatched at 240,298,204, filled at 240,298,394 and was first referenced at instruction 100,101,506, cycle 240,300,579. A scan from source start checked all earlier loads and stores. The trigger led demand by 2,395 cycles and 618 dynamic instructions, exceeding the 256-entry ROB. Stride, Multi-delta and SPP each requested the same line late. A fixed +3 rule under the same grant also reached it on time. [Request log extracts](discovery/full_native_examples.csv) · [First-reference audit](streams/first_reference_examples.csv) · [Fixed +3 control](discovery/bfs_fixed3_witness.csv)

Replacing only that trigger's usual offer, line 3,500,604, with line 3,500,606 made the target timely. The original offer also had a timely use. Stride saved 20,856 cycles over 200M instructions and SPP took 63,240 more. The target's NoPF ROB-head wait was 150 cycles. The measured wait was already zero under both conventional policies and stayed zero after the replacement. [Single-address replay](single_address/README.md) · [Paired results](single_address/results.csv)

## Which results transfer?

- In the full BFS-3 window, 76 of 93 Stride load PCs recorded zero timely first uses. They accounted for only 3,112 of 2,244,917 original misses. Across continuous LLC arrivals, the share of observed successors after a timely use that were late or plain misses was 42.87% in BFS-3 and 53.64% in Soplex. Four short logging windows gave 63.02% and 28.92%, respectively. These are measured outcome sequences. Identifying a source-code loop boundary takes additional evidence. [Per-PC counts](streams/group_counts.csv) · [Continuous arrival-order study](continuous/FINDINGS.md)
- MCF Stride timely coverage went from 1.34% at the start of 472B to 33.10% in 1644B. SPP led both 472B windows and Multi-delta led 1644B. SPP covered 67.30% of original misses in BFS-3 and 69.28% in BFS-14. [Baseline metrics](native/performance.csv)
- In held-out PageRank, 80,763 of 177,784 sampled anchors belonged to local contexts with at least two plausible displacements. Adding ordered PC identities left held-out loss at 3.270 bits per anchor. In BFS-14, producer equality improved the local model from 5.571 to 5.425 bits. A simpler table using page offset and same-PC deltas without absolute PC identity reached 4.148 bits. These tests report address probability. The full replays establish deadline and performance outcomes. [Chronological evaluation](discovery/FINDINGS.md) · [Per-phase data](discovery/phase_address_information.csv)
- With a common issue grant in four 1M-instruction diagnostics, cross-PC arithmetic beat Stride in BFS-3 and BFS-14 but trailed it in PageRank and Soplex. The same-PC rule finished sooner than the cross-PC rule in all four windows. [Arithmetic controls](streams/address_relations.md)

## Research questions supported by the measurements

| Measured behavior | Address prediction question | First control and required outcome |
|---|---|---|
| A cross-PC increment can name a line before demand in MCF and BFS | Which earlier cue can select the correct increment when local predictions disagree? | Compare a frozen cue rule with fixed increments and a capacity-matched local table on unseen inputs. Charge for lookup latency, wrong offers, and replacements. |
| MCF often receives a correct request late | At what earlier event can the same legal address be generated? | Match against multi-delta, SPP and fixed lookahead. Require measured timely fills and a reduction in fixed-work cycles. |
| Held-out PageRank and Soplex expose different replacement costs | Which candidate improves execution when requests compete? | Compare equal issue budgets and report timely gains and losses, queue occupancy and DRAM traffic. |
| Cross-page reach requires a known physical destination | Which measured context supports a stable target across layouts? | Repeat with independent placement and inputs. Report legality, prediction quality and usable lead separately. |

The measured evidence establishes that earlier LLC-visible information can yield useful addresses, including a previously unseen physical line. Generalizable performance gains over strong conventional policies remain open. [Figures and denominators](figures/README.md) · [Traces and raw runs](data/README.md)
