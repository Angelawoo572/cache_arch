# When can an LLC observation identify a useful target?

CactuBSSN supplied the clearest real example of an address that became identifiable substantially earlier than Stride could name it. An earlier demand, a physical displacement of +128 lines, and a previously observed destination page were enough. A native fixed-offset control saved slightly more time than the fitted table on the original placement. A synthetic two-cue control gave the opposite kind of evidence: distinguishing two future targets required retrieving an older event. A small page-indexed table recovered that event.

## Earliest legal cue

Each row below uses **the entire separately held-out recording**. The population is dynamic loads that missed in both policies and had positive retirement-head wait in the Stride run. Draw 48 targets uniformly without replacement from that stratum, independently per recording. Counts are targets with at least one earlier cue meeting the training window's median observed demand dispatch-to-fill latency. Columns overlap.

| Held-out target | Population | Service reference, cycles | Any offset in current page | Different seen page within ±1,024 lines | Any offset in four recently seen pages | Any of 59 enumerated candidates | Actual Stride proposal |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCF 994B, trained on 665B | 43,122 | 81 | 13 | 26 | 9 | 14 | 0 |
| MCF 994B adjacent segment, trained on preceding segment | 14,275 | 190 | 21 | 31 | 21 | 16 | 0 |
| CactuBSSN 3477B, trained on 2421B | 7,077 | 190 | 0 | 38 | 0 | 36 | 0 |
| LBM 3766B, trained on 4268B | 12,913 | 561 | 44 | 1 | 1 | 44 | 0 |
| Xalancbmk 592B, trained on 700B | 1,837 | 190 | 2 | 7 | 22 | 11 | 0 |
| BFS-8 early, trained on BFS-3 early | 1,447 | 70 | 38 | 0 | 0 | 38 | 14 |
| BFS-8 later, trained on BFS-3 later | 298 | 190 | 40 | 0 | 4 | 40 | 2 |

For each target, work backward at most 64 core-local LLC demand ingresses and 4,096 cycles; stop at the preceding demand for that physical line or the held-out split boundary. The earliest qualifying ingress supplies both event and cycle lead. A stricter candidate set can still yield a larger count than the current-page column because it can also produce addresses on other pages. The maximum lead is censored by the two search bounds. The median service reference describes queueing and service in a particular training run. Each detailed target record also carries its own measured dispatch-to-fill time when both timestamps exist.

The 59 action codes span 14 signed current-page offsets, 24 signed displacements to a different previously observed physical page, 16 offsets on four recently observed pages, and five prior-only successor or extrapolation choices. The five use the last successor of an address, PC or PC pair; a per-PC extrapolation; or a PC-pair displacement. Full-page and expanded-arithmetic ceiling rows search wider sets than these issue candidates. No rule sees a virtual page relationship. Cross-page reach assumes the destination physical page can be authorized for this core and address space. The simulator can check previously seen pages without modeling context-switch invalidation or the ports and delay of such a directory. [Definition and examples for each stratum](history_ceiling.csv) · [Every sampled target and its earliest cue](sampled_opportunities.csv.gz)

## Can a causal rule select the address?

For real windows, estimate future-target frequencies on the first 60% of ingresses, check reliability on the next 20%, and test on the final 20%. For synthetic windows, train on 40%, validate on 10% and reserve the final half for test. Transfer tests freeze the source training/validation procedure and evaluate a separate recording or generator variant. We ran 43 chronological and 15 transfer comparisons. The small-working-set control delivered zero measured LLC ingresses, so it contributes no history scores. All scans begin with empty diagnostic history at the measured interval. The simulated Stride tracker itself was warmed.

An answer earns credit when the source's chosen physical line is next touched within the horizon, the matching load originally missed in NoPF, the Stride run has an independently observed lookup for that load, and the elapsed cycles cover the source training service reference. A selected line can be useful to both predictors. The denominator includes ordinary Stride hits and persistent misses. A target counts once even if multiple choices named it. Computation, bandwidth, queue acceptance and eviction require separate live tests.

One grant per eight successive ingresses limits the tables to one chosen address. A deterministic, data-independent varying slot within each block avoids locking the budget to a periodic synthetic pattern. Ranking never inspects future scores, budgets or demands. A second fixed-phase grant control is included. Training uses one data-independent sample per pair of eligible anchors. Anchors near the end of each training, validation or test interval are excluded so labels stay within their own interval.

| Transfer | Grants | Original misses with paired lookup | Current PC + delta, 14 offsets | Same recent key, all candidates | Ordered cross-PC history, 16 | Shuffled cross-PC history, 16 | PC-local history, 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCF 665B → 994B | 15,896 | 80,305 | 64 | 66 | 66 | 66 | 66 |
| MCF 994B → adjacent 994B | 11,267 | 26,697 | 494 | 653 | 615 | 612 | 652 |
| CactuBSSN 2421B → 3477B | 4,795 | 10,149 | 0 | 657 | 657 | 657 | 657 |
| LBM 4268B → 3766B | 3,447 | 27,629 | 3,206 | 3,206 | 3,205 | 3,211 | 3,200 |
| Xalancbmk 700B → 592B | 10,850 | 2,490 | 0 | 0 | 0 | 0 | 0 |
| BFS-3 early → BFS-8 early | 3,164 | 25,372 | 3,080 | 3,080 | 3,080 | 3,080 | 3,080 |
| BFS-3 later → BFS-8 later | 98 | 854 | 24 | 24 | 24 | 24 | 24 |

Three distinctions affect interpretation.

- **Offset vocabulary.** In CactuBSSN, adding the 24 displacements alone took coverage from 0 to 655 targets; the full codebook reached 657. Each of those 657 remained a miss under Stride. In adjacent MCF, adding recent-page choices took the count from 494 to 653. Xalancbmk had sample targets legally reachable via recent pages, yet the frozen frequency rule selected no qualifying target on transfer.
- **Access ordering.** In adjacent MCF, ordered cross-PC windows of 4, 8, 16, 32, and 64 events reached 679, 639, 615, 620, and 646 distinct original misses, respectively. The shuffled 4-event control reached 635. The ordered 4-event result merits inspection, but numerical ordering varied with window and benchmark. The same trace's 16-event PC-local rule reached 652. Each representation reduces each time quarter to coarse delta behavior and whether PC changes dominate. The shuffle preserves the window's underlying tokens and changes their positions with a reproducible per-anchor permutation. A longer history can also lower table support and cause fallback. These comparisons do not test all possible forms of attention. [All windows, fallback support, output families, cache-feedback and Stride-state ablations](history_models.csv)
- **PC identity.** Out of 15,896 MCF 665B → 994B grants, 15,869 had a PC absent from training. Page-relative PC bits did not restore reach. The early BFS recordings each exposed one PC at the LLC, with different full addresses and equal low 12 bits, 0x4a0. The full-PC key had no training support at any grant; its global fallback reached 3,080 targets, including 96 persistent misses. A fixed low-12-bit key reached 3,114, including 105 persistent misses. This is an alias-prone executable-displacement control. The files do not establish the exact code-image mapping or equivalence of every instruction. During early BFS-3 → late BFS-3 transfer all 7,330 granted PCs were unseen; the expanded rule still reached 4,092 targets through fallback. Names alone convey little about the memory stream of a recorded interval.

## Earlier than Stride, with an actual dynamic load

In CactuBSSN 3477B, core 0, instruction **10,145,041**, operand 0, arrived on physical line **72,743** at cycle **5,813,437**. The frozen recent-context table chose +128. Instruction **10,145,357**, operand 0, demanded physical line **72,871** at cycle **5,814,402**, 965 cycles and 16 LLC ingresses later. It missed in both original arms, waited 104 cycles at the retirement head, and took 190 cycles from demand dispatch to fill. Stride proposed no matching target in the bounded interval.

The R1 training key had eight sampled decisions and three qualifying outcomes for this action. The training-support gate requires at least eight observations, three positives and a binomial Wilson lower score of 0.10 or more. The method's support tier also had to achieve at least 10% on at least eight predictions in two of three contiguous validation blocks. Eleven of 24 supported development decisions by this recent-context rule were positive; 60 distinct R2 targets passed the two checks and were correctly selected. These checks describe recurrence and held-out reliability. Serially correlated events do not supply independent trials or per-target guarantees. The sample ceiling and this full-cohort search have different denominators. [Held-out supported cues with instruction IDs](history_verified.csv) · [Other earliest-cue examples](history_examples.csv)

On the original placement the native table issued the example's prefetch at **5,812,784**, filled at **5,812,935**, and the target hit at its rescheduled demand ingress, **5,813,409**, with zero recorded head wait. Fixed +128 also produced a timely hit. Across the complete window, the frozen table saved 35,880 cycles; fixed +128 saved 35,999. Limiting issuer confirmation to 16 recent physical pages reduced the table's saving to 1,784 cycles. The physical remapping sensitivity changes which addresses are close; compare each layout to its own baseline. [Native placement and capacity tests](../physical_replay/placement_results.csv) · [Example replay chronology](../physical_replay/target_example.csv)

The reference backend stores 113 supported R1 context rankings and a global ranking. Packed 6-bit action ranks alone would need about 5,045 bytes, before tags or confidence data. In R2 it saw 339 distinct physical pages and 457 PCs. The diagnostic also has four 256-entry successor/displacement tables and per-PC and per-page history. A 16-page issuer gate bounds confirmation, while the upstream reference generator still has the larger state. No hardware timing or area claim follows from these counts.

## When a nonadjacent event actually carries information

The extension that considers **all 64 offsets on the current physical page** separates output vocabulary from history. Its narrowly defined target is the next eligible same-page access. It uses the same grant schedule. Ground-truth labels enter only after choices are made.

| XOR control, held-out later half | Granted opportunities for the joint target | Current PC/offset | Last two adjacent PCs | Prior same-page offset only | Prior same-page PC and offset |
|---|---:|---:|---:|---:|---:|
| Both cues observable | 55 | 13 | 23 | 23 | **55** |
| First cue masked | 55 | 13 | 23 | 23 | **23** |

For example, instruction 811,538 on line 196,738 selected offset 8 for instruction 811,934 on line 196,744, three ingresses and 1,167 cycles later. The earlier observed load on this page conveyed which target offset to choose. The target missed under Stride and recorded 118 head-wait cycles. Masking the first cue removed that advantage. The earlier 59-code frequency probe reached **zero of 512** joint targets in its intact test half, even though it reached 276 other original misses. Keeping the auxiliary page-offset experiment distinct prevents that output limitation from being mistaken for an absence of context. [Page-association results](page_association.csv) · [Source-labeled examples](page_association_examples.csv)

The nested-loop controls show a different failure. In transfer from the two-row development schedule to the five-row schedule, the recent full rule reached 783 of 7,681 row-body misses and **0 of 511 row starts**. In the multilevel row test half, it reached 473 of 3,840 row-body misses and zero of the 192 subrow, 42 band and 22 slab starts. A large total of correct accesses can therefore conceal a complete transition failure. [Counts by ground-truth target class](history_classes.csv)

### Does the same relationship appear in a real BFS trace?

On the held-out final fifth of the later BFS-3 recording, 1,087 of 1,460 grants had a qualifying next demand on the same physical page. Current PC and offset selected 511, adding the two immediately preceding PCs selected 519, and adding the last offset observed on that page selected **759**. Adding the previous page access's PC also selected 759. The page-offset rule gained 249 decisions and lost one against the current-only rule. The three successive test thirds gained 71, 93 and 85, with losses of zero, one and zero. Among the 249 added matches, 218 had an actual earlier page reference, 195 had intervening ingresses and 111 referred back to a different load PC. A bounded-history absence marker can itself distinguish a context on the other 31 decisions. Of the extra 249 matches, 193 targeted a load that missed in both baseline arms and experienced some ROB-head wait. Repeated target-head cycles across interventions require full simulation to value. [Decision counts and transfer controls](page_memory_stability.csv)

One case starts with instruction **51,608,107**, operand 0, on physical line 191,187. Two accesses at PCs 0x5556e3e3c7d9 and 0x5556e3e3c7e7 intervened before instruction **51,608,215**, operand 0, reached line 191,188, 1,002 cycles after the earlier reference. At this trigger the current-only rule predicted offset 17. Adding the retrieved offset produced offset 21, observed in all 43 qualifying training decisions for this context (43 decisions total). Three ingresses and 684 cycles later, instruction **51,608,294**, operand 0, demanded line 191,189, offset 21. It missed in both baselines, had 118 cycles of operand head wait and received service in 70 dispatch-to-fill cycles; the training median used to define a qualifying lead was 190. The same target was also the next physical line. In the live offset arm, the trigger issued at 12,761,683, the prefetch filled at 12,761,893, and the target hit at 12,762,387 with 48 operand head-wait cycles. Next-line also made it a timely hit. The current-only arm chose line 191,185 and the demand missed, with 238 head-wait cycles. [Earlier-cue evidence](page_memory_witnesses.csv) · [Native request, fill and lookup times](page_memory_native_witness.csv)

Across recordings, later BFS-3 → later BFS-8 gave both choices only two matches on 98 grants, 24 of which had qualifying targets. The earlier BFS transfer gave both choices 3,080 successes out of 3,080 qualifying targets. MCF's adjacent-window transfer did show a second instance of context dependence, with 357 → 465 matches on 11,267 grants and 837 qualifying targets. In a native BFS-3 run, the bounded 64-ingress offset diagnostic saved **48,515 cycles** over Stride across the complete interval, compared with 26,919 for current PC/offset and 46,243 with a 16-entry page table. A same-page next-line control saved **55,552 cycles**, the largest saving among these four. Next-line accounted for 246 of the 249 BFS gains over the current-only rule, and 107 of the 120 corresponding MCF gains. Compared directly on the same grants, the offset rule had four exclusive BFS targets and next-line had 73. In adjacent MCF the corresponding counts were 13 and 26. On later BFS-3 → BFS-8 transfer, next-line reached all 24 qualifying targets. These comparisons favor the fixed next-line rule on the examined recordings. [Native results](../page_memory/results.csv)

The offset table and its 16-entry variant turned **all four BFS targets that next-line could not name** into timely hits from the original frozen cue. NoPF, Stride, next-line, repeating the previous within-page delta, and the tested multi-stride mechanism missed all four. Each target had 48 operand head-wait cycles under both offset arms. The table below shows how many also distinguish access history from the other predictors.

| Target instruction | Previous → current → target page offset | Earlier cue age, ingresses / cycles | Qualifying outcomes at target offset / training observations | Current-only | SPP | Original Stride head wait |
|---|---|---|---|---|---|---|
| 51,681,598 | 50 → 51 → 55 | 2 / 350 | 5 / 26 | Miss | Miss | 118 |
| 51,766,261 | 9 → 10 → 12 | 2 / 684 | 6 / 21 | Miss | Miss | 238 |
| 51,839,379 | 33 → 35 → 54 | 2 / 787 | 0 / 5 | Timely hit | Miss | 238 |
| 51,967,364 | 23 → 24 → 27 | 4 / 749 | 3 / 21 | Miss | Timely hit | 238 |

The first two targets were exposed misses under every tested comparator in this four-load analysis. Their preceding accesses were instructions 51,681,399 and 51,766,031. Repeating the most recent within-page displacement predicted offsets 52 and 11. The stored offset association selected 55 and 12. For the third, five observations fell below the eight-observation threshold, and the current-only fallback predicted the target independently. For the fourth, SPP generated the same line from the original cue. These four loads were selected after comparing the fixed held-out predictions with next-line; they do not estimate a general success rate. SPP and multi-stride per-load reruns reproduced all aggregate counters and core cycles from their original runs. [Cue history and nine-policy load joins](page_exclusive_loads.csv) · [SPP and multi-stride lifecycle evidence](../conventional/bfs_strong_targets.csv)

## Reproduction and visibility

- The event extractor joins NoPF and Stride by core, original instruction and operand, and checks PC and virtual line for observed paired lookups. Candidate generation, recent-page retrieval, address arithmetic and target equality all use physical lines. Per-core order contains other PCs on that core; shared mixed runs retain their original contention. No feature combines another core's private address history.
- The basic key uses current PC, an eight-line group of page offsets and the latest physical delta class. Feedback variants add only the last completed tag outcome strictly before ingress and, separately, the conventional tracker's current ingress state. Both richer-history and fallback predictions are counted. Future hit/miss, service time, ROB-head wait, source labels and paired NoPF outcomes are exclusively offline evaluation or training labels. An actual predictor would need access to the PC at the LLC.
- Run the following from the study directory with Python and NumPy. The `history/` scripts preserve all original run files. The output records include the target, cue, physical address, cue time, actual Stride proposal/first-use timing when present, head-wait exposure and service reference. Cohorts are stratified; summing unweighted 48-target samples across unequal cohorts does not estimate a workload-wide rate.

```sh
python3 history/extract.py
python3 history/extract.py --runs raw/runs
python3 history/test_history.py
python3 history/run.py --jobs 4
python3 history/page_association.py
python3 history/page_witness.py
python3 history/page_exclusive.py
python3 history/export_rule.py
python3 history/check_frozen.py
```

The audit perturbs future addresses, PCs and future lookup outcomes and validates an identical decision prefix. It checks page membership, first-touch labels, preserved shuffled token multisets and chronological label guards. The independent C++14 checker reproduced all 4,807 full-window grant choices for both the frozen rule and the +128 control on the original R2 ingress stream; 4,795 grants remained in the right-censored offline scoring window. [Frozen training table](cactus_r1_rankings.txt) · [Native diagnostic](frozen_rule.h)
