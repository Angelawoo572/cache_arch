# Was the last hit visible when the next prefetch needed to issue?

In early BFS-8 with LBM, **2,742** first late loads immediately followed at least two attributed timely hits in the core's complete LLC ingress order. For every one of them, Stride had already proposed the correct target **before the preceding hit's tag lookup completed**. A table trained to react to the last completed hit would enter this case after conventional address prediction had already happened. [Cohort counts and signed gaps](last_hit_deadlines.csv)

For 2,447 of these loads, the *same trigger load* produced a prefetch for the *same physical line* consumed by the *same demand load* in solo BFS-8, where it arrived on time. Median proposal-to-demand-ingress was 170 cycles in both runs; median proposal-to-dispatch was 20 in both. Median dispatch-to-fill rose from 70 to 190 cycles with LBM. Here the forward stream continued. A break in timely hits would be weak evidence of an address-pattern transition. The run with a competing workload isolates the co-runner's total effect; it does not independently manipulate the DRAM queue versus every other shared resource. [Matched service intervals](matched_service.csv)

| Same workload / different projection | First late loads just after ≥2 attributed timely hits | Correct target already proposed before the last hit's lookup | Prior hit outcome still unknown when the next load entered |
|---|---:|---:|---:|
| BFS-8 with LBM, core | 2,742 | 2,742 | 81 |
| MCF with CactuBSSN, core | 35 | 35 | 35 |
| MCF with CactuBSSN, within PC | 1,624 | 1,257 | 1,177 |
| MCF with CactuBSSN, within 4 KiB page | 1,254 | 1,254 | 1,177 |
| LBM with Xalancbmk, core | 17 | 17 | 8 |
| LBM with Xalancbmk, within PC | 921 | 844 | 82 |

All columns use the left-hand workload's Stride clock. The projection counts overlap and describe different temporal neighborhoods. A page/PC projection joins observations from the same page/PC after potentially many other demands. An attributed timely label requires a paired original NoPF miss and a tracked first-use hit in Stride. An ordinary hit or any other ingress outcome interrupts a run in this particular sensitivity test. Runs with later persistent misses and intervals with upstream demand paths are handled in the [sequence analysis](../episodes/).

For a concrete BFS example, mixed demand **1,000,774** entered and triggered line **42,567** at cycle 1,396,272. Its own LLC tag result confirmed the last timely hit 20 cycles later. The next demand, 1,000,853, entered at 1,396,533, merged at 1,396,553, and the target filled at 1,396,609. Alone the matched prefetch filled 191 cycles before its consumer's hit lookup. The same physical target and trigger worked in one execution and missed its deadline in the other. The measured mixed proposal-to-fill time also implies a diagnostic latest issue of 1,396,216 to meet the merge timestamp if service duration were held constant. Actual service after an earlier decision must be measured by replay. [Example timelines](examples.csv)

Run `python3 -B timing/compare.py` from the transition study after producing the previous study's paired per-core history records. For each completed request the program checks the identity of the target, matches producers and consumers across executions, and balances the elapsed issue, dispatch and fill intervals. The full table includes both directions and all four selected pairs, including zero-support projections. [Fixed protocol](protocol.md)
