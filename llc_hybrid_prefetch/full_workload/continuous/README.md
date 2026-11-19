# Complete chronological LLC demand observations

This additional order audit uses BFS-3 and Soplex-92B, with a separate Stride execution for each. The 50M-instruction warmup, subsequent 200M retired instructions, physical placement, machine geometry and issuance rules match the primary runs. BFS-3 supplies a previously studied case. Soplex supplies a separately selected sparse solver. These two follow-up cases support conclusions about the effect of reordering and sampling on their observed sequences.

At each admitted LLC demand-data read in the measured interval, an output buffer records the source instruction ordinal and operand, demand PC, physical cache line and the core cycle. The raw format consists of four unsigned little-endian 64-bit fields in that order. The first field packs the ordinal in the high 62 bits and the operand in the low two bits. The scalar ordinal is one-based. File order preserves native admission order within cycle ties. The ordinal and operand identify the same dynamic load across executions. Stride uses its existing online inputs and has no path to the export or joined outcomes. Private hits, upstream merges, writebacks, instruction fetches and other events outside the LLC predictor interface cannot supply an observation.

Join each admission to the same NoPF and Stride retirement operand. Compare actual arrival order against sorting the same operands by instruction ordinal and operand. Count direct timely first uses, ordinary hits, late first merges, other tag misses and no completed tag lookup. The full sequence and sequence restricted to original NoPF LLC misses have separate denominators. The original-miss count absent at Stride ingress connects this study to the primary retirement-order census.

Four groupings follow the core, a demand PC, a 4 KiB physical page or a 256 KiB physical region. Every group has left- and right-censored boundaries at the observation limits. Consecutive events in a PC or region may be separated by arbitrarily many other observations. A hit run starts at a directly attributed timely first use. Ordinary tag hits may extend the run. A no-tag observation censors it, and a late or plain tag miss closes it. Quantiles report the distance from the most recent direct hit of that run to the closing miss. Rank differences count the distance in the underlying complete ingress sequence. Negative gaps in the program-order comparison identify actual reversals. These history gaps and offline last-hit markers are diagnostic labels.

The four fixed 250K-instruction islands use the same program-ordinal boundaries as the primary native study. Each starts with empty analysis-group state. Group totals in their aggregate count one instance per group and island. Instructions outside each island can interleave at its arrival-order edges. Event identity, PC, physical target, timestamp and ordering inside the islands must agree row-by-row with the original sampled event log. Full native cycles, pressure and lifecycle counters, per-retired-operand LLC outcome streams and physical maps must agree with the earlier Stride run.

Run from the cache root.

```sh
source .runtime/project_env.sh
python3 llc_hybrid_prefetch/full_workload/continuous/build.py
python3 llc_hybrid_prefetch/full_workload/continuous/check.py
python3 llc_hybrid_prefetch/full_workload/continuous/run_arrivals.py
python3 llc_hybrid_prefetch/full_workload/continuous/report.py
python3 llc_hybrid_prefetch/full_workload/continuous/compare_retirement.py
```

The directed fixture fixes eight dynamic operands in an order that reverses one pair, changes a direct-to-plain predecessor and preserves a never-timely PC. The [two full-run comparisons](FINDINGS.md) report actual denominators and censoring. The [three-way comparison](retirement_comparison.csv) also gives whole program order over every original miss, including targets absent at Stride ingress. The [underlying full-workload simulation](../native/README.md) describes the first-use and fixed-work instrumentation.
