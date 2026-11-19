# Following original demand misses through a full measurement interval

Join the same retired instruction and load operand across four policy arms. Verify per-program physical page maps across the arms. Retain every operand with a NoPF LLC tag miss, including those that no longer reach an LLC tag lookup under a policy. A timely conversion requires a lifecycle-attributed first use and a tag hit. An ordinary tag hit can be an indirect cache effect. A missing tag lookup can reflect private-cache filtering or a merge upstream.

For each policy group this stream three ways: across the core, by load PC, and by 256 KiB physical region. In each view count the first observed target and the next original target after a directly timely or nondirect outcome. Over the entire ROI report groups with zero timely conversions and the fraction of the original misses in those groups, both for all groups and those with at least 32 original misses. The latter cut provides a repeat-opportunity denominator. Changed PC, page crossing and intervening other-stream misses are overlapping descriptors. They are compared with consecutive timely successes at the same scope. A change of outcome alone cannot identify a source-code boundary.

These are retirement-order observations, available retrospectively even when the LLC arrival order changes. Use the detailed simulator ingress records in the discovery analysis to test actual causal cues and deadlines. The retirement-order sequence by itself cannot establish when a predictor could have issued an address.

Run after all four baselines for each published window finish:

```sh
python3 -B llc_hybrid_prefetch/full_workload/streams/measure.py
```

## Was the address genuinely new?

Use `python3 -B llc_hybrid_prefetch/full_workload/streams/audit_first_reference.py` after the lifecycle examples have been exported. The audit translates each sampled physical target back to its unique virtual cache line in this run and scans the original recording from instruction one, across warmup and logging gaps. It reports the earliest load or store of that line, including accesses that hit privately. An earliest use after the anchor but within the 256-entry ROB leaves ambiguity about out-of-order arrivals. An earliest use beyond anchor + 256 establishes that the line could not already have appeared as a demand from a younger in-flight instruction. The result concerns data references in this recording. A prior speculative proposal, instruction access or reference before the recording started is outside its scope. Far-view examples are excluded from this check because their materialization starts later. Repeated rows for one load identify its status under different conventional controls. [Checked examples](first_reference_examples.csv)
