# Active-core interference control

Selection frozen before measuring the pair. Use the lowest registered SPEC17 MCF and GAP BFS recordings. They represent two different programs in the panel. Assign physical placement slot 0 to MCF and slot 1 to BFS, including during isolated runs. Hold instruction start, cache geometry, predictor configuration, per-core work and DRAM timing fixed. Both workloads warm together in the pair. The isolated BFS uses logical core 0 but retains address-placement slot 1. Each address space has its own physical pages.

The diagnostic retires 5 million warmup and 20 million measured instructions per active core, with no skip. Cross NoPF, Stride, Multi-delta and SPP with three placements (each alone and both together). The primary 50M/200M panel is reported separately. The two intervals answer different questions and should not be pooled.

For each policy compute each workload's paired cycles divided by its own isolated cycles. Weighted speedup is the sum of their reciprocals. Also compare the paired makespan, DRAM requests, LLC ingress pressure and the change in timely and late demand attribution. All ratios use the same policy in the isolated denominator. A change in misses or lateness can coexist with a small change in completion time when useful work overlaps service. These results characterize one workload pairing.

From the repository root, with the validated traces in place and the native binary built:

```sh
python3 -B llc_hybrid_prefetch/full_workload/interference/run.py --jobs 2
python3 -B llc_hybrid_prefetch/full_workload/interference/measure.py
```

## What changed when the second core ran

| LLC policy | Paired completion cycles | MCF paired / solo time | BFS paired / solo time | Weighted speedup |
|---|---:|---:|---:|---:|
| NoPF | 115,936,972 | 1.192 | 1.122 | 1.730 |
| Stride | 95,558,354 | 1.323 | 1.127 | 1.643 |
| Multi-delta | 93,325,951 | 1.276 | 1.126 | 1.672 |
| SPP | 85,405,066 | 1.267 | 1.125 | 1.678 |

SPP completed both workloads earliest. MCF's paired Stride execution actually took 4,381,789 more cycles than paired NoPF, while BFS finished 20,378,618 cycles earlier. A single aggregate IPC would hide which core paid for that policy. Weighted speedup here compares each policy with its own solo runs, so the fastest system can have a smaller value than another policy.

With SPP, BFS had 295,176 dynamic operands that missed under both solo and paired NoPF. Among these, 17,474 loads changed from directly timely alone to a late merge with MCF present. Another 2,184 changed in the other direction. For the entire paired system, average LLC MSHR occupancy rose from 1.46 under NoPF to 3.23 with SPP. These are changes in the measured system with shared resources. They do not isolate a DRAM scheduling effect from changed cache residency or predictor input. [Per-core paired results](per_core.csv) · [Shared-system totals](shared_system.csv)
