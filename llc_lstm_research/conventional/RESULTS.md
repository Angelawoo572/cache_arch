# Sacramento two-core 602 results

Both cores replayed `602.gcc_s-734B` after skipping 10 million trace records. Each used a 200,000-instruction warmup target and a 1,000,000-instruction ROI target. The shared issuer allowed one accepted prefetch per eligible demand. Simulated clock frequency was 4 GHz. [Reproduction steps](README.md)

| Policy | LLC hit rate | LLC miss rate | LLC MPKI | Prefetch timeliness | Net MPKI reduction | Aggregate IPC | ROI time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPF | 0.11% | 99.89% | 8.030 | — | 0.00% | 0.708 | 0.706 |
| Next-Line | 95.76% | 4.24% | 0.341 | 99.03% | 95.75% | 0.848 | 0.589 |
| Stride | 16.38% | 83.62% | 6.722 | 99.85% | 16.29% | 0.715 | 0.700 |
| Streamer | 91.92% | 8.08% | 0.649 | 99.41% | 91.91% | 0.840 | 0.596 |
| AMPM | 69.31% | 30.69% | 2.466 | 99.96% | 69.29% | 0.803 | 0.623 |
| SPP | 94.76% | 5.24% | 0.421 | 99.11% | 94.75% | 0.845 | 0.592 |

- Next-Line finished 16.5% sooner than NoPF. SPP came close at 0.592 ms and transferred 0.7% fewer DRAM bytes than Next-Line. Stride issued only 2,642 requests, so high timeliness coincided with modest net miss reduction.
- CMU 18-447 accuracy (used / sent) ranged from 97.3% to 99.4%. The resolved independent accuracy diagnostic was 100% for all five issuing policies. It excludes redundant and unresolved requests. [Metric definitions](../PERFORMANCE.md)
- Aggregate IPC is total ROI instructions divided by the last core's finish cycle. The JSON includes per-core IPC, DRAM bytes, prefetch slack, and queue counts. The runs measured net MPKI reduction. Causal coverage needs stable demand IDs across policies.
- These results describe one segment of the same workload on both cores. Conventional prefetches issue at ingress in this model. Later model comparisons must account for inference and queueing delay.

## What the development segment says about labels

- A separate NoPF run from the trace start recorded 16,178 completed demand lookups, including 16,144 misses. Most candidate offsets reach the same misses with different lead times. This is why a model that merely predicts the next address would have little room to help on this workload.

| Same-page offset | Unique future misses within 2,000 cycles | Additional misses in offset order | Median lead time (cycles, before applying the cap) |
| --- | ---: | ---: | ---: |
| `+1` | 15,068 | 15,068 | 334 |
| `+2` | 14,678 | 29 | 704 |
| `+4` | 13,865 | 6 | 1,388 |

- A sensitivity check with a 500-cycle minimum and the same 2,000-cycle maximum reduced the `+1` count to 1,035. The three offsets together still reached 14,802 distinct misses. The bounds illustrate the effect of latency and have yet to be calibrated. These counts assume any same-page match can be issued and provide an upper bound before filtering, bandwidth limits, and interference.
- Raw logs and metrics are in `runs/602_dual_10m/`. The separate NoPF demand CSV and log are in `runs/602_dual_development_events/`. Both directories are present locally and on Sacramento.
