# Finding a harder SPEC CPU2017 trace

- **First choice for the offset question:** `620.omnetpp_s-874B`. Next-Line was the fastest conventional policy in both sampled regions. It saved about 3.3% of ROI time, left about 8 LLC misses per thousand instructions, and transferred about 44% more DRAM bytes than NoPF. The fixed offsets still reach a measurable set of misses at longer lead times. [SPEC workload description](https://spec.org/cpu2017/Docs/benchmarks/620.omnetpp_s.html)
- **Harder check:** `605.mcf_s-994B`. The fastest policy saved 2.24% in the first region and 1.26% in the second. The simple offset set reaches fewer misses when lead time is required. MCF uses the network simplex algorithm with pointer and integer operations. Keep it as a second workload for testing candidate generation. [SPEC workload description](https://spec.org/cpu2017/Docs/benchmarks/605.mcf_s.html)
- **Other controls:** All five issuing policies slowed down the sampled `619.lbm_s` region. AMPM cut MPKI from 11.100 to 4.151 and ran 6.04% longer. The sampled `623.xalancbmk_s` region has only 0.490 NoPF MPKI. Other Xalan regions can behave differently. [ChampSim SPEC trace archive](https://zenodo.org/records/10960004) · [Twilight, Fig. 8](https://www.cs.utexas.edu/~lin/papers/isca24.pdf)

## Conventional results on Sacramento

- Each run replayed the same trace on two cores with separate address spaces. All six policies used a 200,000-instruction warmup target, 1,000,000 ROI instructions per core, and at most one issued prefetch per eligible event. Private-cache prefetching was disabled. `10M` and `30M` denote initial trace records skipped. Both regions were examined for screening. Leave a new region or SimPoint untouched for any future final test.

| Trace | Skip | NoPF LLC MPKI | Fastest issuing policy | Its LLC MPKI | ROI time saved |
| --- | ---: | ---: | --- | ---: | ---: |
| `605.mcf_s-994B` | 10M | 9.709 | SPP | 8.815 | 2.24% |
| `605.mcf_s-994B` | 30M | 6.174 | Next-Line | 5.430 | 1.26% |
| `619.lbm_s-4268B` | 10M | 11.100 | Streamer | 9.898 | −0.32% |
| `620.omnetpp_s-874B` | 10M | 9.894 | Next-Line | 8.077 | 3.30% |
| `620.omnetpp_s-874B` | 30M | 10.000 | Next-Line | 8.220 | 3.38% |
| `623.xalancbmk_s-700B` | 10M | 0.490 | SPP | 0.344 | 0.57% |

- See the [620 baseline comparison](620_RESULTS.md) for each policy's accuracy, timeliness, coverage, and IPC in both regions.

## How many misses could the offsets reach?

- From the NoPF event logs, sort accepted demands by ingress order. For each trigger and same-page offset, look up the first later demand to that physical line. Count a target once if it LLC-misses and its ingress lead falls in the specified cycle range. Divide by all recorded NoPF demand misses.
- The table uses a 2,000-cycle maximum throughout. The lower bounds are sensitivity checks. Candidate reach assumes the target is eligible and there is room to issue. It has no service-latency calibration, one-issue limit, or interference model.

| Trace | Skip | `+1,+2,+4`, 0–2,000 cycles | `+1,+2,+4`, 300–2,000 | `+1,+2,+4`, 500–2,000 | `±1,±2,±4,±8,±16,±32`, 300–2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `605.mcf_s-994B` | 10M | 37.34% | 5.22% | 2.44% | 7.77% |
| `605.mcf_s-994B` | 30M | 38.44% | 6.54% | 3.00% | 10.03% |
| `620.omnetpp_s-874B` | 10M | 30.85% | 16.34% | 11.27% | 19.87% |
| `620.omnetpp_s-874B` | 30M | 30.60% | 15.89% | 10.68% | 19.18% |

- With the illustrative 300–2,000 cycle range, `+1` alone reaches 5.82% of the first OMNeT++ region's misses. `+2` and `+4` add 2,081 distinct misses, bringing the union to 16.34%. Use OMNeT++ to test whether history can choose among those offsets. On MCF, first measure whether a small temporal candidate table reaches more timely targets at the same degree.
- Raw `summary.csv`, per-core `summary.json`, simulator logs and NoPF demand CSVs are under `conventional/runs/spec_screen/` locally and on Sacramento. [Reproduction steps](README.md)
