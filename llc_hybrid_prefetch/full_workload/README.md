# Full-workload LLC prefetching evaluation

Can LLC-visible history generate a useful physical address early enough to improve execution? This study compares five conventional policies, then replays a few causal rules that can generate their own addresses.

The [learning opportunity study](../learning_opportunity/README.md) extends these runs with continuous histories, external recordings and data-driven prediction probes.

**Start with [Findings](FINDINGS.md)** for the results, [Figures](figures/README.md) for the annotated charts, or [Data](data/README.md) to locate an exact trace or raw run. [Reproduce](REPRODUCE.md) gives the build and run commands.

| Experiment | Measured work | Results |
|---|---|---|
| Conventional baselines | Nine 200M-instruction intervals from seven complete SPEC and GAP recordings | [Main comparison](native/BASELINE_FINDINGS.md) · [All policy metrics](native/performance.csv) · [Quarter results](native/quarters.csv) |
| PageRank through end of file | 450,000,001 measured instructions after 50M warmup | [Complete-recording results](native/entire/BASELINE_FINDINGS.md) · [Policy metrics](native/entire/performance.csv) |
| Independent address generation | Seven 200M-instruction intervals, one complete PageRank run, and shorter matched controls | [Discovery results and exact examples](discovery/FINDINGS.md) · [Run register](discovery/run_registry.csv) |
| Targeted replays | Cross-PC MCF, one BFS address, and two-stride lookahead on held-out programs | [MCF](streams/mcf_transfer/README.md) · [BFS](single_address/README.md) · [Lookahead](long_lookahead/README.md) |
| Robustness checks | Continuous arrival order, active-core interference, and a macOS/Linux repeat | [Arrival order](continuous/FINDINGS.md) · [Interference](interference/README.md) · [Cross-host checks](verification/README.md) |

The primary intervals each start with 50M warmup instructions and use the same two-core geometry with one active core. Far windows begin with a fresh warmup and a seeded physical page map. The complete PageRank run overlaps its 200M interval. These are instruction recordings, with no full-program weights. Soplex and the GAP PageRank executable were held out of fitting. [Trace selection and validation](intake/README.md) · [Simulator and metric definitions](native/README.md)

[Earlier LLC studies](../README.md)
