# Can additional LLC-visible history generate useful physical addresses?

A complementary predictor must connect a causal observation to a legal address in time to help a particular load. This study tests the proposition against a PC stride tracker, Multi-delta, SPP and an exact temporal predictor, by fixing the number of retired instructions and joining the same demand operand across policies. The fitted controls can propose a new physical address themselves. Their decisions can also displace a useful Stride proposal. The primary performance measure is fixed-work completion time, with IPC, original-miss conversion, service timing and traffic as separate diagnostics.

## Reading the experiments

| Question | Artifact |
|---|---|
| Which recorded programs and phases represent the experiment? | [Trace provenance, source lengths and publication precedents](intake/README.md) · [Selected sources](intake/selection.csv) · [Native views](intake/views.csv) |
| What happened to all original LLC misses and processor throughput? | [Conventional performance and measurement definitions](native/README.md) · [Five-arm comparison](native/performance.csv) · [Whole PageRank recording](native/entire/BASELINE_FINDINGS.md) · [Per-quarter results](native/quarters.csv) · [Joint coverage](native/joint_coverage.csv) · [Held-out two-stride lookahead](long_lookahead/README.md) |
| How much of a stream does conventional prediction handle, including streams with zero direct hits? | [Whole-interval aligned episodes](streams/README.md) · [Per-stream counts](streams/group_counts.csv) · [Complete arrival-order census on BFS and Soplex](continuous/FINDINGS.md) · [Causal opportunity study](discovery/FINDINGS.md) |
| Can history generate a physical line a stronger conventional arm would miss or request late? | [Chronological splits and issue protocol](discovery/protocol.md) · [Real replay, displacement and load examples](discovery/FINDINGS.md) · [Independent first-reference audit](streams/first_reference_examples.csv) · [Single-address marginal replay](single_address/README.md) · [Address-increment controls](streams/address_relations.md) · [Full-length MCF cross-PC transfer](streams/mcf_transfer/README.md) |
| Does the conclusion change with another core in the system? | [Matched MCF and BFS co-runs](interference/README.md) |
| How can the study be repeated? | [Reproduction guide](REPRODUCE.md) · [Independent Linux check](verification/README.md) |

Seven complete source recordings cover SPEC2006 MCF and Soplex, SPEC2017 MCF, and GAP BFS and PageRank. The main panel contains nine 50M-warmup/200M-measurement windows, including disjoint far windows from two sources. Each is a fixed interval drawn from a complete recording. The additional PageRank pass executes the original 500,000,001 records from the first instruction through EOF, carrying cache and predictor state throughout its 450,000,001 measured instructions. Its first 200M measured instructions overlap the primary PageRank interval and must be counted once in any pooled analysis. None of the trace labels supplies an input weight or establishes that a recording spans the original application's whole lifetime.

Training and test have separate uses. Q1-to-Q3/Q4 tests phase transfer within a program. Separate MCF and BFS recordings test recording transfer. The Soplex and GAP PageRank executables were excluded from fitting. The older study had explored the PageRank algorithm under Ligra. GAP recording numbers establish different recordings; the artifact does not identify different underlying graphs. Counts over four chosen 250K-instruction logging islands are reported separately from the full-retirement population.

[Measured findings and research priorities](FINDINGS.md)
