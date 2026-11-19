# Full-workload pattern discovery and replay

[FINDINGS.md](FINDINGS.md) reports dynamic examples, late-to-timely conversions, displaced native offers, changes in timely usefulness, retained work and cross-phase generalization. [protocol.md](protocol.md) specifies scope, split, timing reference, legal actions, table budget and the late exploratory ablation. The native structural rule independently generates one physical-line address on eligible slots. It was frozen on two existing discovery prefixes before collection of the new results.

## Reproduction

Run from `llc_hybrid_prefetch/full_workload/discovery` with Python 3.10+ and NumPy. Prepare certified current inputs using [the registry and intake instructions](../intake/README.md). Build [the existing native conventional simulator](../../opportunity_study/conventional/conventional_baselines.md) and the [full-window instrumentation](../native/build.py). The three committed rule files suffice to run the frozen interventions. Rebuilding the donor fit, context-support counts, and historical comparisons also requires the two earlier discovery histories. Prepare those once, from this discovery directory after the conventional simulator build. [Earlier experiment and data provenance](../../workload_expansion/REPRODUCE.md)

```sh
(
  cd ../../workload_expansion
  python3 -B intake/fetch.py --jobs 4
  python3 -B screen/run.py --case spec_mcf graph_bfs3 --jobs 2
  python3 -B screen/history.py --case spec_mcf graph_bfs3 --jobs 2
)
```

```bash
python3 pattern.py
python3 build.py
python3 replay.py --jobs 2
python3 full_replay.py --build bfs_discovery bfs_transfer pr_heldout
python3 full_replay.py --build --policy sparse3 bfs_discovery
python3 full_replay.py mcf17_discovery mcf17_transfer mcf06_discovery soplex06_heldout pr_heldout_entire
python3 characterize.py --scope source
python3 characterize.py --scope diagnostic
python3 characterize.py --scope full
python3 failures.py --scope diagnostic
python3 failures.py --scope full
python3 opportunities.py --scope diagnostic
python3 opportunities.py --scope full
python3 phase.py
python3 ambiguity.py
python3 score_replay.py --scope diagnostic
python3 score_replay.py --scope full
python3 alternatives.py --scope diagnostic
python3 alternatives.py --scope full
python3 joint_targets.py
python3 whole_union.py
python3 fixed3.py
python3 registry.py
python3 report.py
python3 verify.py
```

The [intervention registry](run_registry.csv) gives one row and exact command per case and policy. The entire post-simulation pipeline can also be rerun with `python3 summarize.py`.

Use the [native runner](../native/run.py) for all comparator arms. A full-length replay joins every ROI outcome from the compressed per-instruction streams in bounded memory. `score_replay.py` separately audits physical candidate lifecycles inside sampled islands. The independently reset 500K runs log every operand and decision. Raw files and command metadata remain in `../raw/discovery/`. Scripts use only completed simulation markers when producing summaries; [verify.py](verify.py) checks the required final matrix.

## Reading the outputs

- `*_performance.csv` records native fixed-work execution, accepted physical requests, traffic and head exposure. The diagnostic version also checks all proposed addresses and records replacements, agreements and subsequent admission failures.
- `full_complete_load_alignment.csv` joins all dynamic operands across each full intervention, reference and NoPF. It partitions original timely uses into kept, gained and lost. `*_first_uses.csv` uses detailed event streams, with absolute counts from the four logged islands in a long run or the complete diagnostic in a short one.
- `*_native_examples.csv` uses actual attributed candidate events. `*_examples.csv` instead contains prefix-fitted offline projections.
- `*_transition_causes.csv` reports attributed or ordinary outcomes for all eligible original misses and separately for misses immediately after a timely use in the same PC/page/global projection. `*_populations.csv` gives denominators including never-timely streams. These statistics describe different projections of the same load and should not be summed across views.
- `phase_address_information.csv` tests whether additional producer history reduces held-out target log loss and counts locally ambiguous contexts, retaining inexpressible targets in the denominator. It evaluates the entire stream independently of first-use attribution.
- `*_opportunity_leads.csv` compares physical reach with the earliest grant on which each frozen rule actually named a target, the extra cycles relative to a Stride-late claim for the same dynamic load, and remaining time between first legal reach and identification. Sparse grant indices restart at each observed island for this offline projection; full simulated policy grants run continuously.
- `*_scores.csv` is the frozen two-donor comparison at sparse grant points. `phase_transfer.csv` reports the donor-quarter-to-later-quarter table experiment at three fixed horizons and every ingress. `phase_reach.csv` gives an address-vocabulary upper bound and per-target lead distributions. `phase_conditional_examples.csv` imposes the stronger requirement of distinct future displacements under matched local history and matched horizon. Frequency (votes and support), time and physical addresses accompany each example; no examples survive when only one outcome is observed.
- `full_joint_coverage.csv` joins every dynamic operand across NoPF, the hybrid, and all of Stride/Multi-delta/SPP/temporal replays, to count genuinely unique timely loads and the reverse losses. `heldout_lookahead_union.csv` also includes full native two-stride lookahead on GAP PageRank and Soplex. The union is retrospective, and the complete masks do not identify the producer of each hybrid timely use.
- `diagnostic_joint_coverage.csv` joins every original timely first use actually produced by the independent source across six initial and two post-hoc conventional controls. The survivor groups and dynamic examples are in `diagnostic_joint_patterns.csv` and `diagnostic_joint_examples.csv`; the group counts refer to the six-control cohort. `bfs_fixed3_witness.csv` tests the full-window BFS +3 producer and `bfs_fixed3_alignment.csv` counts all gains and displacements.
- Empty optional counts in a measured row are zero. Missing candidate lifecycle timestamps use -1. `first observed` is scoped to the logged interval. The whole-recording GAP PageRank and its initial 200M measured interval overlap. Report each separately.

All physical address arithmetic uses 64-byte lines and 64 lines/page. Exact hardware area, bank conflicts, table lookup timing, the coverage of arbitrary learned functions, and the optimality of the controls remain open.
