# cache_arch

## gem5 study of dependent-access prefetching

This branch holds the gem5 study of a small slow path that plans dependent accesses beside conventional prefetchers: a design memo, a validated capture built on gem5 26.0, and a first measurement study on GAP. The measurements say what stride and signature-path prefetching leave, what names those addresses, and how early they can be known. No prefetcher of ours has run yet, so there is no speedup result.

[Measurement findings](gem5_neural_prefetch/measurement/FINDINGS.md) · [Design memo](gem5_neural_prefetch/README.md) · [Data-collection specification](gem5_neural_prefetch/DATA_COLLECTION.md) · [How to build and run](gem5_neural_prefetch/sim/README.md) · [Prompt for the next step](gem5_neural_prefetch/NEXT_STEP_PROMPT.md)

## LLC prefetcher study

This branch studies when a shared LLC prefetcher can predict useful physical lines early enough to reduce execution time. Start with the full SPEC and GAP runs. Earlier studies cover workload screening, miss-to-hit transitions and conventional controls.

[Full-scale study](llc_hybrid_prefetch/full_workload/README.md) · [Figure gallery](llc_hybrid_prefetch/full_workload/figures/README.md) · [Trace and raw-data references](llc_hybrid_prefetch/full_workload/data/README.md) · [Earlier LLC experiments](llc_hybrid_prefetch/README.md)

## Earlier 602 studies

- **[Read the project overview and completed results](RESULTS_INDEX.md).** Four native prefetchers versus frozen H8/H16 NN students: IPC, accuracy, coverage, timeliness and Pareto trade-offs, with evidence and limits.
- [Current figures and CSVs](formal_study/results/602_width_templates/README.md) · [Experiment code and model contracts](formal_NN_training/experiments/) · [Layout and recovery](docs/602_width_templates/LAYOUT_AND_RECOVERY.md)
- The earlier 602 studies use `main` as their reading branch. Their overview lists the local and Sacramento paths for full runs, traces and builds.
