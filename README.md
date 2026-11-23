# cache_arch

## gem5 neural prefetch design

This branch starts the gem5 study of a tiny learned predictor that plans dependent accesses beside conventional prefetchers. It holds a research design memo and a capture specification. No gem5 code or gem5 result exists yet.

[Design memo](gem5_neural_prefetch/README.md) · [Data-collection specification](gem5_neural_prefetch/DATA_COLLECTION.md) · [Prompt for the next step](gem5_neural_prefetch/NEXT_STEP_PROMPT.md)

## LLC prefetcher study

This branch studies when a shared LLC prefetcher can predict useful physical lines early enough to reduce execution time. Start with the full SPEC and GAP runs. Earlier studies cover workload screening, miss-to-hit transitions and conventional controls.

[Full-scale study](llc_hybrid_prefetch/full_workload/README.md) · [Figure gallery](llc_hybrid_prefetch/full_workload/figures/README.md) · [Trace and raw-data references](llc_hybrid_prefetch/full_workload/data/README.md) · [Earlier LLC experiments](llc_hybrid_prefetch/README.md)

## Earlier 602 studies

- **[Read the project overview and completed results](RESULTS_INDEX.md).** Four native prefetchers versus frozen H8/H16 NN students: IPC, accuracy, coverage, timeliness and Pareto trade-offs, with evidence and limits.
- [Current figures and CSVs](formal_study/results/602_width_templates/README.md) · [Experiment code and model contracts](formal_NN_training/experiments/) · [Layout and recovery](docs/602_width_templates/LAYOUT_AND_RECOVERY.md)
- The earlier 602 studies use `main` as their reading branch. Their overview lists the local and Sacramento paths for full runs, traces and builds.
