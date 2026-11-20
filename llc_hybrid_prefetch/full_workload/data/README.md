# Traces and raw results

Every path in the inventories is relative to the cache root. On the project Mac use `/Users/angelawoo/Documents/CMU/ece/cache/`. On Sacramento use `qianruw@sacramento.ece.local.cmu.edu:/home/qianruw/cache/`. GitHub holds the scripts, source URLs, validation records and compact results. The cache roots hold the original compressed traces, simulator inputs and event logs.

| I need… | Where to look |
|---|---|
| The source URL, retained archive and exact simulator input | [Trace files](trace_files.csv) maps each of the 10 simulator views to one of the seven complete source recordings. It gives byte counts, source span, physical-placement seed where applicable, and the validation report. |
| The raw output for a case and policy | [Run files](run_files.csv) lists 223 completed folders, measured work, the input trace paths, and files available in each folder. A vertical bar separates multiple paths or files in a cell. |
| Confirmation that the files are present | [Mac availability](local_availability.csv) · [Sacramento availability](sacramento_availability.csv). All indexed paths passed the presence check on both hosts. The check also compared the expected byte counts for compressed archives and decoded views. |
| The original experiment design or upstream source | [Trace provenance and validation](../intake/README.md) · [Source selection with public URLs](../intake/selection.csv) · [View boundaries](../intake/views.csv) · [Discovery commands and run status](../discovery/run_registry.csv). |

A trace view is a decoded instruction range. Each binary record is 64 bytes. The two far views also use a prefix VPN seed to preserve simulated physical placement. Most views supply a 50M warmup and 200M measured region. The `pr_heldout_entire` view includes all 500,000,001 source records.

## Follow a number back to its files

| Experiment | Compact input to the figures | Raw folder or folder pattern |
|---|---|---|
| Conventional windows and complete PageRank | [Policy metrics](../native/performance.csv) · [Complete PageRank metrics](../native/entire/performance.csv) · [Quarter metrics](../native/quarters.csv) · [Sampled timing](../native/sampled_timeliness.csv) | `llc_hybrid_prefetch/full_workload/raw/native/runs/<case>__<policy>/` |
| Frozen structural rule and address displacement | [Aligned first uses](../discovery/full_complete_load_alignment.csv) · [Full replay metrics](../discovery/full_performance.csv) · [Native witnesses](../discovery/full_native_examples.csv) | `llc_hybrid_prefetch/full_workload/raw/discovery/full/runs/<case>__<policy>/`. Short runs are in `raw/discovery/runs/<case>/<policy>/` under the study. |
| MCF cross-PC replay | [Matched performance](../streams/mcf_transfer/paired.csv) · [Target-level outcomes](../streams/mcf_transfer/target_details.csv) | `llc_hybrid_prefetch/full_workload/raw/mcf_arithmetic/runs/<case>__<policy>/` |
| Continuous arrivals | [Ordered counts](../continuous/counts.csv) · [Retirement comparison](../continuous/retirement_comparison.csv) | `llc_hybrid_prefetch/full_workload/raw/continuous/runs/<case>__stride/` contains `arrival.u64le.gz` and the normal simulator outputs. |
| BFS single-address and two-stride lookahead | [BFS exact changes](../single_address/results.csv) · [Lookahead results](../long_lookahead/paired.csv) | BFS overrides use `llc_hybrid_prefetch/full_workload/raw/single_address/{stride,spp}/`. Lookahead uses `raw/native/runs/<case>__look2/` under the study. |
| Two active cores | [Each core](../interference/per_core.csv) · [Shared system](../interference/shared_system.csv) | `llc_hybrid_prefetch/full_workload/raw/native/runs/interference_<placement>__<policy>/` |
| Cross-host PageRank repeat | [Cross-host result](../verification/linux_repeat.csv) | The independent run is in `raw/native/runs/pr_heldout__<policy>/` on Sacramento. The selected Linux outputs used for comparison are copied to `raw/linux_repeat/pr_heldout__<policy>/` on the Mac. Both paths are relative to the study. |

For example, open `llc_hybrid_prefetch/full_workload/raw/native/runs/bfs_discovery__spp/stats.csv` under either cache root to inspect the counters behind GAP BFS-3 start, SPP. In the same folder, `complete.json` records the simulator command and configuration, `cores.csv` reports retired work and cycles, and `phases.csv` contains quarter counters. The compressed outcome stream covers every measured instruction. The longer runs log detailed load, candidate, head-wait and victim events in four fixed sampling windows. [Native logging and definitions](../native/README.md)

## Earlier study inputs

These directories were present on both hosts. The earlier reports give the definitions, inputs and commands for their own measurements.

| Earlier study | Cache-relative paths | Provenance |
|---|---|---|
| Original SPEC comparisons, synthetic loop and interventions | `traces/`, `llc_hybrid_prefetch/raw/traces/`, `llc_hybrid_prefetch/raw/` | [Experiments](../../experiments/README.md) · [Measurements](../../MEASUREMENTS.md) |
| Architectural opportunities and conventional controls | `llc_hybrid_prefetch/opportunity_study/raw/traces/`, `llc_hybrid_prefetch/opportunity_study/raw/conventional/` | [Study guide](../../opportunity_study/README.md) |
| Miss-to-hit transitions | `llc_hybrid_prefetch/transition_study/raw/runs/` | [Inputs and replay](../../transition_study/README.md) |
| DPC4 and Google trace screening | `llc_hybrid_prefetch/workload_expansion/raw/intake/traces/`, `llc_hybrid_prefetch/workload_expansion/raw/screen/` | [Source registry](../../workload_expansion/intake/registry.csv) · [Acquisition record](../../workload_expansion/intake/acquisition.csv) |

Regenerate the inventories on the Mac with `python3 -B llc_hybrid_prefetch/full_workload/data/inventory.py`. Check listed paths with `python3 -B llc_hybrid_prefetch/full_workload/data/availability.py .` from either cache root. Use `--sacramento` there to resolve the independent Linux runs to their original location. For input recovery or full trace-integrity validation, see [Reproduce](../REPRODUCE.md).
