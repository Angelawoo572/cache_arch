# Find a trace, a dynamic load or a training example

All paths in the [data registry](data_registry.csv) are relative to the local cache root or `/home/qianruw/cache` on Sacramento. GitHub holds source registers, validation records, summaries and figure data. The local cache holds the complete recordings and generated events. A recording is the deposited program execution. A view is the portion warmed and measured. The source instruction count, starting ordinal, warmup and measured length have separate columns.

## Workspace verification

All 104 label indexes and 47 full-path captures passed their local checks. At the last confirmed Sacramento connection on November 20, file-level validation had returned for 13 original and two external full-path captures. The remaining site comparisons were pending when the connection closed. The [original replica table](capture/replicas.csv) and [external replica table](capture/external_replicas.csv) identify the validated runs. The [workspace checker](workspace.py) also compares each index's event columns, aligned outcomes, rank postings, retired work and source lengths. A registered path alone does not verify a second copy.

| Question | Open | What one row means |
|---|---|---|
| What did the source program load? | [Continuous capture](capture/README.md), `source_load` and the indexed native input | A load operand and core fetch time. The original instruction supplies its PC, virtual address and register fields. |
| Which requests arrived after private-cache filtering? | `cache`, `arrival` and `ingress_state` in the [capture schema](capture/README.md) | A cache callback, an LLC demand admission, or its contemporaneous pressure snapshot. Queue retries are distinct from tag lookups. |
| What could the LLC have observed at a given cue? | [Sequence and batch interfaces](labels/README.md) | A chronological prefix in one policy's execution. Equal-cycle rows retain callback order. |
| What happened to the same load under another policy? | [Aligned label indices](labels/datasets.csv) | The retired instruction ordinal and load-operand slot identify the load. Its physical address alone cannot distinguish repeated demands. |
| What was accessed later? | [Future labels](labels/README.md) | Exact physical lines and cycle gaps for global, PC, page and region queries. Query horizons or time windows separately. Missing successors are censored. |
| Was the physical address constructible and early enough? | [Reach, novelty and lead](labels/RESULTS.md) | A retrospective witness under a stated address vocabulary and service allowance. |
| Could a fitted predictor name the target before demand? | [Prediction probes](probes/README.md) | A held-out top address or abstention from a source-trained vocabulary. Every target requires both an encoding and a physical line. |
| Did offering the address change execution? | [Prospective forecast](one_offer/README.md) and [oracle timing interventions](trials/README.md) | Each treatment produces a new simulated demand stream and independent finish time. |

The original nine views have continuous source and cache instrumentation through all measured instructions. Each of the eleven external recordings has a complete Stride source and private-cache capture alongside the four native baselines. All remaining external policy or placement runs and both named BFS inputs retain continuous LLC admissions, complete per-operand outcomes and sampled lower-level request details. The registry identifies the capture tier, native comparison, rich sidecars and source of each label index on every row. Control-flow and architectural register fields originate at the core. Virtual pages do not reach the LLC as physical targets without a translation. Physical page assignments are simulated first-touch mappings. [Original source URLs and lengths](../full_workload/data/trace_files.csv) · [External trace register](expansion/sources.csv) · [Named graphs](graph_inputs/README.md)

On BFS-3, the full NoPF window contains 47,703,581 source load operands and 5,072,480 distinct LLC demand admissions. Only 10.63% of the source operands supply an LLC cue. A training example built from consecutive program loads would therefore answer a different hardware question. [Stream counts and source audit](capture/README.md)

The earlier PageRank run through EOF also has matched outcomes for 450,000,001 measured instructions. The new study indexes its 200M primary window. Counting both as independent test traces would reuse the first 200M measured instructions. [Through-EOF comparison](../full_workload/native/entire/performance.csv) · [Recording boundaries](../full_workload/intake/README.md)

## What counts as held out?

“Held out” is relative to a fit. Soplex-92B was an evaluation workload in the earlier architecture study. Here some explicitly named transfer experiments train on its first partition. The last partition of the destination recording supplies the evaluation cues. A different recording of the same executable is a recording transfer. A change of graph input is certified only for the two named SNAP runs. Reassigning simulated physical pages tests sensitivity to the allocator using the same source execution. Multiple views, policies, or placements of one recording do not create independent workloads. [Train/development/test protocol](probes/PROTOCOL.md) · [Physical layout comparison](capture/placement.csv)

## Rebuild a dataset

From the cache root, follow the [native baseline build](../full_workload/REPRODUCE.md) and [original source preparation](../full_workload/intake/README.md). Keep the original native outcome files so the rich capture can be checked against a separate run. These are the principal commands for one external view:

```sh
p=llc_hybrid_prefetch/learning_opportunity
python3 -B "$p/expansion/run.py" --cases cc5 --jobs 4 --decoders 1
python3 -B "$p/expansion/summarize.py" --cases cc5 --out "$p/raw/check_cc5"
python3 -B "$p/labels/index_all.py" cc5
python3 -B "$p/labels/process.py" cc5
```

The [capture](capture/README.md), [prediction](probes/README.md), [request timing](trials/README.md) and [plot](figures/README.md) directories give their own reproduction commands. Simulations use a C++ compiler and zlib. The analyses use NumPy, pandas, scikit-learn and Matplotlib. Reading a labeled example requires only NumPy and Python's standard library. The local analysis used Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2 and Matplotlib 3.10.6. The Sacramento label reconstruction used Python 3.11 and NumPy 1.26.4. The simulator build on macOS used GCC 15.2.

## Validation

The generalized native outcome join handles three or four active conventional policies. On the original BFS-3 case, the five-policy run exactly reproduced every entry of the previously published performance, joint-coverage, quarter, rejection and transition tables. The new four-policy path applies the same dynamic-operand and quarter checks. It validates the original-miss partition and equality of physical page maps in each comparison. [Native counting implementation](../full_workload/native/join.cc) · [Expansion outcomes](expansion/results/transitions.csv)

Linux and macOS enumerate the GCC-734B slot-1 page dictionary in different orders. All 9,112 virtual-to-physical mappings agree in all four policies. The native comparison checks each keyed mapping and rejects duplicate virtual pages. File order has no role in reconstructing a physical address.

The label test checks scopes, time cutoffs, simultaneous events and observations across chunk boundaries. The probe test changes a synthetic future, checks that the causal prefix stays identical, and tests chronological purging, page-rank decoding and first-observation labels. Native capture validation compares with the previous simulator and checks source PC, operand and physical-line joins. To repeat the dataset checks from the cache root:

```sh
p=llc_hybrid_prefetch/learning_opportunity
python3 -B "$p/labels/check.py"
python3 -B "$p/probes/check.py"
python3 -B "$p/probes/audit.py" --require-complete
python3 -B "$p/probes/verify_spp_clock.py"
python3 -B "$p/catalog.py"
python3 -B "$p/check.py"
```
